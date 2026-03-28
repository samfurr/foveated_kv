#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Foveated 2-tier attention kernel
//
// Three logical tiers over one heterogeneous sequence:
//
//   Near  — fp16 K+V.  Sinks, window, recent middle, AND promoted tokens.
//           The near tier is "living": the C++ promotion worker writes fp16
//           K,V into headroom slots and atomically increments near_valid[h].
//           The kernel reads that count once per dispatch — promoted tokens
//           appear as ordinary near tokens with zero overhead.
//
//   Far   — fp8 E4M3 K + int4 per-token V.  Oldest middle context.
//           Asymmetric precision: K needs enough bits for score *ranking*
//           (which tokens matter), V precision is softmax-weighted (errors
//           on low-weight tokens vanish in the weighted sum).
//
//   Decode — fp16 K+V.  Tokens generated after compression.
//
// The kernel also feeds the promotion pipeline: it tracks the highest-
// scoring far token and the lowest-scoring near token per query head.
// When max_far > min_near + margin, spike_flags[bh_q] = 1 and
// spike_tokens[bh_q] = far-local index of the candidate. The C++ worker
// filters (cooldown, dedup, budget), reads fp16 from disk mmap, and
// writes directly into the blob's near headroom. Next dispatch, the
// kernel sees the promoted token — closed-loop, zero-copy.
//
// Performance techniques:
//   - Pre-scaled query: q *= 1/sqrt(D) at load time, amortizing one
//     multiply per token across the entire sequence
//   - Single-exp online softmax: each exp() computed exactly once
//   - FMA accumulation: rescale + accumulate in one pass over acc[]
//   - LUT fp8 decode: 256-entry tgmem table, 1 read vs 10+ ALU ops
//   - Vectorized loads: uint32 for fp8 K, uint16 for int4 V
//   - 4-token blocked far loop: ILP across independent K dot products
//   - Score-gated V: skip int4 dequant when exp(score - m) < 1e-7
//   - Branch-free near tier: loop bound = min(split_end, valid_count)
// ============================================================================

constant uint FC_N_NEAR     [[function_constant(0)]];
constant uint FC_N_FAR      [[function_constant(1)]];
constant uint FC_H_Q        [[function_constant(2)]];
constant uint FC_H_KV       [[function_constant(3)]];
constant uint FC_GQA_RATIO  [[function_constant(4)]];
constant uint FC_SPLIT_SIZE [[function_constant(5)]];
constant uint FC_NUM_SPLITS [[function_constant(6)]];

struct FoveatedParams {
    uint total_bh_q;
    uint n_decode;
    float spike_margin;
};

struct BlobOffsets {
    uint near_k;
    uint near_v;
    uint far_k;
    uint far_v;
    uint far_v_scale;
    uint far_v_zero;
    uint near_valid;
};


// ============================================================================
// Online softmax — single-exp, FMA accumulation
//
// Standard online softmax recomputes exp(score - m_new) redundantly.
// This version computes alpha and w once each, then rescales and
// accumulates acc[] in a single fused pass (half the iterations).
// ============================================================================

template <int CPT>
inline void softmax_accum(
    float score, thread float& m, thread float& l, thread float* acc,
    const thread float* v_vals)
{
    float m_new = max(m, score);
    float alpha = exp(m - m_new);       // rescale factor for old state
    float w     = exp(score - m_new);   // weight for new value — computed once
    l = fma(alpha, l, w);
    for (int c = 0; c < CPT; c++)
        acc[c] = fma(alpha, acc[c], w * v_vals[c]);
    m = m_new;
}


// ============================================================================
// fp16 K·Q + softmax accumulate — shared by near tier and decode buffer
//
// Both tiers store fp16 K and fp16 V with identical layout. This helper
// eliminates the code duplication between the two loops.
// ============================================================================

template <int CPT>
inline void attend_fp16(
    const thread float* q_reg,
    const device half* k_ptr, const device half* v_ptr,
    uint kv_off,
    thread float& m, thread float& l, thread float* acc,
    thread float& min_score)
{
    float dot = 0.0f;
    for (uint c = 0; c < CPT; c++)
        dot += q_reg[c] * (float)k_ptr[kv_off + c];
    float score = simd_sum(dot);    // q is pre-scaled — no per-token multiply
    min_score = min(min_score, score);

    float v_vals[CPT];
    for (uint c = 0; c < CPT; c++)
        v_vals[c] = (float)v_ptr[kv_off + c];
    softmax_accum<CPT>(score, m, l, acc, v_vals);
}


// ============================================================================
// Far K dot product — LUT fp8 decode + vectorized uint32 loads
// ============================================================================

template <int CPT, int K_GROUPS>
inline float dot_far_k(
    const thread float* q_reg,
    const device uint8_t* far_k,
    uint k_off,
    const threadgroup half* lut)
{
    float dot = 0.0f;
    if (K_GROUPS > 0) {
        // Vectorized: load 4 fp8 bytes as uint32, decode via LUT
        for (int g = 0; g < K_GROUPS; g++) {
            uint32_t packed = *(const device uint32_t*)(far_k + k_off + g * 4);
            dot += q_reg[g * 4 + 0] * (float)lut[packed & 0xFF];
            dot += q_reg[g * 4 + 1] * (float)lut[(packed >> 8) & 0xFF];
            dot += q_reg[g * 4 + 2] * (float)lut[(packed >> 16) & 0xFF];
            dot += q_reg[g * 4 + 3] * (float)lut[(packed >> 24) & 0xFF];
        }
    } else {
        // CPT < 4 (D=64, CPT=2): element-wise with LUT
        for (uint c = 0; c < CPT; c++)
            dot += q_reg[c] * (float)lut[far_k[k_off + c]];
    }
    return dot;
}


// ============================================================================
// int4 V dequantization — vectorized uint16 loads + FMA
// ============================================================================

template <int CPT>
inline void load_far_v(
    thread float* v_vals,
    const device uint8_t* far_v,
    uint v_off,
    float vs, float vz)
{
    if (CPT >= 4) {
        for (uint c = 0; c < CPT; c += 4) {
            uint16_t p16 = *(const device uint16_t*)(far_v + v_off + c / 2);
            v_vals[c]     = fma((float)(p16 & 0x0F), vs, vz);
            v_vals[c + 1] = fma((float)((p16 >> 4) & 0x0F), vs, vz);
            v_vals[c + 2] = fma((float)((p16 >> 8) & 0x0F), vs, vz);
            v_vals[c + 3] = fma((float)((p16 >> 12) & 0x0F), vs, vz);
        }
    } else {
        for (uint c = 0; c < CPT; c += 2) {
            uint8_t pb = far_v[v_off + c / 2];
            v_vals[c]     = fma((float)(pb & 0x0F), vs, vz);
            v_vals[c + 1] = fma((float)((pb >> 4) & 0x0F), vs, vz);
        }
    }
}


// ============================================================================
// Main kernel
// ============================================================================

template <typename QT, int HEAD_DIM, int MAX_SPLITS>
[[kernel, max_total_threads_per_threadgroup(MAX_SPLITS * 32)]]
void foveated_2tier(
    const device uint8_t* blob              [[buffer(0)]],
    const device QT* query                  [[buffer(1)]],
    const device half* decode_k             [[buffer(2)]],
    const device half* decode_v             [[buffer(3)]],
    device QT* out                          [[buffer(4)]],
    device int32_t* spike_flags             [[buffer(5)]],
    device int32_t* spike_tokens            [[buffer(6)]],
    const constant FoveatedParams& params   [[buffer(7)]],
    const constant BlobOffsets& offsets      [[buffer(8)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint thread_pos [[thread_position_in_threadgroup]])
{
    constexpr int CPT = HEAD_DIM / 32;
    constexpr int HEAD_DIM_HALF = HEAD_DIM / 2;
    constexpr int K_GROUPS = CPT / 4;

    // Score-gating threshold: skip V load when score < m - SCORE_SKIP.
    // exp(-16) ≈ 1.1e-7 — even summed over 8K tokens, worst-case
    // normalizer error is ~0.001, well below fp16 precision.
    const float SCORE_SKIP = 16.0f;

    const uint N_NEAR = FC_N_NEAR;
    const uint N_FAR  = FC_N_FAR;
    const uint H_Q = FC_H_Q;
    const uint H_KV = FC_H_KV;
    const uint GQA_RATIO = FC_GQA_RATIO;
    const uint SPLIT_SIZE = FC_SPLIT_SIZE;

    const uint N_DECODE = params.n_decode;
    const float SPIKE_MARGIN = params.spike_margin;

    // Unpack blob pointers
    auto near_k      = (const device half*)     (blob + offsets.near_k);
    auto near_v      = (const device half*)     (blob + offsets.near_v);
    auto far_k       = (const device uint8_t*)  (blob + offsets.far_k);
    auto far_v       = (const device uint8_t*)  (blob + offsets.far_v);
    auto far_v_scale = (const device half*)     (blob + offsets.far_v_scale);
    auto far_v_zero  = (const device half*)     (blob + offsets.far_v_zero);
    auto near_valid  = (const device uint32_t*) (blob + offsets.near_valid);

    const uint bh_q     = tg_pos;
    const uint simd_id  = thread_pos / 32;
    const uint lane_id  = thread_pos % 32;
    const uint split_id = simd_id;

    const uint batch_idx = bh_q / H_Q;
    const uint q_head    = bh_q % H_Q;
    const uint kv_head   = q_head / GQA_RATIO;
    const uint bh_kv     = batch_idx * H_KV + kv_head;

    // ================================================================
    // fp8 E4M3 → fp16 LUT in threadgroup memory (512 bytes)
    // ================================================================
    threadgroup half e4m3_lut[256];
    {
        const uint threads_per_tg = FC_NUM_SPLITS * 32;
        for (uint i = thread_pos; i < 256; i += threads_per_tg) {
            uint v = i;
            uint sign = v >> 7;
            uint exp8 = (v >> 3) & 0xF;
            uint mant = v & 0x7;
            if (exp8 == 0) {
                e4m3_lut[i] = as_type<half>((ushort)(sign << 15));
            } else {
                ushort fp16 = (ushort)((sign << 15) | ((exp8 + 8) << 10) | (mant << 7));
                e4m3_lut[i] = as_type<half>(fp16);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load query into registers, pre-scaled by 1/sqrt(D).
    // This amortizes the scaling cost: HEAD_DIM multiplies once vs
    // one multiply per token across the entire sequence.
    const float INV_SQRT_D = rsqrt((float)HEAD_DIM);
    float q_reg[CPT];
    for (uint c = 0; c < CPT; c++)
        q_reg[c] = (float)query[bh_q * HEAD_DIM + lane_id * CPT + c] * INV_SQRT_D;

    // Online softmax state
    float m = -INFINITY, l = 0.0f;
    float acc[CPT];
    for (uint c = 0; c < CPT; c++) acc[c] = 0.0f;

    // Spike detection state (reduced across splits at the end)
    float min_near_score = INFINITY, max_far_score = -INFINITY;
    int max_far_token = -1;

    // Split range computation
    const uint S_total = N_NEAR + N_FAR + N_DECODE;
    const uint gstart  = split_id * SPLIT_SIZE;
    const uint gend    = min(gstart + SPLIT_SIZE, S_total);

    const uint near_start = min(gstart, N_NEAR);
    const uint near_end   = min(gend,   N_NEAR);
    const uint far_start  = (gstart > N_NEAR) ? min(gstart - N_NEAR, N_FAR) : 0u;
    const uint far_end    = (gend   > N_NEAR) ? min(gend   - N_NEAR, N_FAR) : 0u;
    const uint nf = N_NEAR + N_FAR;
    const uint dec_start  = (gstart > nf) ? min(gstart - nf, N_DECODE) : 0u;
    const uint dec_end    = (gend   > nf) ? min(gend   - nf, N_DECODE) : 0u;

    // ==== NEAR (fp16 K+V — living tier) ====
    // Loop bound uses min(split_end, near_valid[h]) — headroom slots
    // beyond valid count are never read. As the C++ worker promotes
    // tokens, near_valid[h] grows and the kernel sees them next dispatch.
    const uint near_valid_count = near_valid[kv_head];
    const uint near_loop_end = min(near_end, near_valid_count);
    const uint near_kv_base = bh_kv * N_NEAR * HEAD_DIM;
    for (uint t = near_start; t < near_loop_end; t++) {
        attend_fp16<CPT>(q_reg, near_k, near_v,
                         near_kv_base + t * HEAD_DIM + lane_id * CPT,
                         m, l, acc, min_near_score);
    }

    // ==== FAR (fp8 E4M3 K + int4 per-token V — compressed tier) ====
    //
    // 4-token blocked for instruction-level parallelism:
    //   Phase 1: 4 independent K dot products (GPU pipelines the loads)
    //   Phase 2: 4 simd_sum reductions + spike tracking
    //   Phase 3: score-gated V dequant + softmax accumulate
    const uint far_k_base  = bh_kv * N_FAR * HEAD_DIM;
    const uint far_v_base  = bh_kv * N_FAR * HEAD_DIM_HALF;
    const uint far_vs_base = bh_kv * N_FAR;

    const uint far_count = far_end - far_start;
    const uint far_end_4 = far_start + (far_count / 4) * 4;

    for (uint tb = far_start; tb < far_end_4; tb += 4) {
        float dots[4];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) {
            uint k_off = far_k_base + (tb + (uint)i) * HEAD_DIM + lane_id * CPT;
            dots[i] = dot_far_k<CPT, K_GROUPS>(q_reg, far_k, k_off, e4m3_lut);
        }

        float scores[4];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) {
            scores[i] = simd_sum(dots[i]);
            uint ti = tb + (uint)i;
            if (scores[i] > max_far_score) {
                max_far_score = scores[i];
                max_far_token = (int)ti;
            }
        }

        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; i++) {
            if (scores[i] < m - SCORE_SKIP) continue;

            uint ti = tb + (uint)i;
            float vs = (float)far_v_scale[far_vs_base + ti];
            float vz = (float)far_v_zero[far_vs_base + ti];
            uint v_off = far_v_base + ti * HEAD_DIM_HALF + lane_id * CPT / 2;
            float v_vals[CPT];
            load_far_v<CPT>(v_vals, far_v, v_off, vs, vz);
            softmax_accum<CPT>(scores[i], m, l, acc, v_vals);
        }
    }

    // Tail: 0-3 remaining far tokens
    for (uint t = far_end_4; t < far_end; t++) {
        uint k_off = far_k_base + t * HEAD_DIM + lane_id * CPT;
        float dot = dot_far_k<CPT, K_GROUPS>(q_reg, far_k, k_off, e4m3_lut);
        float score = simd_sum(dot);

        if (score > max_far_score) { max_far_score = score; max_far_token = (int)t; }

        if (score >= m - SCORE_SKIP) {
            float vs = (float)far_v_scale[far_vs_base + t];
            float vz = (float)far_v_zero[far_vs_base + t];
            uint v_off = far_v_base + t * HEAD_DIM_HALF + lane_id * CPT / 2;
            float v_vals[CPT];
            load_far_v<CPT>(v_vals, far_v, v_off, vs, vz);
            softmax_accum<CPT>(score, m, l, acc, v_vals);
        }
    }

    // ==== DECODE (fp16 K+V — tokens generated after compression) ====
    // Same format as near tier — uses the shared attend_fp16 helper.
    // Decode scores contribute to min_near_score for spike detection
    // (they're full-precision and shouldn't trigger promotions).
    const uint dec_kv_base = bh_kv * N_DECODE * HEAD_DIM;
    for (uint t = dec_start; t < dec_end; t++) {
        attend_fp16<CPT>(q_reg, decode_k, decode_v,
                         dec_kv_base + t * HEAD_DIM + lane_id * CPT,
                         m, l, acc, min_near_score);
    }

    // ==== SPLIT REDUCE ====
    // Merge partial softmax results across splits using the same
    // rescaling technique: alpha = exp(m_old - m_new).
    threadgroup float shared_acc[MAX_SPLITS * HEAD_DIM];
    threadgroup float shared_l[MAX_SPLITS];
    threadgroup float shared_m[MAX_SPLITS];
    threadgroup float shared_min_near[MAX_SPLITS];
    threadgroup float shared_max_far[MAX_SPLITS];
    threadgroup int   shared_far_tok[MAX_SPLITS];

    for (uint c = 0; c < CPT; c++)
        shared_acc[split_id * HEAD_DIM + lane_id * CPT + c] = acc[c];
    if (lane_id == 0) {
        shared_l[split_id]         = l;
        shared_m[split_id]         = m;
        shared_min_near[split_id]  = min_near_score;
        shared_max_far[split_id]   = max_far_score;
        shared_far_tok[split_id]   = max_far_token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float m_global = -INFINITY, l_global = 0.0f;
        float racc[CPT];
        for (uint c = 0; c < CPT; c++) racc[c] = 0.0f;
        float min_near = INFINITY, max_far = -INFINITY;
        int far_token = -1;

        for (uint s = 0; s < FC_NUM_SPLITS; s++) {
            float m_s = shared_m[s];
            if (m_s == -INFINITY) continue;
            float l_s = shared_l[s];
            float m_new = max(m_global, m_s);
            float ag = (m_global > -INFINITY) ? exp(m_global - m_new) : 0.0f;
            float as_val = exp(m_s - m_new);
            l_global = fma(ag, l_global, as_val * l_s);
            for (uint c = 0; c < CPT; c++) {
                float pv = shared_acc[s * HEAD_DIM + lane_id * CPT + c];
                racc[c] = fma(ag, racc[c], as_val * pv);
            }
            m_global = m_new;
            min_near = min(min_near, shared_min_near[s]);
            float fs = shared_max_far[s];
            if (fs > max_far) { max_far = fs; far_token = shared_far_tok[s]; }
        }

        float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
        for (uint c = 0; c < CPT; c++)
            out[bh_q * HEAD_DIM + lane_id * CPT + c] = (QT)(racc[c] * inv_l);

        // Spike output: feeds the C++ promotion pipeline.
        // The pipeline filters via cooldown/dedup/budget, reads fp16 from
        // disk mmap, and writes into the blob's near headroom — closing the
        // loop for the next dispatch.
        if (lane_id == 0) {
            spike_flags[bh_q]  = (max_far > min_near + SPIKE_MARGIN) ? 1 : 0;
            spike_tokens[bh_q] = far_token;
        }
    }
}


// ============================================================================
// Template instantiations
// ============================================================================

#define INST(QT, QN, D, S) \
template [[host_name("foveated_2tier_" #QN "_d" #D "_s" #S)]] \
kernel void foveated_2tier<QT, D, S>( \
    const device uint8_t*, const device QT*, const device half*, const device half*, \
    device QT*, device int32_t*, device int32_t*, \
    const constant FoveatedParams&, const constant BlobOffsets&, uint, uint);

INST(half, f16, 64, 1) INST(half, f16, 64, 2) INST(half, f16, 64, 4) INST(half, f16, 64, 8) INST(half, f16, 64, 16)
INST(half, f16, 128, 1) INST(half, f16, 128, 2) INST(half, f16, 128, 4) INST(half, f16, 128, 8) INST(half, f16, 128, 16)
INST(bfloat, bf16, 64, 1) INST(bfloat, bf16, 64, 2) INST(bfloat, bf16, 64, 4) INST(bfloat, bf16, 64, 8) INST(bfloat, bf16, 64, 16)
INST(bfloat, bf16, 128, 1) INST(bfloat, bf16, 128, 2) INST(bfloat, bf16, 128, 4) INST(bfloat, bf16, 128, 8) INST(bfloat, bf16, 128, 16)
#undef INST


// ============================================================================
// TurboQuant 2-tier attention kernel
//
// Far tier uses TurboQuant: 2-bit Lloyd-Max key indices + 1-bit QJL signs
// + 2-bit symmetric group-quantized values.
//
// Score computation avoids full dequantization:
//   score = norm_k * dot(q_rot, centroids[indices])
//         + sqrt(pi/2)/D * gamma * dot(signs, q_sketch)
//
// where q_rot = q @ Pi^T and q_sketch = S @ q are precomputed once per query.
//
// 4-entry codebook fits in registers (no threadgroup LUT needed).
// Sign dot is conditional negate + accumulate.
// ============================================================================

struct TurboBlobOffsets {
    uint near_k;
    uint near_v;
    uint far_k_indices;   // packed 2-bit Lloyd-Max (D/4 bytes per token)
    uint far_k_signs;     // packed 1-bit QJL signs (D/8 bytes per token)
    uint far_k_norm;      // fp16 key norms
    uint far_k_gamma;     // fp16 residual norms
    uint far_v_packed;    // packed 2-bit values (D/4 bytes per token)
    uint far_v_scale;     // fp16 per-group scales (D/32 per token)
    uint near_valid;
};


// ============================================================================
// TurboQuant far K score — codebook dot + QJL sign dot
// ============================================================================

template <int CPT>
inline float dot_turbo_k(
    const thread float* q_rot,       // pre-rotated query fragment
    const thread float* q_sketch,    // pre-sketched query fragment
    const device uint8_t* indices,   // packed 2-bit, D/4 bytes per token
    uint idx_off,
    const device uint8_t* signs,     // packed 1-bit, D/8 bytes per token
    uint sign_off,
    float norm_k,
    float gamma,
    const thread float* cb,          // 4 centroids in registers
    uint D,
    uint lid)                        // lane_id
{
    // MSE score: norm_k * sum_j(q_rot[j] * centroids[indices[j]])
    float dot_mse = 0.0f;
    for (uint c = 0; c < CPT; c++) {
        uint byte_idx = (lid * CPT + c) / 4;
        uint bit_shift = ((lid * CPT + c) % 4) * 2;
        uint idx = (indices[idx_off + byte_idx] >> bit_shift) & 0x3;
        dot_mse += q_rot[c] * cb[idx];
    }
    float score_mse = norm_k * simd_sum(dot_mse);

    // QJL score: sqrt(pi/2)/D * gamma * sum_j(sign[j] * q_sketch[j])
    float dot_sign = 0.0f;
    for (uint c = 0; c < CPT; c++) {
        uint byte_idx = (lid * CPT + c) / 8;
        uint bit_idx = (lid * CPT + c) % 8;
        float s = ((signs[sign_off + byte_idx] >> bit_idx) & 1) ? 1.0f : -1.0f;
        dot_sign += s * q_sketch[c];
    }
    float score_qjl = (1.2533141f / (float)D) * gamma * simd_sum(dot_sign);

    return score_mse + score_qjl;
}


// ============================================================================
// TurboQuant 2-bit V dequantization — symmetric group quantization
//
// 4 levels: {0,1,2,3} → {-1, -1/3, 1/3, 1} * group_scale
// Group size 32, so D/32 scales per token.
// ============================================================================

constant float TURBO_V_LEVELS[4] = {-1.0f, -0.333333f, 0.333333f, 1.0f};

template <int CPT>
inline void load_turbo_v(
    thread float* v_vals,
    const device uint8_t* packed,
    uint v_off,
    const device half* scales,
    uint scale_off,
    uint group_size,
    uint lid)                       // lane_id
{
    for (uint c = 0; c < CPT; c++) {
        uint global_dim = lid * CPT + c;
        uint byte_idx = global_dim / 4;
        uint bit_shift = (global_dim % 4) * 2;
        uint idx = (packed[v_off + byte_idx] >> bit_shift) & 0x3;
        uint group_idx = global_dim / group_size;
        float scale = (float)scales[scale_off + group_idx];
        v_vals[c] = TURBO_V_LEVELS[idx] * scale;
    }
}


// ============================================================================
// TurboQuant main kernel
// ============================================================================

template <typename QT, int HEAD_DIM, int MAX_SPLITS>
[[kernel, max_total_threads_per_threadgroup(MAX_SPLITS * 32)]]
void foveated_2tier_turbo(
    const device uint8_t* blob              [[buffer(0)]],
    const device QT* query                  [[buffer(1)]],
    const device half* decode_k             [[buffer(2)]],
    const device half* decode_v             [[buffer(3)]],
    device QT* out                          [[buffer(4)]],
    device int32_t* spike_flags             [[buffer(5)]],
    device int32_t* spike_tokens            [[buffer(6)]],
    const constant FoveatedParams& params   [[buffer(7)]],
    const constant TurboBlobOffsets& offsets [[buffer(8)]],
    const device float* q_rot_buf           [[buffer(9)]],
    const device float* q_sketch_buf        [[buffer(10)]],
    const constant float* centroids         [[buffer(11)]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint thread_pos [[thread_position_in_threadgroup]])
{
    constexpr int CPT = HEAD_DIM / 32;
    constexpr int HEAD_DIM_QUARTER = HEAD_DIM / 4;
    constexpr int HEAD_DIM_EIGHTH = HEAD_DIM / 8;
    constexpr int V_GROUP_SIZE = 32;
    constexpr int V_GROUPS_PER_TOKEN = HEAD_DIM / V_GROUP_SIZE;

    const float SCORE_SKIP = 16.0f;

    const uint N_NEAR = FC_N_NEAR;
    const uint N_FAR  = FC_N_FAR;
    const uint H_Q = FC_H_Q;
    const uint H_KV = FC_H_KV;
    const uint GQA_RATIO = FC_GQA_RATIO;
    const uint SPLIT_SIZE = FC_SPLIT_SIZE;

    const uint N_DECODE = params.n_decode;
    const float SPIKE_MARGIN = params.spike_margin;

    // Unpack blob pointers
    auto near_k      = (const device half*)     (blob + offsets.near_k);
    auto near_v      = (const device half*)     (blob + offsets.near_v);
    auto far_idx     = (const device uint8_t*)  (blob + offsets.far_k_indices);
    auto far_signs   = (const device uint8_t*)  (blob + offsets.far_k_signs);
    auto far_k_norm  = (const device half*)     (blob + offsets.far_k_norm);
    auto far_k_gamma = (const device half*)     (blob + offsets.far_k_gamma);
    auto far_v_pack  = (const device uint8_t*)  (blob + offsets.far_v_packed);
    auto far_v_scale = (const device half*)     (blob + offsets.far_v_scale);
    auto near_valid  = (const device uint32_t*) (blob + offsets.near_valid);

    const uint bh_q     = tg_pos;
    const uint simd_id  = thread_pos / 32;
    const uint lane_id  = thread_pos % 32;
    const uint split_id = simd_id;

    const uint batch_idx = bh_q / H_Q;
    const uint q_head    = bh_q % H_Q;
    const uint kv_head   = q_head / GQA_RATIO;
    const uint bh_kv     = batch_idx * H_KV + kv_head;

    // ================================================================
    // Load query, pre-rotated q_rot, and pre-sketched q_sketch.
    // q_rot = q @ Pi^T and q_sketch = S @ q are pre-computed in C++
    // via MLX matmul (pipelines well, no simd_shuffle needed).
    // ================================================================
    const float INV_SQRT_D = rsqrt((float)HEAD_DIM);

    // q_reg = q * 1/sqrt(D) for near/decode attention
    float q_reg[CPT];
    for (uint c = 0; c < CPT; c++)
        q_reg[c] = (float)query[bh_q * HEAD_DIM + lane_id * CPT + c] * INV_SQRT_D;

    // Load pre-computed q_rot and q_sketch from buffers
    float q_rot[CPT];
    for (uint c = 0; c < CPT; c++)
        q_rot[c] = q_rot_buf[bh_q * HEAD_DIM + lane_id * CPT + c];

    float q_sketch[CPT];
    for (uint c = 0; c < CPT; c++)
        q_sketch[c] = q_sketch_buf[bh_q * HEAD_DIM + lane_id * CPT + c];

    // Load centroids into registers (only 4 values)
    float cb[4];
    cb[0] = centroids[0]; cb[1] = centroids[1];
    cb[2] = centroids[2]; cb[3] = centroids[3];

    // Online softmax state
    float m_val = -INFINITY, l_val = 0.0f;
    float acc[CPT];
    for (uint c = 0; c < CPT; c++) acc[c] = 0.0f;

    float min_near_score = INFINITY, max_far_score = -INFINITY;
    int max_far_token = -1;

    // Split range computation (identical to fp8 kernel)
    const uint S_total = N_NEAR + N_FAR + N_DECODE;
    const uint gstart  = split_id * SPLIT_SIZE;
    const uint gend    = min(gstart + SPLIT_SIZE, S_total);

    const uint near_start = min(gstart, N_NEAR);
    const uint near_end   = min(gend,   N_NEAR);
    const uint far_start  = (gstart > N_NEAR) ? min(gstart - N_NEAR, N_FAR) : 0u;
    const uint far_end    = (gend   > N_NEAR) ? min(gend   - N_NEAR, N_FAR) : 0u;
    const uint nf = N_NEAR + N_FAR;
    const uint dec_start  = (gstart > nf) ? min(gstart - nf, N_DECODE) : 0u;
    const uint dec_end    = (gend   > nf) ? min(gend   - nf, N_DECODE) : 0u;

    // ==== NEAR (fp16 K+V — identical to fp8 kernel) ====
    const uint near_valid_count = near_valid[kv_head];
    const uint near_loop_end = min(near_end, near_valid_count);
    const uint near_kv_base = bh_kv * N_NEAR * HEAD_DIM;
    for (uint t = near_start; t < near_loop_end; t++) {
        attend_fp16<CPT>(q_reg, near_k, near_v,
                         near_kv_base + t * HEAD_DIM + lane_id * CPT,
                         m_val, l_val, acc, min_near_score);
    }

    // ==== FAR (TurboQuant K + 2-bit V) ====
    const uint far_idx_base   = bh_kv * N_FAR * HEAD_DIM_QUARTER;
    const uint far_sign_base  = bh_kv * N_FAR * HEAD_DIM_EIGHTH;
    const uint far_norm_base  = bh_kv * N_FAR;
    const uint far_vp_base    = bh_kv * N_FAR * HEAD_DIM_QUARTER;
    const uint far_vs_base    = bh_kv * N_FAR * V_GROUPS_PER_TOKEN;

    for (uint t = far_start; t < far_end; t++) {
        float nk = (float)far_k_norm[far_norm_base + t];
        float gm = (float)far_k_gamma[far_norm_base + t];

        float score = dot_turbo_k<CPT>(
            q_rot, q_sketch,
            far_idx, far_idx_base + t * HEAD_DIM_QUARTER,
            far_signs, far_sign_base + t * HEAD_DIM_EIGHTH,
            nk, gm, cb, HEAD_DIM, lane_id) * INV_SQRT_D;

        if (score > max_far_score) {
            max_far_score = score;
            max_far_token = (int)t;
        }

        if (score >= m_val - SCORE_SKIP) {
            float v_vals[CPT];
            load_turbo_v<CPT>(v_vals, far_v_pack,
                              far_vp_base + t * HEAD_DIM_QUARTER,
                              far_v_scale,
                              far_vs_base + t * V_GROUPS_PER_TOKEN,
                              V_GROUP_SIZE, lane_id);
            softmax_accum<CPT>(score, m_val, l_val, acc, v_vals);
        }
    }

    // ==== DECODE (fp16 K+V — identical to fp8 kernel) ====
    const uint dec_kv_base = bh_kv * N_DECODE * HEAD_DIM;
    for (uint t = dec_start; t < dec_end; t++) {
        attend_fp16<CPT>(q_reg, decode_k, decode_v,
                         dec_kv_base + t * HEAD_DIM + lane_id * CPT,
                         m_val, l_val, acc, min_near_score);
    }

    // ==== SPLIT REDUCE (identical to fp8 kernel) ====
    threadgroup float shared_acc[MAX_SPLITS * HEAD_DIM];
    threadgroup float shared_l[MAX_SPLITS];
    threadgroup float shared_m[MAX_SPLITS];
    threadgroup float shared_min_near[MAX_SPLITS];
    threadgroup float shared_max_far[MAX_SPLITS];
    threadgroup int   shared_far_tok[MAX_SPLITS];

    for (uint c = 0; c < CPT; c++)
        shared_acc[split_id * HEAD_DIM + lane_id * CPT + c] = acc[c];
    if (lane_id == 0) {
        shared_l[split_id]         = l_val;
        shared_m[split_id]         = m_val;
        shared_min_near[split_id]  = min_near_score;
        shared_max_far[split_id]   = max_far_score;
        shared_far_tok[split_id]   = max_far_token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float m_global = -INFINITY, l_global = 0.0f;
        float racc[CPT];
        for (uint c = 0; c < CPT; c++) racc[c] = 0.0f;
        float min_near = INFINITY, max_far = -INFINITY;
        int far_token = -1;

        for (uint s = 0; s < FC_NUM_SPLITS; s++) {
            float m_s = shared_m[s];
            if (m_s == -INFINITY) continue;
            float l_s = shared_l[s];
            float m_new = max(m_global, m_s);
            float ag = (m_global > -INFINITY) ? exp(m_global - m_new) : 0.0f;
            float as_val = exp(m_s - m_new);
            l_global = fma(ag, l_global, as_val * l_s);
            for (uint c = 0; c < CPT; c++) {
                float pv = shared_acc[s * HEAD_DIM + lane_id * CPT + c];
                racc[c] = fma(ag, racc[c], as_val * pv);
            }
            m_global = m_new;
            min_near = min(min_near, shared_min_near[s]);
            float fs = shared_max_far[s];
            if (fs > max_far) { max_far = fs; far_token = shared_far_tok[s]; }
        }

        float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
        for (uint c = 0; c < CPT; c++)
            out[bh_q * HEAD_DIM + lane_id * CPT + c] = (QT)(racc[c] * inv_l);

        if (lane_id == 0) {
            spike_flags[bh_q]  = (max_far > min_near + SPIKE_MARGIN) ? 1 : 0;
            spike_tokens[bh_q] = far_token;
        }
    }
}


// TurboQuant kernel instantiations
#define INST_TURBO(QT, QN, D, S) \
template [[host_name("foveated_2tier_turbo_" #QN "_d" #D "_s" #S)]] \
kernel void foveated_2tier_turbo<QT, D, S>( \
    const device uint8_t*, const device QT*, const device half*, const device half*, \
    device QT*, device int32_t*, device int32_t*, \
    const constant FoveatedParams&, const constant TurboBlobOffsets&, \
    const device float*, const device float*, const constant float*, uint, uint);

INST_TURBO(half, f16, 64, 1) INST_TURBO(half, f16, 64, 2) INST_TURBO(half, f16, 64, 4) INST_TURBO(half, f16, 64, 8) INST_TURBO(half, f16, 64, 16)
INST_TURBO(half, f16, 128, 1) INST_TURBO(half, f16, 128, 2) INST_TURBO(half, f16, 128, 4) INST_TURBO(half, f16, 128, 8) INST_TURBO(half, f16, 128, 16)
INST_TURBO(bfloat, bf16, 64, 1) INST_TURBO(bfloat, bf16, 64, 2) INST_TURBO(bfloat, bf16, 64, 4) INST_TURBO(bfloat, bf16, 64, 8) INST_TURBO(bfloat, bf16, 64, 16)
INST_TURBO(bfloat, bf16, 128, 1) INST_TURBO(bfloat, bf16, 128, 2) INST_TURBO(bfloat, bf16, 128, 4) INST_TURBO(bfloat, bf16, 128, 8) INST_TURBO(bfloat, bf16, 128, 16)
#undef INST_TURBO
