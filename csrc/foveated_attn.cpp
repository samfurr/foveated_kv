// Phase 1: Use mlx::core::fast::metal_kernel from C++ to build the kernel
// callable. This eliminates the Python dispatch overhead while reusing MLX's
// proven kernel compilation and dispatch machinery.
//
// The key win: one C++ function call per layer instead of:
//   Python interceptor → _dispatch_kernel → mx.fast.metal_kernel × 2

#include "foveated_attn.h"

#include <cmath>
#include <sstream>

#include "mlx/fast.h"

namespace foveated {

using namespace mlx::core;

// ---------------------------------------------------------------------------
// Metal kernel source strings — transplanted from metal_foveated.py
// ---------------------------------------------------------------------------

static const char* TIER_PROCESSING = R"METAL(
    // ==== FOVEAL (fp16 K + fp16 V, with padding mask) ====
    uint fov_valid = foveal_valid[kv_head];
    uint fov_kv_base = bh_kv * N_FOV * HEAD_DIM;
    for (uint t = fov_start; t < fov_end; t++) {
        if (t >= fov_valid) continue;
        float dot = 0.0f;
        for (uint c = 0; c < CPT; c++)
            dot += q_reg[c] * (float)foveal_k[fov_kv_base + t * HEAD_DIM + lane_id * CPT + c];
        float score = simd_sum(dot) * INV_SQRT_D;
        min_fov_score = min(min_fov_score, score);

        float m_new = max(m, score);
        float alpha = exp(m - m_new);
        for (uint c = 0; c < CPT; c++) acc[c] *= alpha;
        l = alpha * l + exp(score - m_new);
        m = m_new;

        float w = exp(score - m);
        for (uint c = 0; c < CPT; c++)
            acc[c] += w * (float)foveal_v[fov_kv_base + t * HEAD_DIM + lane_id * CPT + c];
    }

    // ==== PERIPHERAL (INT8 K + INT8 V) ====
    uint per_kv_base = bh_kv * N_PER * HEAD_DIM;
    uint per_vs_base = bh_kv * N_PER;
    for (uint t = per_start; t < per_end; t++) {
        float dot = 0.0f;
        for (uint c = 0; c < CPT; c++) {
            float k_val = to_fp16((float)periph_k[per_kv_base + t * HEAD_DIM + lane_id * CPT + c]
                          * pk_s[c] + pk_z[c]);
            dot += q_reg[c] * k_val;
        }
        float score = simd_sum(dot) * INV_SQRT_D;

        float m_new = max(m, score);
        float alpha = exp(m - m_new);
        for (uint c = 0; c < CPT; c++) acc[c] *= alpha;
        l = alpha * l + exp(score - m_new);
        m = m_new;

        float w  = exp(score - m);
        float vs = (float)periph_v_scale[per_vs_base + t];
        float vz = (float)periph_v_zero[per_vs_base + t];
        for (uint c = 0; c < CPT; c++) {
            float v_val = to_fp16((float)periph_v[per_kv_base + t * HEAD_DIM + lane_id * CPT + c]
                          * vs + vz);
            acc[c] += w * v_val;
        }
    }

    // ==== FAR (INT8 K + INT4 V packed, with promotion override buffer) ====
    uint far_k_base  = bh_kv * N_FAR * HEAD_DIM;
    uint far_v_base  = bh_kv * N_FAR * HEAD_DIM_HALF;
    uint far_vs_base = bh_kv * N_FAR;
    uint n_ov = min((uint)override_count[kv_head], MAX_OV);

    // Override buffer is pre-sorted by far index (CPU worker does sorted insert
    // into a double buffer, then atomic swap). Merge-scan: O(N_FAR + n_ov).
    uint ov_ptr = 0;
    while (ov_ptr < n_ov && (uint)override_far_idx[kv_head * MAX_OV + ov_ptr] < far_start) ov_ptr++;

    for (uint t = far_start; t < far_end; t++) {
        bool overridden = (ov_ptr < n_ov && (uint)override_far_idx[kv_head * MAX_OV + ov_ptr] == t);
        uint oi = ov_ptr;
        if (overridden) ov_ptr++;

        float dot = 0.0f;
        if (overridden) {
            uint ok_base = kv_head * MAX_OV * HEAD_DIM + oi * HEAD_DIM;
            for (uint c = 0; c < CPT; c++)
                dot += q_reg[c] * (float)override_k[ok_base + lane_id * CPT + c];
        } else {
            for (uint c = 0; c < CPT; c++) {
                float k_val = to_fp16((float)far_k[far_k_base + t * HEAD_DIM + lane_id * CPT + c]
                              * fk_s[c] + fk_z[c]);
                dot += q_reg[c] * k_val;
            }
        }
        float score = simd_sum(dot) * INV_SQRT_D;

        if (score > max_far_score) {
            max_far_score = score;
            max_far_token = (int)t;
        }

        float m_new = max(m, score);
        float alpha = exp(m - m_new);
        for (uint c = 0; c < CPT; c++) acc[c] *= alpha;
        l = alpha * l + exp(score - m_new);
        m = m_new;

        float w  = exp(score - m);
        if (overridden) {
            uint ov_base = kv_head * MAX_OV * HEAD_DIM + oi * HEAD_DIM;
            for (uint c = 0; c < CPT; c++)
                acc[c] += w * (float)override_v[ov_base + lane_id * CPT + c];
        } else {
            float vs = (float)far_v_scale[far_vs_base + t];
            float vz = (float)far_v_zero[far_vs_base + t];
            for (uint c = 0; c < CPT; c += 2) {
                uint d_even   = lane_id * CPT + c;
                uint packed_d = d_even / 2;
                uint8_t packed_byte = far_v[far_v_base + t * HEAD_DIM_HALF + packed_d];
                float v_even = to_fp16((float)(packed_byte & 0x0F)        * vs + vz);
                float v_odd  = to_fp16((float)((packed_byte >> 4) & 0x0F) * vs + vz);
                acc[c]     += w * v_even;
                acc[c + 1] += w * v_odd;
            }
        }
    }

    // ==== DECODE BUFFER (fp16 K + fp16 V, new tokens since compression) ====
    uint dec_kv_base = bh_kv * N_DECODE * HEAD_DIM;
    for (uint t = dec_start; t < dec_end; t++) {
        float dot = 0.0f;
        for (uint c = 0; c < CPT; c++)
            dot += q_reg[c] * (float)decode_k[dec_kv_base + t * HEAD_DIM + lane_id * CPT + c];
        float score = simd_sum(dot) * INV_SQRT_D;
        min_fov_score = min(min_fov_score, score);

        float m_new = max(m, score);
        float alpha = exp(m - m_new);
        for (uint c = 0; c < CPT; c++) acc[c] *= alpha;
        l = alpha * l + exp(score - m_new);
        m = m_new;

        float w = exp(score - m);
        for (uint c = 0; c < CPT; c++)
            acc[c] += w * (float)decode_v[dec_kv_base + t * HEAD_DIM + lane_id * CPT + c];
    }
)METAL";

static const char* SPLITK_SETUP = R"METAL(
    uint TOTAL_BH_Q = rt_params[0];
    uint N_DECODE    = rt_params[1];

    uint tg_idx = threadgroup_position_in_grid.x;
    uint bh_q   = tg_idx % TOTAL_BH_Q;
    uint split_id = tg_idx / TOTAL_BH_Q;
    uint batch_idx = bh_q / H_Q;
    uint q_head    = bh_q % H_Q;
    uint kv_head   = q_head / GQA_RATIO;
    uint lane_id   = thread_position_in_threadgroup.x;

    uint bh_kv = batch_idx * H_KV + kv_head;

    float q_reg[CPT];
    for (uint c = 0; c < CPT; c++)
        q_reg[c] = (float)query[bh_q * HEAD_DIM + lane_id * CPT + c];

    float pk_s[CPT], pk_z[CPT];
    for (uint c = 0; c < CPT; c++) {
        uint d = lane_id * CPT + c;
        pk_s[c] = (float)periph_k_scale[bh_kv * HEAD_DIM + d];
        pk_z[c] = (float)periph_k_zero[bh_kv * HEAD_DIM + d];
    }
    float fk_s[CPT], fk_z[CPT];
    for (uint c = 0; c < CPT; c++) {
        uint d = lane_id * CPT + c;
        fk_s[c] = (float)far_k_scale[bh_kv * HEAD_DIM + d];
        fk_z[c] = (float)far_k_zero[bh_kv * HEAD_DIM + d];
    }

    float m = -INFINITY, l = 0.0f;
    float acc[CPT];
    for (uint c = 0; c < CPT; c++) acc[c] = 0.0f;
    float min_fov_score = INFINITY, max_far_score = -INFINITY;
    int max_far_token = -1;

    uint S_total = N_FOV + N_PER + N_FAR + N_DECODE;
    uint gstart  = split_id * SPLIT_SIZE;
    uint gend    = min(gstart + SPLIT_SIZE, S_total);

    uint fov_start = min(gstart, N_FOV);
    uint fov_end   = min(gend,   N_FOV);
    uint per_start = (gstart > N_FOV) ? min(gstart - N_FOV, N_PER) : 0u;
    uint per_end   = (gend   > N_FOV) ? min(gend   - N_FOV, N_PER) : 0u;
    uint nfp = N_FOV + N_PER;
    uint far_start = (gstart > nfp) ? min(gstart - nfp, N_FAR) : 0u;
    uint far_end   = (gend   > nfp) ? min(gend   - nfp, N_FAR) : 0u;
    uint nfpf = nfp + N_FAR;
    uint dec_start = (gstart > nfpf) ? min(gstart - nfpf, N_DECODE) : 0u;
    uint dec_end   = (gend   > nfpf) ? min(gend   - nfpf, N_DECODE) : 0u;
)METAL";

static const char* SPLITK_WRITE = R"METAL(
    uint out_idx = split_id * TOTAL_BH_Q + bh_q;
    for (uint c = 0; c < CPT; c++)
        partial_out[out_idx * HEAD_DIM + lane_id * CPT + c] = acc[c];
    if (lane_id == 0) {
        partial_lse[out_idx]       = l;
        partial_max[out_idx]       = m;
        partial_min_fov[out_idx]   = min_fov_score;
        partial_max_far[out_idx]   = max_far_score;
        partial_far_token[out_idx] = max_far_token;
    }
)METAL";

static const char* REDUCE_SOURCE = R"METAL(
    uint NUM_SPLITS = rt_params[0];
    uint TOTAL_BH_Q = rt_params[1];

    uint tg_idx = threadgroup_position_in_grid.x;
    uint bh_q   = tg_idx;
    uint lane_id = thread_position_in_threadgroup.x;

    float m_global = -INFINITY, l_global = 0.0f;
    float acc[CPT];
    for (uint c = 0; c < CPT; c++) acc[c] = 0.0f;
    float min_fov = INFINITY, max_far = -INFINITY;
    int far_token = -1;

    for (uint s = 0; s < NUM_SPLITS; s++) {
        uint idx = s * TOTAL_BH_Q + bh_q;
        float m_s = partial_max[idx];
        if (m_s == -INFINITY) continue;
        float l_s = partial_lse[idx];

        float m_new = max(m_global, m_s);
        float ag = (m_global > -INFINITY) ? exp(m_global - m_new) : 0.0f;
        float as_val = exp(m_s - m_new);
        l_global = ag * l_global + as_val * l_s;
        for (uint c = 0; c < CPT; c++) {
            float pv = partial_out[idx * HEAD_DIM + lane_id * CPT + c];
            acc[c] = ag * acc[c] + as_val * pv;
        }
        m_global = m_new;

        float mf = partial_min_fov[idx];
        min_fov = min(min_fov, mf);
        float fs = partial_max_far[idx];
        if (fs > max_far) {
            max_far = fs;
            far_token = partial_far_token[idx];
        }
    }

    float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    for (uint c = 0; c < CPT; c++)
        out[bh_q * HEAD_DIM + lane_id * CPT + c] = (half)(acc[c] * inv_l);

    if (lane_id == 0) {
        spike_flags[bh_q]  = (max_far > min_fov + SPIKE_MARGIN) ? 1 : 0;
        spike_tokens[bh_q] = far_token;
    }
)METAL";


// ---------------------------------------------------------------------------
// Build kernel header with compile-time constants
// ---------------------------------------------------------------------------

static std::string build_header(const Config& cfg) {
    int cpt = cfg.head_dim / 32;
    std::ostringstream h;
    h << "#include <metal_stdlib>\nusing namespace metal;\n\n"
      << "constant uint N_FOV = " << cfg.n_fov << ";\n"
      << "constant uint N_PER = " << cfg.n_per << ";\n"
      << "constant uint N_FAR = " << cfg.n_far << ";\n"
      << "constant uint HEAD_DIM = " << cfg.head_dim << ";\n"
      << "constant uint HEAD_DIM_HALF = " << cfg.head_dim / 2 << ";\n"
      << "constant uint H_Q = " << cfg.h_q << ";\n"
      << "constant uint H_KV = " << cfg.h_kv << ";\n"
      << "constant uint GQA_RATIO = " << cfg.h_q / cfg.h_kv << ";\n"
      << "constant uint CPT = " << cpt << ";\n"
      << "constant float INV_SQRT_D = " << (1.0 / std::sqrt((double)cfg.head_dim)) << "f;\n"
      << "constant float SPIKE_MARGIN = " << cfg.spike_margin << "f;\n"
      << "constant uint MAX_OV = " << cfg.max_ov << ";\n"
      << "constant uint SPLIT_SIZE = " << cfg.split_size << ";\n\n"
      << "inline float to_fp16(float x) { return (float)((half)x); }\n";
    return h.str();
}


// ---------------------------------------------------------------------------
// Cached kernel callables (one per tier configuration)
// ---------------------------------------------------------------------------

struct KernelCache {
    fast::CustomKernelFunction sk_fn;
    fast::CustomKernelFunction red_fn;
};

static std::unordered_map<std::string, KernelCache> _cache;

static std::string cache_key(const Config& cfg) {
    std::ostringstream k;
    k << cfg.n_fov << "_" << cfg.n_per << "_" << cfg.n_far << "_"
      << cfg.head_dim << "_" << cfg.h_q << "_" << cfg.h_kv << "_"
      << (int)(cfg.spike_margin * 1000) << "_" << cfg.split_size;
    return k.str();
}

static const KernelCache& get_kernels(const Config& cfg) {
    auto key = cache_key(cfg);
    auto it = _cache.find(key);
    if (it != _cache.end()) return it->second;

    std::string header = build_header(cfg);
    std::string sk_source = std::string(SPLITK_SETUP) + TIER_PROCESSING + SPLITK_WRITE;
    std::string red_source = std::string(REDUCE_SOURCE);

    std::string sk_name = "fov_sk_" + key;
    std::string red_name = "fov_red_" + key;

    auto sk_fn = fast::metal_kernel(
        sk_name,
        {"rt_params", "query", "foveal_k", "foveal_v",
         "periph_k", "periph_v", "periph_k_scale", "periph_k_zero",
         "periph_v_scale", "periph_v_zero",
         "far_k", "far_v", "far_k_scale", "far_k_zero",
         "far_v_scale", "far_v_zero",
         "foveal_valid", "decode_k", "decode_v",
         "override_k", "override_v", "override_far_idx", "override_count"},
        {"partial_out", "partial_lse", "partial_max",
         "partial_min_fov", "partial_max_far", "partial_far_token"},
        sk_source, header, /*ensure_row_contiguous=*/true);

    auto red_fn = fast::metal_kernel(
        red_name,
        {"rt_params", "partial_out", "partial_lse", "partial_max",
         "partial_min_fov", "partial_max_far", "partial_far_token"},
        {"out", "spike_flags", "spike_tokens"},
        red_source, header, /*ensure_row_contiguous=*/true);

    _cache[key] = {std::move(sk_fn), std::move(red_fn)};
    return _cache[key];
}


// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<array> foveated_attention(
    const array& foveal_k, const array& foveal_v,
    const array& periph_k, const array& periph_v,
    const array& periph_k_scale, const array& periph_k_zero,
    const array& periph_v_scale, const array& periph_v_zero,
    const array& far_k, const array& far_v,
    const array& far_k_scale, const array& far_k_zero,
    const array& far_v_scale, const array& far_v_zero,
    const array& foveal_valid,
    const array& query, const array& decode_k, const array& decode_v,
    const array& override_k, const array& override_v,
    const array& override_far_idx, const array& override_count,
    float spike_margin, int split_size) {

    int B = foveal_k.shape(0);
    int H_kv = foveal_k.shape(1);
    int H_q = query.shape(1);
    int D = foveal_k.shape(3);
    int N_fov = foveal_k.shape(2);
    int N_per = periph_k.shape(2);
    int N_far = far_k.shape(2);
    int n_decode = decode_k.shape(2);
    int total_bh_q = B * H_q;

    Config cfg;
    cfg.n_fov = N_fov;
    cfg.n_per = N_per;
    cfg.n_far = N_far;
    cfg.head_dim = D;
    cfg.h_q = H_q;
    cfg.h_kv = H_kv;
    cfg.split_size = split_size;
    cfg.max_ov = override_k.shape(1);  // MAX_OV from array shape
    cfg.spike_margin = spike_margin;

    const auto& kernels = get_kernels(cfg);

    // Prepare inputs
    auto q_flat = reshape(query, {total_bh_q, D});
    auto pv_s = reshape(periph_v_scale, {B, H_kv, std::max(N_per, 0)});
    auto pv_z = reshape(periph_v_zero, {B, H_kv, std::max(N_per, 0)});
    auto fv_s = reshape(far_v_scale, {B, H_kv, std::max(N_far, 0)});
    auto fv_z = reshape(far_v_zero, {B, H_kv, std::max(N_far, 0)});
    auto fov_valid = astype(foveal_valid, uint32);

    int S_total = N_fov + N_per + N_far + n_decode;
    int num_splits = (S_total + split_size - 1) / split_size;
    int partial_size = num_splits * total_bh_q;

    auto sk_rt = array({(uint32_t)total_bh_q, (uint32_t)n_decode}, uint32);

    // Split-K kernel
    std::vector<array> sk_inputs = {
        sk_rt, q_flat, foveal_k, foveal_v,
        periph_k, periph_v, periph_k_scale, periph_k_zero, pv_s, pv_z,
        far_k, far_v, far_k_scale, far_k_zero, fv_s, fv_z,
        fov_valid, decode_k, decode_v,
        override_k, override_v, override_far_idx, override_count
    };

    auto partials = kernels.sk_fn(
        sk_inputs,
        {{partial_size, D}, {partial_size}, {partial_size},
         {partial_size}, {partial_size}, {partial_size}},
        {float32, float32, float32, float32, float32, int32},
        {num_splits * total_bh_q * 32, 1, 1},  // grid
        {32, 1, 1},  // threadgroup
        {},  // template args
        0.0f,  // init_value
        false,  // verbose
        StreamOrDevice{});

    // Reduce kernel
    auto red_rt = array({(uint32_t)num_splits, (uint32_t)total_bh_q}, uint32);
    std::vector<array> red_inputs = {red_rt};
    red_inputs.insert(red_inputs.end(), partials.begin(), partials.end());

    auto outputs = kernels.red_fn(
        red_inputs,
        {{total_bh_q, D}, {total_bh_q}, {total_bh_q}},
        {float16, int32, int32},
        {total_bh_q * 32, 1, 1},
        {32, 1, 1},
        {},
        std::nullopt,
        false,
        StreamOrDevice{});

    // Reshape outputs
    return {
        reshape(outputs[0], {B, H_q, 1, D}),
        reshape(outputs[1], {B, H_q}),
        reshape(outputs[2], {B, H_q}),
    };
}

} // namespace foveated
