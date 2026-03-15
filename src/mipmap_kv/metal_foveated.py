"""
Fused Split-K Metal kernel for foveated mixed-precision decode attention on Apple Silicon.

Tokens are partitioned across multiple SIMD groups per head, then reduced.
Massively higher GPU occupancy — same approach as the CUDA Split-K kernel.
For short sequences (S_total <= split_size), num_splits=1 naturally degrades
to a single threadgroup — no separate code path needed.

Features:
  - Register-only K path: INT8/INT4 K loaded from memory, dequanted in registers
  - Online softmax across all three tiers
  - V accumulated in registers with per-token dequant
  - Spike detection piggybacked on softmax
  - Zero fp16 intermediate materialization
"""

import math

import mlx.core as mx

_splitk_cache: dict[tuple, tuple] = {}

# Base split size — tokens per split. Adaptive: grows with context length
# to keep num_splits ≤ 16, avoiding reduce kernel bottleneck at long contexts.
_BASE_SPLIT_SIZE = 256
_MAX_SPLITS = 16


def optimal_split_size(s_total: int) -> int:
    """Compute split size that keeps num_splits capped at _MAX_SPLITS.

    At short contexts (≤4K): split_size=256, num_splits=1-16 (fine).
    At long contexts: split_size grows to keep reduce overhead constant.
    """
    if s_total <= _BASE_SPLIT_SIZE * _MAX_SPLITS:
        return _BASE_SPLIT_SIZE
    # Round up to multiple of 256 for alignment
    return ((s_total + _MAX_SPLITS - 1) // _MAX_SPLITS + 255) // 256 * 256


# Keep for backward compat
DEFAULT_SPLIT_SIZE = _BASE_SPLIT_SIZE

# Max overrides per KV head. Background worker writes promoted fp16 K,V here;
# the Metal kernel reads from this buffer instead of dequanting INT8/INT4.
MAX_OV = 32


# ============================================================================
# Shared Metal code fragments
# ============================================================================

# Score + softmax + accumulate logic used by the split-K kernel.
# Parameterized by tier loop bounds set in each kernel variant.
_TIER_PROCESSING = """
    // ==== FOVEAL (fp16 K + fp16 V, with padding mask) ====
    uint fov_valid = foveal_valid[kv_head];  // per-head valid count
    uint fov_kv_base = bh_kv * N_FOV * HEAD_DIM;
    for (uint t = fov_start; t < fov_end; t++) {
        if (t >= fov_valid) continue;  // skip zero-padded slots
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
    // Dequant rounds to fp16 via to_fp16(...) to match the reference path
    // which dequants to fp16 before feeding to Apple's SDPA. Without this,
    // float32 dequant introduces ~1 ULP differences that compound over 24 layers.
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
        // Merge-scan: O(1) amortized — both sequences are monotonically increasing
        bool overridden = (ov_ptr < n_ov && (uint)override_far_idx[kv_head * MAX_OV + ov_ptr] == t);
        uint oi = ov_ptr;  // buffer is sorted in-place, slot == position
        if (overridden) ov_ptr++;

        float dot = 0.0f;
        if (overridden) {
            // Exact fp16 K from override buffer
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
            // Exact fp16 V from override buffer
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
        min_fov_score = min(min_fov_score, score);  // decode tokens are foveal-quality

        float m_new = max(m, score);
        float alpha = exp(m - m_new);
        for (uint c = 0; c < CPT; c++) acc[c] *= alpha;
        l = alpha * l + exp(score - m_new);
        m = m_new;

        float w = exp(score - m);
        for (uint c = 0; c < CPT; c++)
            acc[c] += w * (float)decode_v[dec_kv_base + t * HEAD_DIM + lane_id * CPT + c];
    }
"""


# ============================================================================
# Split-K main kernel — each threadgroup handles a token range
# ============================================================================

_SPLITK_SOURCE = (
    """
    // N_FOV/N_PER/N_FAR are compile-time. N_DECODE is runtime (grows during decode).
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

    // ---- Compute token range for this split ----
    // Token order: [foveal | peripheral | far | decode]
    uint S_total = N_FOV + N_PER + N_FAR + N_DECODE;
    uint gstart  = split_id * SPLIT_SIZE;
    uint gend    = min(gstart + SPLIT_SIZE, S_total);

    // Map global range → per-tier local ranges
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
"""
    + _TIER_PROCESSING
    + """
    // ---- Write partial results (unnormalized) ----
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
"""
)


# ============================================================================
# Reduce kernel — merge Split-K partial results via online softmax
# ============================================================================

_REDUCE_SOURCE = """
    // Runtime parameters
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

    // Normalize and write final output
    float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    for (uint c = 0; c < CPT; c++)
        out[bh_q * HEAD_DIM + lane_id * CPT + c] = (half)(acc[c] * inv_l);

    if (lane_id == 0) {
        spike_flags[bh_q]  = (max_far > min_fov + SPIKE_MARGIN) ? 1 : 0;
        spike_tokens[bh_q] = far_token;
    }
"""


# ============================================================================
# Kernel builders
# ============================================================================

def _make_header(n_fov, n_per, n_far, head_dim, h_q, h_kv, spike_margin, **extra):
    """Compile-time constants — N_FOV/N_PER/N_FAR fixed after compression."""
    cpt = head_dim // 32
    lines = f"""
    #include <metal_stdlib>
    using namespace metal;

    constant uint N_FOV = {n_fov};
    constant uint N_PER = {n_per};
    constant uint N_FAR = {n_far};
    constant uint HEAD_DIM = {head_dim};
    constant uint HEAD_DIM_HALF = {head_dim // 2};
    constant uint H_Q = {h_q};
    constant uint H_KV = {h_kv};
    constant uint GQA_RATIO = {h_q // h_kv};
    constant uint CPT = {cpt};
    constant float INV_SQRT_D = {1.0 / math.sqrt(head_dim):.10f}f;
    constant float SPIKE_MARGIN = {spike_margin:.6f}f;
    constant uint MAX_OV = {MAX_OV};

    // Round float32 to fp16 precision (matches reference dequant path)
    inline float to_fp16(float x) {{ return (float)((half)x); }}
    """
    for k, v in extra.items():
        lines += f"    constant uint {k} = {v};\n"
    return lines


def _build_splitk_kernels(n_fov, n_per, n_far, head_dim, h_q, h_kv,
                           spike_margin, split_size):
    """Build Split-K + Reduce kernels. Compiled ONCE per tier config."""
    assert head_dim % 32 == 0
    assert h_q % h_kv == 0

    # Split-K main kernel — N_FOV is compile-time (foveal fixed after compression)
    sk_header = _make_header(
        n_fov, n_per, n_far, head_dim, h_q, h_kv, spike_margin,
        SPLIT_SIZE=split_size,
    )
    sk_kernel = mx.fast.metal_kernel(
        name=f"fov_splitk_{n_fov}_{n_per}_{n_far}_{head_dim}_{h_q}_{h_kv}",
        input_names=[
            "rt_params",  # [TOTAL_BH_Q, N_DECODE] uint32
            "query", "foveal_k", "foveal_v",
            "periph_k", "periph_v", "periph_k_scale", "periph_k_zero",
            "periph_v_scale", "periph_v_zero",
            "far_k", "far_v", "far_k_scale", "far_k_zero",
            "far_v_scale", "far_v_zero",
            "foveal_valid",          # (H_kv,) uint32 — per-head valid count
            "decode_k", "decode_v",  # (B, H_kv, N_decode, D) fp16
            "override_k", "override_v",        # (H_kv, MAX_OV, D) fp16
            "override_far_idx", "override_count",  # (H_kv, MAX_OV) int32, (H_kv,) int32
        ],
        output_names=[
            "partial_out", "partial_lse", "partial_max",
            "partial_min_fov", "partial_max_far", "partial_far_token",
        ],
        header=sk_header, source=_SPLITK_SOURCE, ensure_row_contiguous=True,
    )

    # Reduce kernel — merges Split-K partial results
    red_header = _make_header(
        n_fov, n_per, n_far, head_dim, h_q, h_kv, spike_margin,
    )
    red_kernel = mx.fast.metal_kernel(
        name=f"fov_reduce_{n_fov}_{head_dim}_{h_q}",
        input_names=[
            "rt_params",  # [NUM_SPLITS, TOTAL_BH_Q] uint32
            "partial_out", "partial_lse", "partial_max",
            "partial_min_fov", "partial_max_far", "partial_far_token",
        ],
        output_names=["out", "spike_flags", "spike_tokens"],
        header=red_header, source=_REDUCE_SOURCE, ensure_row_contiguous=True,
    )

    return sk_kernel, red_kernel


# ============================================================================
# Public API
# ============================================================================

def _get_splitk_kernels(n_fov, n_per, n_far, head_dim, h_q, h_kv,
                         spike_margin, split_size):
    """Get or build cached kernels. Compiled once per tier configuration."""
    key = (n_fov, n_per, n_far, head_dim, h_q, h_kv, spike_margin, split_size)
    if key not in _splitk_cache:
        _splitk_cache[key] = _build_splitk_kernels(
            n_fov, n_per, n_far, head_dim, h_q, h_kv, spike_margin, split_size,
        )
    return _splitk_cache[key]


def _prepare_inputs(query, foveal_k, foveal_v,
                    periph_k, periph_v, periph_k_scale, periph_k_zero,
                    periph_v_scale, periph_v_zero,
                    far_k, far_v, far_k_scale, far_k_zero,
                    far_v_scale, far_v_zero,
                    foveal_valid=None):
    """Flatten/squeeze inputs for kernel consumption."""
    B, H_q, _, D = query.shape
    H_kv = foveal_k.shape[1]
    N_fov = foveal_k.shape[2]
    N_per = periph_k.shape[2]
    N_far = far_k.shape[2]

    q_flat = query.reshape(B * H_q, D)
    pv_s = periph_v_scale.reshape(B, H_kv, max(N_per, 0))
    pv_z = periph_v_zero.reshape(B, H_kv, max(N_per, 0))
    fv_s = far_v_scale.reshape(B, H_kv, max(N_far, 0))
    fv_z = far_v_zero.reshape(B, H_kv, max(N_far, 0))

    # Default foveal_valid: all slots valid (no padding)
    if foveal_valid is None:
        foveal_valid = mx.full((H_kv,), N_fov, dtype=mx.uint32)
    else:
        foveal_valid = foveal_valid.astype(mx.uint32)

    return (
        [q_flat, foveal_k, foveal_v,
         periph_k, periph_v, periph_k_scale, periph_k_zero, pv_s, pv_z,
         far_k, far_v, far_k_scale, far_k_zero, fv_s, fv_z,
         foveal_valid],
        B, H_q, H_kv, D, N_fov, N_per, N_far,
    )


def foveated_attention_metal(
    query, foveal_k, foveal_v,
    periph_k, periph_v, periph_k_scale, periph_k_zero,
    periph_v_scale, periph_v_zero,
    far_k, far_v, far_k_scale, far_k_zero,
    far_v_scale, far_v_zero,
    spike_margin: float = 0.5,
    split_size: int = None,
    decode_k: mx.array = None,
    decode_v: mx.array = None,
    foveal_valid: mx.array = None,
    override_k: mx.array = None,
    override_v: mx.array = None,
    override_far_idx: mx.array = None,
    override_count: mx.array = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused foveated attention via custom Split-K Metal kernel.

    Always uses Split-K. For short sequences (S_total <= split_size),
    num_splits=1 naturally degrades to a single threadgroup.

    Returns:
        output: (B, H_q, 1, D) float16
        spike_flags: (B, H_q) int32
        spike_tokens: (B, H_q) int32
    """
    inputs, B, H_q, H_kv, D, N_fov, N_per, N_far = _prepare_inputs(
        query, foveal_k, foveal_v,
        periph_k, periph_v, periph_k_scale, periph_k_zero,
        periph_v_scale, periph_v_zero,
        far_k, far_v, far_k_scale, far_k_zero,
        far_v_scale, far_v_zero,
        foveal_valid=foveal_valid,
    )

    total_bh_q = B * H_q
    n_decode = decode_k.shape[2] if decode_k is not None else 0
    if split_size is None:
        split_size = optimal_split_size(N_fov + N_per + N_far + n_decode)

    return _run_splitk(inputs, B, H_q, H_kv, D, N_fov, N_per, N_far,
                       spike_margin, total_bh_q, split_size,
                       decode_k=decode_k, decode_v=decode_v,
                       override_k=override_k, override_v=override_v,
                       override_far_idx=override_far_idx,
                       override_count=override_count)


def _run_splitk(inputs, B, H_q, H_kv, D, N_fov, N_per, N_far,
                spike_margin, total_bh_q, split_size,
                decode_k=None, decode_v=None,
                override_k=None, override_v=None,
                override_far_idx=None, override_count=None):
    sk_kernel, red_kernel = _get_splitk_kernels(
        N_fov, N_per, N_far, D, H_q, H_kv, spike_margin, split_size,
    )

    # Decode buffer (4th tier — new tokens since compression)
    n_decode = decode_k.shape[2] if decode_k is not None else 0
    if decode_k is None:
        decode_k = mx.zeros((B, H_kv, 0, D), dtype=mx.float16)
        decode_v = mx.zeros((B, H_kv, 0, D), dtype=mx.float16)

    # Override buffer — promoted fp16 K,V for far-tier tokens
    if override_k is None:
        override_k = mx.zeros((H_kv, MAX_OV, D), dtype=mx.float16)
        override_v = mx.zeros((H_kv, MAX_OV, D), dtype=mx.float16)
        override_far_idx = mx.zeros((H_kv, MAX_OV), dtype=mx.int32)
        override_count = mx.zeros((H_kv,), dtype=mx.int32)

    # S_total includes decode tokens for Split-K partitioning
    S_total_with_dec = N_fov + N_per + N_far + n_decode
    num_splits = (S_total_with_dec + split_size - 1) // split_size
    partial_size = num_splits * total_bh_q

    # Runtime params
    sk_rt_params = mx.array([total_bh_q, n_decode], dtype=mx.uint32)
    red_rt_params = mx.array([num_splits, total_bh_q], dtype=mx.uint32)

    # Kernel 1: Split-K main — each threadgroup handles a token range
    # (includes decode buffer as 4th tier in the split partitioning)
    partials = sk_kernel(
        inputs=[sk_rt_params] + inputs + [decode_k, decode_v,
                override_k, override_v, override_far_idx, override_count],
        output_shapes=[
            (partial_size, D),   # partial_out (float32)
            (partial_size,),     # partial_lse
            (partial_size,),     # partial_max
            (partial_size,),     # partial_min_fov
            (partial_size,),     # partial_max_far
            (partial_size,),     # partial_far_token
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.float32,
            mx.float32, mx.float32, mx.int32,
        ],
        grid=(num_splits * total_bh_q * 32, 1, 1),
        threadgroup=(32, 1, 1),
        init_value=0.0,
    )

    # Kernel 2: Reduce — merge Split-K partial results
    outputs = red_kernel(
        inputs=[red_rt_params] + list(partials),
        output_shapes=[(total_bh_q, D), (total_bh_q,), (total_bh_q,)],
        output_dtypes=[mx.float16, mx.int32, mx.int32],
        grid=(total_bh_q * 32, 1, 1),
        threadgroup=(32, 1, 1),
    )

    return (
        outputs[0].reshape(B, H_q, 1, D),
        outputs[1].reshape(B, H_q),
        outputs[2].reshape(B, H_q),
    )


def is_available() -> bool:
    """Check if the Metal foveated kernel can be used."""
    try:
        q = mx.zeros((1, 1, 1, 64), dtype=mx.float16)
        fk = mx.zeros((1, 1, 1, 64), dtype=mx.float16)
        fv = mx.zeros((1, 1, 1, 64), dtype=mx.float16)
        pk = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        pv = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        ps = mx.zeros((1, 1, 64), dtype=mx.float16)
        pvs = mx.zeros((1, 1, 1), dtype=mx.float16)
        fkk = mx.zeros((1, 1, 1, 64), dtype=mx.uint8)
        fvv = mx.zeros((1, 1, 1, 32), dtype=mx.uint8)
        fks = mx.zeros((1, 1, 64), dtype=mx.float16)
        fvs = mx.zeros((1, 1, 1), dtype=mx.float16)
        out, flags, tokens = foveated_attention_metal(
            q, fk, fv, pk, pv, ps, ps, pvs, pvs, fkk, fvv, fks, fks, fvs, fvs,
        )
        mx.eval(out)
        return True
    except Exception:
        return False
