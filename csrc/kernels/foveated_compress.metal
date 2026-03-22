// Foveated compression kernels — precompiled metallib (2-tier)
//
// Three kernels:
//   compress_foveal  — gather non-contiguous fp16 segments into padded near buffer
//   compress_fp8     — fp16 → fp8 E4M3 with round-to-nearest-even (far K)
//   compress_int4    — fp16 → int4 per-token quantize + nibble pack (far V)
//
// Templated on D (head dimension).

#include <metal_stdlib>
using namespace metal;

// Must match C++ structs exactly.
struct FovealParams {
    uint S;
    uint n_sinks;
    uint fov_mid_start;
    uint fov_mid_count;
    uint window_start;
    uint window_count;
    uint n_near_padded;
    uint near_actual;
};

struct CompressParams {
    uint S;              // source sequence length (stride)
    uint src_offset;     // start token index in source
    uint N;              // number of tokens to compress
};


// ============================================================================
// fp8 E4M3 encode — register-level quantization
//
// E4M3FN: sign(1) + exp(4, bias=7) + mantissa(3), no inf, max=448
// Round-to-nearest-even when reducing mantissa from 10→3 bits.
// ============================================================================

inline uint8_t fp16_to_e4m3(half val) {
    ushort bits = as_type<ushort>(val);
    uint sign = bits >> 15;
    int exp16 = (int)((bits >> 10) & 0x1F);
    uint mant16 = bits & 0x3FF;

    // fp16 zero/subnormal → fp8 zero
    if (exp16 == 0) return (uint8_t)(sign << 7);
    // fp16 inf/nan → fp8 max (E4M3FN has no inf/nan)
    if (exp16 == 31) return (uint8_t)((sign << 7) | 0x7E);

    // Rebias exponent: fp16 bias=15, E4M3 bias=7
    int exp8 = exp16 - 8;
    if (exp8 >= 15) return (uint8_t)((sign << 7) | 0x7E);  // overflow → max
    if (exp8 <= 0)  return (uint8_t)(sign << 7);            // underflow → zero

    // Round-to-nearest-even: 10→3 mantissa bits
    uint trunc = mant16 >> 7;
    uint round_bit = (mant16 >> 6) & 1;
    uint sticky = (mant16 & 0x3F) != 0;
    if (round_bit && (sticky || (trunc & 1))) {
        trunc++;
        if (trunc >= 8) { trunc = 0; exp8++; }
        if (exp8 >= 15) return (uint8_t)((sign << 7) | 0x7E);
    }

    return (uint8_t)((sign << 7) | (exp8 << 3) | trunc);
}


// ============================================================================
// compress_foveal — copy 3 non-contiguous segments into zero-padded output
//
// Grid:  (n_near_padded, B * H, 1)
// Group: (32, 1, 1)
// ============================================================================

template<int D>
[[kernel, max_total_threads_per_threadgroup(32)]]
void compress_foveal(
    const device half* src       [[buffer(0)]],
    device half* dst             [[buffer(1)]],
    constant FovealParams& p     [[buffer(2)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]])
{
    uint t_out = gid.x;
    uint bh = gid.y;

    if (t_out >= p.n_near_padded) return;

    int src_t = -1;
    if (t_out < p.n_sinks) {
        src_t = (int)t_out;
    } else if (t_out < p.n_sinks + p.fov_mid_count) {
        src_t = (int)(p.fov_mid_start + (t_out - p.n_sinks));
    } else if (t_out < p.near_actual) {
        src_t = (int)(p.window_start + (t_out - p.n_sinks - p.fov_mid_count));
    }

    constexpr int ELEMS = D / 32;
    uint dst_base = (bh * p.n_near_padded + t_out) * D;
    uint src_base = bh * p.S * D;

    for (int i = 0; i < ELEMS; i++) {
        uint d = lid * ELEMS + i;
        dst[dst_base + d] = (src_t >= 0)
            ? src[src_base + (uint)src_t * D + d]
            : (half)0;
    }
}


// ============================================================================
// compress_fp8 — fp16 → fp8 E4M3 for far K
//
// Register-level quantization with round-to-nearest-even.
// Grid:  (N, B * H, 1)
// Group: (32, 1, 1)
// ============================================================================

template<int D>
[[kernel, max_total_threads_per_threadgroup(32)]]
void compress_fp8(
    const device half* src       [[buffer(0)]],
    device uint8_t* dst          [[buffer(1)]],
    constant CompressParams& p   [[buffer(2)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]])
{
    uint t = gid.x;
    uint bh = gid.y;

    if (t >= p.N) return;

    constexpr int ELEMS = D / 32;
    uint src_row = bh * p.S * D + (p.src_offset + t) * D;
    uint dst_row = bh * p.N * D + t * D;

    for (int i = 0; i < ELEMS; i++) {
        uint d = lid * ELEMS + i;
        dst[dst_row + d] = fp16_to_e4m3(src[src_row + d]);
    }
}


// ============================================================================
// compress_int4 — INT4 per-token quantize + nibble pack for far V
//
// Grid:  (N, B * H, 1)
// Group: (32, 1, 1)
// Output: (B, H, N, D/2) uint8 — packed[p] = val[2p] | (val[2p+1] << 4)
// ============================================================================

template<int D>
[[kernel, max_total_threads_per_threadgroup(32)]]
void compress_int4(
    const device half* src         [[buffer(0)]],
    device uint8_t* packed_out     [[buffer(1)]],
    device half* scale_out         [[buffer(2)]],
    device half* zero_out          [[buffer(3)]],
    constant CompressParams& p     [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]])
{
    uint t = gid.x;
    uint bh = gid.y;

    if (t >= p.N) return;

    uint src_row = bh * p.S * D + (p.src_offset + t) * D;

    // Per-token min/max via SIMD reduction
    constexpr int ELEMS = D / 32;
    float local_min = INFINITY, local_max = -INFINITY;
    for (int i = 0; i < ELEMS; i++) {
        float val = (float)src[src_row + lid * ELEMS + i];
        local_min = min(local_min, val);
        local_max = max(local_max, val);
    }

    float token_min = simd_min(local_min);
    float token_max = simd_max(local_max);
    float scale = max((token_max - token_min) / 15.0f, 1e-8f);

    if (lid == 0) {
        scale_out[bh * p.N + t] = (half)scale;
        zero_out[bh * p.N + t] = (half)token_min;
    }

    // Quantize + nibble pack
    constexpr int PACKED = ELEMS / 2;
    uint packed_base = (bh * p.N + t) * (D / 2);

    for (int i = 0; i < PACKED; i++) {
        uint d_lo = lid * ELEMS + i * 2;
        uint d_hi = d_lo + 1;

        float v_lo = (float)src[src_row + d_lo];
        float v_hi = (float)src[src_row + d_hi];

        uint8_t q_lo = (uint8_t)clamp(round((v_lo - token_min) / scale), 0.0f, 15.0f);
        uint8_t q_hi = (uint8_t)clamp(round((v_hi - token_min) / scale), 0.0f, 15.0f);

        packed_out[packed_base + lid * PACKED + i] = q_lo | (q_hi << 4);
    }
}


// ============================================================================
// Template instantiations — D=64 (0.5B) and D=128 (7B+)
// ============================================================================

#define INST_FOVEAL(D) \
    template [[host_name("compress_foveal_d" #D)]] \
    [[kernel]] void compress_foveal<D>( \
        const device half*, device half*, \
        constant FovealParams&, uint2, uint);

#define INST_FP8(D) \
    template [[host_name("compress_fp8_d" #D)]] \
    [[kernel]] void compress_fp8<D>( \
        const device half*, device uint8_t*, \
        constant CompressParams&, uint2, uint);

#define INST_INT4(D) \
    template [[host_name("compress_int4_d" #D)]] \
    [[kernel]] void compress_int4<D>( \
        const device half*, device uint8_t*, device half*, device half*, \
        constant CompressParams&, uint2, uint);

INST_FOVEAL(64)
INST_FOVEAL(128)
INST_FP8(64)
INST_FP8(128)
INST_INT4(64)
INST_INT4(128)
