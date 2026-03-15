// Direct CommandEncoder dispatch — bypasses fast::metal_kernel entirely.
// Key optimizations over first attempt:
//   - Pre-allocated partial buffers (no malloc in eval_gpu)
//   - No memset (kernel writes all slots)
//   - Cached pipeline state pointer (no map lookup per call)
//   - Single command encoder for both Split-K + Reduce

#include "foveated_primitive.h"
#include "foveated_attn.h"  // for kernel source strings

#include <cmath>
#include <sstream>

namespace foveated {

using namespace mlx::core;

static int adaptive_split_size(int s_total) {
    const int base = 256, max_splits = 16;
    if (s_total <= base * max_splits) return base;
    return ((s_total + max_splits - 1) / max_splits + 255) / 256 * 256;
}

// Global compiled pipeline cache
struct PipelineCache {
    MTL::ComputePipelineState* splitk_pso = nullptr;
    MTL::ComputePipelineState* reduce_pso = nullptr;
    MTL::ComputePipelineState* singlepass_pso = nullptr;
};
static std::unordered_map<std::string, PipelineCache> _pso_cache;

// Build Metal source with constants injected
static std::string build_full_source(
    const char* body,
    int n_fov, int n_per, int n_far,
    int head_dim, int h_q, int h_kv,
    int split_size, int max_ov, float spike_margin)
{
    int cpt = head_dim / 32;
    std::ostringstream s;
    s << "#include <metal_stdlib>\nusing namespace metal;\n\n"
      << "constant uint N_FOV = " << n_fov << ";\n"
      << "constant uint N_PER = " << n_per << ";\n"
      << "constant uint N_FAR = " << n_far << ";\n"
      << "constant uint HEAD_DIM = " << head_dim << ";\n"
      << "constant uint HEAD_DIM_HALF = " << head_dim / 2 << ";\n"
      << "constant uint H_Q = " << h_q << ";\n"
      << "constant uint H_KV = " << h_kv << ";\n"
      << "constant uint GQA_RATIO = " << h_q / h_kv << ";\n"
      << "constant uint CPT = " << cpt << ";\n"
      << "constant float INV_SQRT_D = " << 1.0 / std::sqrt((double)head_dim) << "f;\n"
      << "constant float SPIKE_MARGIN = " << spike_margin << "f;\n"
      << "constant uint MAX_OV = " << max_ov << ";\n"
      << "constant uint SPLIT_SIZE = " << split_size << ";\n\n"
      << "inline float to_fp16(float x) { return (float)((half)x); }\n\n"
      << body;
    return s.str();
}

// Metal kernel signatures with explicit buffer bindings
static const char* SPLITK_KERNEL_SIG = R"METAL(
[[kernel]] void fov_splitk(
    const device uint32_t* rt_params [[buffer(0)]],
    const device half* query [[buffer(1)]],
    const device half* foveal_k [[buffer(2)]],
    const device half* foveal_v [[buffer(3)]],
    const device uint8_t* periph_k [[buffer(4)]],
    const device uint8_t* periph_v [[buffer(5)]],
    const device half* periph_k_scale [[buffer(6)]],
    const device half* periph_k_zero [[buffer(7)]],
    const device half* periph_v_scale [[buffer(8)]],
    const device half* periph_v_zero [[buffer(9)]],
    const device uint8_t* far_k [[buffer(10)]],
    const device uint8_t* far_v [[buffer(11)]],
    const device half* far_k_scale [[buffer(12)]],
    const device half* far_k_zero [[buffer(13)]],
    const device half* far_v_scale [[buffer(14)]],
    const device half* far_v_zero [[buffer(15)]],
    const device uint32_t* foveal_valid [[buffer(16)]],
    const device half* decode_k [[buffer(17)]],
    const device half* decode_v [[buffer(18)]],
    const device half* override_k [[buffer(19)]],
    const device half* override_v [[buffer(20)]],
    const device int32_t* override_far_idx [[buffer(21)]],
    const device int32_t* override_count [[buffer(22)]],
    device float* partial_out [[buffer(23)]],
    device float* partial_lse [[buffer(24)]],
    device float* partial_max [[buffer(25)]],
    device float* partial_min_fov [[buffer(26)]],
    device float* partial_max_far [[buffer(27)]],
    device int32_t* partial_far_token [[buffer(28)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup_3 [[thread_position_in_threadgroup]]
) {
    uint3 thread_position_in_threadgroup = thread_position_in_threadgroup_3;
)METAL";

static const char* REDUCE_KERNEL_SIG = R"METAL(
[[kernel]] void fov_reduce(
    const device uint32_t* rt_params [[buffer(0)]],
    const device float* partial_out [[buffer(1)]],
    const device float* partial_lse [[buffer(2)]],
    const device float* partial_max [[buffer(3)]],
    const device float* partial_min_fov [[buffer(4)]],
    const device float* partial_max_far [[buffer(5)]],
    const device int32_t* partial_far_token [[buffer(6)]],
    device half* out [[buffer(7)]],
    device int32_t* spike_flags [[buffer(8)]],
    device int32_t* spike_tokens [[buffer(9)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup_3 [[thread_position_in_threadgroup]]
) {
    uint3 thread_position_in_threadgroup = thread_position_in_threadgroup_3;
)METAL";

// Kernel body source (shared with Python path)
extern const char* SPLITK_SETUP;
extern const char* TIER_PROCESSING;
extern const char* SPLITK_WRITE;
extern const char* REDUCE_SOURCE;

// Single-pass kernel: one threadgroup per bh_q, processes ALL tokens.
// No partials, no reduce, no barrier. 23 inputs + 3 outputs.
static const char* SINGLEPASS_KERNEL_SIG = R"METAL(
[[kernel]] void fov_singlepass(
    const device uint32_t* rt_params [[buffer(0)]],
    const device half* query [[buffer(1)]],
    const device half* foveal_k [[buffer(2)]],
    const device half* foveal_v [[buffer(3)]],
    const device uint8_t* periph_k [[buffer(4)]],
    const device uint8_t* periph_v [[buffer(5)]],
    const device half* periph_k_scale [[buffer(6)]],
    const device half* periph_k_zero [[buffer(7)]],
    const device half* periph_v_scale [[buffer(8)]],
    const device half* periph_v_zero [[buffer(9)]],
    const device uint8_t* far_k [[buffer(10)]],
    const device uint8_t* far_v [[buffer(11)]],
    const device half* far_k_scale [[buffer(12)]],
    const device half* far_k_zero [[buffer(13)]],
    const device half* far_v_scale [[buffer(14)]],
    const device half* far_v_zero [[buffer(15)]],
    const device uint32_t* foveal_valid [[buffer(16)]],
    const device half* decode_k [[buffer(17)]],
    const device half* decode_v [[buffer(18)]],
    const device half* override_k [[buffer(19)]],
    const device half* override_v [[buffer(20)]],
    const device int32_t* override_far_idx [[buffer(21)]],
    const device int32_t* override_count [[buffer(22)]],
    device half* out [[buffer(23)]],
    device int32_t* spike_flags [[buffer(24)]],
    device int32_t* spike_tokens [[buffer(25)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup_3 [[thread_position_in_threadgroup]]
) {
    uint3 thread_position_in_threadgroup = thread_position_in_threadgroup_3;
)METAL";

// Single-pass setup: no split partitioning, process all tokens
static const char* SINGLEPASS_SETUP = R"METAL(
    uint TOTAL_BH_Q = rt_params[0];
    uint N_DECODE    = rt_params[1];

    uint bh_q      = threadgroup_position_in_grid.x;
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

    // Process ALL tokens — no split partitioning
    uint fov_start = 0, fov_end = N_FOV;
    uint per_start = 0, per_end = N_PER;
    uint far_start = 0, far_end = N_FAR;
    uint dec_start = 0, dec_end = N_DECODE;
)METAL";

// Single-pass output: normalize and write directly
static const char* SINGLEPASS_OUTPUT = R"METAL(
    // Normalize
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint c = 0; c < CPT; c++)
        out[bh_q * HEAD_DIM + lane_id * CPT + c] = (half)(acc[c] * inv_l);

    // Spike detection
    if (lane_id == 0) {
        spike_flags[bh_q]  = (max_far_score > min_fov_score + SPIKE_MARGIN) ? 1 : 0;
        spike_tokens[bh_q] = max_far_token;
    }
)METAL";


// ---------------------------------------------------------------------------
// FoveatedPrimitive
// ---------------------------------------------------------------------------

FoveatedPrimitive::FoveatedPrimitive(
    Stream stream,
    int n_fov, int n_per, int n_far,
    int head_dim, int h_q, int h_kv,
    int split_size, int max_ov, float spike_margin,
    int n_static,
    std::vector<array> static_arrays)
    : Primitive(stream),
      n_fov_(n_fov), n_per_(n_per), n_far_(n_far),
      head_dim_(head_dim), h_q_(h_q), h_kv_(h_kv),
      split_size_(split_size), max_ov_(max_ov),
      spike_margin_(spike_margin), n_static_(n_static),
      static_arrays_(std::move(static_arrays)) {}

std::string FoveatedPrimitive::kernel_key_() const {
    std::ostringstream k;
    k << n_fov_ << "_" << n_per_ << "_" << n_far_ << "_"
      << head_dim_ << "_" << h_q_ << "_" << h_kv_ << "_"
      << split_size_ << "_" << max_ov_;
    return k.str();
}

bool FoveatedPrimitive::is_equivalent(const Primitive& other) const {
    auto* o = dynamic_cast<const FoveatedPrimitive*>(&other);
    if (!o) return false;
    return n_fov_ == o->n_fov_ && n_per_ == o->n_per_ && n_far_ == o->n_far_ &&
           head_dim_ == o->head_dim_ && h_q_ == o->h_q_ && h_kv_ == o->h_kv_ &&
           split_size_ == o->split_size_;
}

void FoveatedPrimitive::ensure_pipelines_() const {
    auto key = kernel_key_();
    if (_pso_cache.count(key)) return;

    auto& d = metal::device(default_device());

    std::string sk_body = std::string(SPLITK_SETUP) + TIER_PROCESSING + SPLITK_WRITE;
    std::string sk_source = build_full_source(
        (std::string(SPLITK_KERNEL_SIG) + sk_body + "\n}\n").c_str(),
        n_fov_, n_per_, n_far_, head_dim_, h_q_, h_kv_,
        split_size_, max_ov_, spike_margin_);

    std::string red_source = build_full_source(
        (std::string(REDUCE_KERNEL_SIG) + REDUCE_SOURCE + "\n}\n").c_str(),
        n_fov_, n_per_, n_far_, head_dim_, h_q_, h_kv_,
        split_size_, max_ov_, spike_margin_);

    std::string sk_lib_name = "fov_prim_sk_" + key;
    std::string red_lib_name = "fov_prim_red_" + key;

    auto* sk_lib = d.get_library(sk_lib_name, [&]() { return sk_source; });
    auto* red_lib = d.get_library(red_lib_name, [&]() { return red_source; });

    // Single-pass kernel (no partials, one dispatch)
    std::string sp_body = std::string(SINGLEPASS_SETUP) + TIER_PROCESSING + SINGLEPASS_OUTPUT;
    std::string sp_source = build_full_source(
        (std::string(SINGLEPASS_KERNEL_SIG) + sp_body + "\n}\n").c_str(),
        n_fov_, n_per_, n_far_, head_dim_, h_q_, h_kv_,
        split_size_, max_ov_, spike_margin_);

    std::string sp_lib_name = "fov_prim_sp_" + key;
    auto* sp_lib = d.get_library(sp_lib_name, [&]() { return sp_source; });

    PipelineCache pc;
    pc.splitk_pso = d.get_kernel("fov_splitk", sk_lib);
    pc.reduce_pso = d.get_kernel("fov_reduce", red_lib);
    pc.singlepass_pso = d.get_kernel("fov_singlepass", sp_lib);
    _pso_cache[key] = pc;
}

void FoveatedPrimitive::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    ensure_pipelines_();
    auto& pc = _pso_cache[kernel_key_()];

    auto& d = metal::device(stream().device);
    auto& enc = d.get_command_encoder(stream().index);

    // Graph inputs (7 dynamic only — no statics, no partials):
    //   [0]: query_flat (total_bh_q, D)
    //   [1]: decode_k, [2]: decode_v
    //   [3..6]: override arrays

    int n_decode = inputs[1].shape(2);
    int total_bh_q = inputs[0].shape(0);
    int D = head_dim_;

    // Allocate final outputs
    for (auto& o : outputs)
        o.set_data(allocator::malloc(o.nbytes()));

    // Single-pass kernel: one threadgroup per bh_q, ALL tokens in one pass.
    // No partials, no reduce, no barrier. 23 inputs + 3 outputs.
    enc.set_compute_pipeline_state(pc.singlepass_pso);

    uint32_t rt_params[2] = {(uint32_t)total_bh_q, (uint32_t)n_decode};
    enc.set_bytes(rt_params, 2, 0);

    // Buffer 1: query
    enc.set_input_array(inputs[0], 1);
    // Buffers 2-16: static arrays (stored on primitive, bypass graph)
    for (int i = 0; i < 15; i++)
        enc.set_input_array(static_arrays_[i], i + 2);
    // Buffers 17-18: decode
    enc.set_input_array(inputs[1], 17);
    enc.set_input_array(inputs[2], 18);
    // Buffers 19-22: overrides
    for (int i = 0; i < 4; i++)
        enc.set_input_array(inputs[3 + i], 19 + i);
    // Buffers 23-25: final outputs
    enc.set_output_array(outputs[0], 23);
    enc.set_output_array(outputs[1], 24);
    enc.set_output_array(outputs[2], 25);

    enc.dispatch_threadgroups(
        MTL::Size(total_bh_q, 1, 1),
        MTL::Size(32, 1, 1));
}


// ---------------------------------------------------------------------------
// FoveatedHandleDirect
// ---------------------------------------------------------------------------

static const int MAX_SPLITS = 16;

FoveatedHandleDirect::FoveatedHandleDirect(
    const array& foveal_k, const array& foveal_v,
    const array& periph_k, const array& periph_v,
    const array& periph_k_scale, const array& periph_k_zero,
    const array& periph_v_scale, const array& periph_v_zero,
    const array& far_k, const array& far_v,
    const array& far_k_scale, const array& far_k_zero,
    const array& far_v_scale, const array& far_v_zero,
    const array& foveal_valid,
    float spike_margin, int max_ov)
    : spike_margin_(spike_margin), max_ov_(max_ov)
{
    B_ = foveal_k.shape(0);
    H_kv_ = foveal_k.shape(1);
    D_ = foveal_k.shape(3);
    N_fov_ = foveal_k.shape(2);
    N_per_ = periph_k.shape(2);
    N_far_ = far_k.shape(2);
    N_static_ = N_fov_ + N_per_ + N_far_;

    // Pre-reshape static arrays (indices 0-14)
    static_arrays_ = {
        foveal_k, foveal_v,
        periph_k, periph_v, periph_k_scale, periph_k_zero,
        reshape(periph_v_scale, {B_, H_kv_, std::max(N_per_, 0)}),
        reshape(periph_v_zero, {B_, H_kv_, std::max(N_per_, 0)}),
        far_k, far_v, far_k_scale, far_k_zero,
        reshape(far_v_scale, {B_, H_kv_, std::max(N_far_, 0)}),
        reshape(far_v_zero, {B_, H_kv_, std::max(N_far_, 0)}),
        astype(foveal_valid, uint32),
    };

    // Materialize static arrays so Metal buffers exist
    eval(static_arrays_);

    // Extract Metal buffer pointers (stable as long as static_arrays_ refs are held)
    static_bufs_.reserve(15);
    for (auto& a : static_arrays_) {
        static_bufs_.push_back({a.buffer().raw_ptr(), a.offset()});
    }

    max_total_bh_q_ = 0;
}

std::vector<array> FoveatedHandleDirect::operator()(
    const array& query,
    const array& decode_k, const array& decode_v,
    const array& override_k, const array& override_v,
    const array& override_far_idx, const array& override_count)
{
    int H_q = query.shape(1);
    int n_decode = decode_k.shape(2);
    int S_total = N_static_ + n_decode;
    int split_size = adaptive_split_size(S_total);
    int total_bh_q = B_ * H_q;
    int num_splits = (S_total + split_size - 1) / split_size;
    int partial_size = MAX_SPLITS * total_bh_q;  // max size, not actual

    // Pre-allocate partials once (or when H_q changes)
    if (total_bh_q != max_total_bh_q_) {
        max_total_bh_q_ = total_bh_q;
        // Allocate + eval once so they're materialized Metal buffers
        partials_ = {
            zeros({partial_size, D_}, float32),
            zeros({partial_size}, float32),
            zeros({partial_size}, float32),
            zeros({partial_size}, float32),
            zeros({partial_size}, float32),
            zeros({partial_size}, int32),
        };
        eval(partials_);
    }

    auto prim = std::make_shared<FoveatedPrimitive>(
        default_stream(default_device()),
        N_fov_, N_per_, N_far_,
        D_, H_q, H_kv_,
        split_size, max_ov_, spike_margin_, 0,
        static_arrays_);  // primitive holds refs to static arrays

    // Graph inputs: ONLY 7 dynamic arrays (statics on primitive, no partials)
    std::vector<array> inputs;
    inputs.push_back(reshape(query, {total_bh_q, D_}));  // 0: query
    inputs.push_back(decode_k);                           // 1: decode_k
    inputs.push_back(decode_v);                           // 2: decode_v
    inputs.push_back(override_k);                         // 3
    inputs.push_back(override_v);                         // 4
    inputs.push_back(override_far_idx);                   // 5
    inputs.push_back(override_count);                     // 6

    auto flat_outputs = array::make_arrays(
        {{total_bh_q, D_}, {total_bh_q}, {total_bh_q}},
        {float16, int32, int32},
        prim,
        inputs);

    return {
        reshape(flat_outputs[0], {B_, H_q, 1, D_}),
        reshape(flat_outputs[1], {B_, H_q}),
        reshape(flat_outputs[2], {B_, H_q}),
    };
}

} // namespace foveated
