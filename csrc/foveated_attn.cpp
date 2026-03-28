// Foveated 2-tier attention C++ extension for Apple Silicon.
//
// Near (fp16 K+V) + Far (fp8 E4M3 K, int4 V). Precompiled metallib with
// templated kernels specialized via Metal function constants.
//
// Promotions go directly into near-tier headroom slots in the blob.
// No override buffers — the kernel reads near_valid[h] dynamically.

#include "foveated_attn.h"

#include <cmath>
#include <sstream>

#include "mlx/backend/metal/device.h"

namespace foveated {

using namespace mlx::core;

// ============================================================================
// Helpers
// ============================================================================

static size_t align16(size_t x) { return (x + 15) & ~(size_t)15; }

static int adaptive_split_size(int s_total) {
    const int base = 256, max_splits = 16;
    if (s_total <= base * max_splits) return base;
    return ((s_total + max_splits - 1) / max_splits + 255) / 256 * 256;
}

static int round_up_max_splits(int num_splits) {
    if (num_splits <= 1) return 1;
    if (num_splits <= 2) return 2;
    if (num_splits <= 4) return 4;
    if (num_splits <= 8) return 8;
    return 16;
}


// ============================================================================
// FoveatedPrimitive::eval_gpu
// ============================================================================

void FoveatedPrimitive::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    auto& s = stream();
    auto& d = metal::device(s.device);

    for (auto& o : outputs)
        o.set_data(allocator::malloc(o.nbytes()));

    auto& enc = d.get_command_encoder(s.index);
    enc.set_compute_pipeline_state(pipeline_);

    enc.set_buffer(blob_buf_, 0, blob_offset_);

    for (int i = 0; i < (int)inputs.size(); i++)
        enc.set_input_array(inputs[i], 1 + i);

    enc.set_output_array(outputs[0], 4);
    enc.set_output_array(outputs[1], 5);
    enc.set_output_array(outputs[2], 6);

    enc.set_bytes(params_, 7);
    enc.set_bytes(blob_offsets_, 8);

    enc.dispatch_threadgroups(
        MTL::Size(total_bh_q_, 1, 1),
        MTL::Size(num_splits_ * 32, 1, 1));
}


// ============================================================================
// Blob builder
// ============================================================================

static BlobLayout build_blob(
    array& blob_out,
    const array& near_k, const array& near_v,
    const array& far_k, const array& far_v,
    const array& far_v_scale, const array& far_v_zero,
    const array& near_valid)
{
    auto fvs = astype(far_v_scale, float16);
    auto fvz = astype(far_v_zero, float16);
    auto nv = astype(near_valid, uint32);
    eval({near_k, near_v, far_k, far_v, fvs, fvz, nv});

    BlobLayout layout{};
    size_t off = 0;
    auto place = [&](size_t bytes) -> size_t {
        size_t pos = off;
        off = align16(off + bytes);
        return pos;
    };

    layout.near_k      = place(near_k.nbytes());
    layout.near_v      = place(near_v.nbytes());
    layout.far_k       = place(far_k.nbytes());
    layout.far_v       = place(far_v.nbytes());
    layout.far_v_scale = place(fvs.nbytes());
    layout.far_v_zero  = place(fvz.nbytes());
    layout.near_valid  = place(nv.nbytes());
    layout.total       = off;

    blob_out = zeros({(int)layout.total}, uint8);
    eval(blob_out);

    auto* dst = blob_out.data<uint8_t>();
    auto copy_in = [&](size_t offset, const array& src) {
        std::memcpy(dst + offset, src.data<uint8_t>(), src.nbytes());
    };
    copy_in(layout.near_k,      near_k);
    copy_in(layout.near_v,      near_v);
    copy_in(layout.far_k,       far_k);
    copy_in(layout.far_v,       far_v);
    copy_in(layout.far_v_scale, fvs);
    copy_in(layout.far_v_zero,  fvz);
    copy_in(layout.near_valid,  nv);

    return layout;
}


// ============================================================================
// FoveatedHandle
// ============================================================================

FoveatedHandle::FoveatedHandle(
    const array& near_k, const array& near_v,
    const array& far_k, const array& far_v,
    const array& far_v_scale, const array& far_v_zero,
    const array& near_valid,
    float spike_margin,
    const std::string& metallib_path)
    : blob_(array({1}, uint8)),
      blob_buf_(nullptr), blob_offset_(0),
      B_(near_k.shape(0)), H_kv_(near_k.shape(1)), D_(near_k.shape(3)),
      N_near_(near_k.shape(2)), N_far_(far_k.shape(2)),
      N_static_(near_k.shape(2) + far_k.shape(2)),
      spike_margin_(spike_margin),
      metallib_path_(metallib_path)
{
    auto nk = astype(near_k, float16);
    auto nv = astype(near_v, float16);

    auto layout = build_blob(blob_, nk, nv, far_k, far_v,
                             far_v_scale, far_v_zero, near_valid);

    blob_buf_ = static_cast<const MTL::Buffer*>(blob_.buffer().ptr());
    blob_offset_ = blob_.offset();

    blob_offsets_.near_k      = layout.near_k;
    blob_offsets_.near_v      = layout.near_v;
    blob_offsets_.far_k       = layout.far_k;
    blob_offsets_.far_v       = layout.far_v;
    blob_offsets_.far_v_scale = layout.far_v_scale;
    blob_offsets_.far_v_zero  = layout.far_v_zero;
    blob_offsets_.near_valid  = layout.near_valid;
}

std::vector<array> FoveatedHandle::operator()(
    const array& query,
    const array& decode_k, const array& decode_v)
{
    // Query cast to fp16 for kernel; output is fp16, caller casts if needed.
    // The fp16→bf16 cast in the interceptor also acts as a native MLX graph
    // node that enables pipelining with downstream FFN ops.
    auto q = (query.dtype() == float16) ? query : astype(query, float16);
    auto dk = (decode_k.dtype() == float16) ? decode_k : astype(decode_k, float16);
    auto dv = (decode_v.dtype() == float16) ? decode_v : astype(decode_v, float16);

    int H_q = q.shape(1);
    int n_decode = dk.shape(2);
    int total_bh_q = B_ * H_q;
    int S_total = N_static_ + n_decode;
    int split_size = adaptive_split_size(S_total);
    int num_splits = (S_total + split_size - 1) / split_size;
    int max_splits = round_up_max_splits(num_splits);

    uint64_t cache_key = ((uint64_t)split_size << 32) | (uint64_t)num_splits;
    auto it = pipelines_.find(cache_key);
    MTL::ComputePipelineState* pipeline;

    if (it != pipelines_.end()) {
        pipeline = it->second;
    } else {
        auto& d = metal::device(Device::gpu);
        auto* lib = d.get_library("foveated_attn", metallib_path_);

        std::string kernel_name = "foveated_2tier_f16_d" + std::to_string(D_)
                                + "_s" + std::to_string(max_splits);

        uint32_t fc_vals[7] = {
            (uint32_t)N_near_, (uint32_t)N_far_,
            (uint32_t)H_q, (uint32_t)H_kv_, (uint32_t)(H_q / H_kv_),
            (uint32_t)split_size, (uint32_t)num_splits
        };
        metal::MTLFCList fc = {
            {&fc_vals[0], MTL::DataTypeUInt, 0},
            {&fc_vals[1], MTL::DataTypeUInt, 1},
            {&fc_vals[2], MTL::DataTypeUInt, 2},
            {&fc_vals[3], MTL::DataTypeUInt, 3},
            {&fc_vals[4], MTL::DataTypeUInt, 4},
            {&fc_vals[5], MTL::DataTypeUInt, 5},
            {&fc_vals[6], MTL::DataTypeUInt, 6},
        };

        std::ostringstream hash;
        hash << kernel_name << "_" << N_near_ << "_" << N_far_
             << "_" << H_q << "_" << H_kv_
             << "_" << split_size << "_" << num_splits;

        pipeline = d.get_kernel(kernel_name, lib, hash.str(), fc);
        pipelines_[cache_key] = pipeline;
    }

    auto q_flat = reshape(q, {total_bh_q, D_});

    FoveatedParams params;
    params.total_bh_q = total_bh_q;
    params.n_decode = n_decode;
    params.spike_margin = spike_margin_;

    auto prim = std::make_shared<FoveatedPrimitive>(
        default_stream(Device::gpu),
        pipeline,
        blob_buf_,
        blob_offset_,
        blob_offsets_,
        params,
        total_bh_q,
        num_splits);

    return array::make_arrays(
        {{B_, H_q, 1, D_}, {B_, H_q}, {B_, H_q}},
        {float16, int32, int32},
        prim,
        {q_flat, dk, dv});
}

BlobWriteInfo FoveatedHandle::get_blob_info() const {
    BlobWriteInfo info;
    // const_cast is safe: blob_ is mutable unified memory that the worker
    // writes to. The handle keeps blob_ alive via its mlx::core::array member.
    info.blob_ptr = const_cast<uint8_t*>(blob_.data<uint8_t>());
    info.near_k_offset = blob_offsets_.near_k;
    info.near_v_offset = blob_offsets_.near_v;
    info.near_valid_offset = blob_offsets_.near_valid;
    info.N_near_alloc = N_near_;
    info.H_kv = H_kv_;
    info.D = D_;
    return info;
}

// ============================================================================
// TurboPrimitive::eval_gpu
// ============================================================================

void TurboPrimitive::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    // inputs: [query_flat, decode_k, decode_v, Pi, S_mat, centroids]
    auto& s = stream();
    auto& d = metal::device(s.device);

    for (auto& o : outputs)
        o.set_data(allocator::malloc(o.nbytes()));

    auto& enc = d.get_command_encoder(s.index);
    enc.set_compute_pipeline_state(pipeline_);

    enc.set_buffer(blob_buf_, 0, blob_offset_);

    // query, decode_k, decode_v → buffers 1-3
    for (int i = 0; i < 3; i++)
        enc.set_input_array(inputs[i], 1 + i);

    enc.set_output_array(outputs[0], 4);   // out
    enc.set_output_array(outputs[1], 5);   // spike_flags
    enc.set_output_array(outputs[2], 6);   // spike_tokens

    enc.set_bytes(params_, 7);
    enc.set_bytes(blob_offsets_, 8);

    // Pi, S_mat → buffers 9-10 (device memory)
    enc.set_input_array(inputs[3], 9);
    enc.set_input_array(inputs[4], 10);

    // centroids → buffer 11 (constant memory, 4 floats = 16 bytes)
    const auto& cents = inputs[5];
    enc.set_bytes(cents.data<float>(), cents.nbytes(), 11);

    enc.dispatch_threadgroups(
        MTL::Size(total_bh_q_, 1, 1),
        MTL::Size(num_splits_ * 32, 1, 1));
}


// ============================================================================
// Turbo blob builder
// ============================================================================

static TurboBlobLayout build_turbo_blob(
    array& blob_out,
    const array& near_k, const array& near_v,
    const array& far_k_indices, const array& far_k_signs,
    const array& far_k_norm, const array& far_k_gamma,
    const array& far_v_packed, const array& far_v_scale,
    const array& near_valid)
{
    auto nv = astype(near_valid, uint32);
    auto fkn = astype(far_k_norm, float16);
    auto fkg = astype(far_k_gamma, float16);
    auto fvs = astype(far_v_scale, float16);
    eval({near_k, near_v, far_k_indices, far_k_signs, fkn, fkg,
          far_v_packed, fvs, nv});

    TurboBlobLayout layout{};
    size_t off = 0;
    auto place = [&](size_t bytes) -> size_t {
        size_t pos = off;
        off = align16(off + bytes);
        return pos;
    };

    layout.near_k        = place(near_k.nbytes());
    layout.near_v        = place(near_v.nbytes());
    layout.far_k_indices = place(far_k_indices.nbytes());
    layout.far_k_signs   = place(far_k_signs.nbytes());
    layout.far_k_norm    = place(fkn.nbytes());
    layout.far_k_gamma   = place(fkg.nbytes());
    layout.far_v_packed  = place(far_v_packed.nbytes());
    layout.far_v_scale   = place(fvs.nbytes());
    layout.near_valid    = place(nv.nbytes());
    layout.total         = off;

    blob_out = zeros({(int)layout.total}, uint8);
    eval(blob_out);

    auto* dst = blob_out.data<uint8_t>();
    auto copy_in = [&](size_t offset, const array& src) {
        std::memcpy(dst + offset, src.data<uint8_t>(), src.nbytes());
    };
    copy_in(layout.near_k,        near_k);
    copy_in(layout.near_v,        near_v);
    copy_in(layout.far_k_indices, far_k_indices);
    copy_in(layout.far_k_signs,   far_k_signs);
    copy_in(layout.far_k_norm,    fkn);
    copy_in(layout.far_k_gamma,   fkg);
    copy_in(layout.far_v_packed,  far_v_packed);
    copy_in(layout.far_v_scale,   fvs);
    copy_in(layout.near_valid,    nv);

    return layout;
}


// ============================================================================
// TurboFoveatedHandle
// ============================================================================

TurboFoveatedHandle::TurboFoveatedHandle(
    const array& near_k, const array& near_v,
    const array& far_k_indices, const array& far_k_signs,
    const array& far_k_norm, const array& far_k_gamma,
    const array& far_v_packed, const array& far_v_scale,
    const array& near_valid,
    const array& Pi, const array& S_mat, const array& centroids,
    float spike_margin,
    const std::string& metallib_path)
    : blob_(array({1}, uint8)),
      blob_buf_(nullptr), blob_offset_(0),
      Pi_(astype(Pi, float32)),
      S_mat_(astype(S_mat, float32)),
      centroids_(astype(centroids, float32)),
      B_(near_k.shape(0)), H_kv_(near_k.shape(1)), D_(near_k.shape(3)),
      N_near_(near_k.shape(2)), N_far_(far_k_indices.shape(2)),
      N_static_(near_k.shape(2) + far_k_indices.shape(2)),
      spike_margin_(spike_margin),
      metallib_path_(metallib_path)
{
    auto nk = astype(near_k, float16);
    auto nv = astype(near_v, float16);

    auto layout = build_turbo_blob(blob_, nk, nv,
                                   far_k_indices, far_k_signs,
                                   far_k_norm, far_k_gamma,
                                   far_v_packed, far_v_scale,
                                   near_valid);

    blob_buf_ = static_cast<const MTL::Buffer*>(blob_.buffer().ptr());
    blob_offset_ = blob_.offset();

    blob_offsets_.near_k        = layout.near_k;
    blob_offsets_.near_v        = layout.near_v;
    blob_offsets_.far_k_indices = layout.far_k_indices;
    blob_offsets_.far_k_signs   = layout.far_k_signs;
    blob_offsets_.far_k_norm    = layout.far_k_norm;
    blob_offsets_.far_k_gamma   = layout.far_k_gamma;
    blob_offsets_.far_v_packed  = layout.far_v_packed;
    blob_offsets_.far_v_scale   = layout.far_v_scale;
    blob_offsets_.near_valid    = layout.near_valid;

    // Ensure constant arrays are materialized
    eval({Pi_, S_mat_, centroids_});
}

std::vector<array> TurboFoveatedHandle::operator()(
    const array& query,
    const array& decode_k, const array& decode_v)
{
    auto q = (query.dtype() == float16) ? query : astype(query, float16);
    auto dk = (decode_k.dtype() == float16) ? decode_k : astype(decode_k, float16);
    auto dv = (decode_v.dtype() == float16) ? decode_v : astype(decode_v, float16);

    int H_q = q.shape(1);
    int n_decode = dk.shape(2);
    int total_bh_q = B_ * H_q;
    int S_total = N_static_ + n_decode;
    int split_size = adaptive_split_size(S_total);
    int num_splits = (S_total + split_size - 1) / split_size;
    int max_splits = round_up_max_splits(num_splits);

    uint64_t cache_key = ((uint64_t)split_size << 32) | (uint64_t)num_splits;
    auto it = pipelines_.find(cache_key);
    MTL::ComputePipelineState* pipeline;

    if (it != pipelines_.end()) {
        pipeline = it->second;
    } else {
        auto& d = metal::device(Device::gpu);
        auto* lib = d.get_library("foveated_attn", metallib_path_);

        std::string kernel_name = "foveated_2tier_turbo_f16_d"
                                + std::to_string(D_)
                                + "_s" + std::to_string(max_splits);

        uint32_t fc_vals[7] = {
            (uint32_t)N_near_, (uint32_t)N_far_,
            (uint32_t)H_q, (uint32_t)H_kv_, (uint32_t)(H_q / H_kv_),
            (uint32_t)split_size, (uint32_t)num_splits
        };
        metal::MTLFCList fc = {
            {&fc_vals[0], MTL::DataTypeUInt, 0},
            {&fc_vals[1], MTL::DataTypeUInt, 1},
            {&fc_vals[2], MTL::DataTypeUInt, 2},
            {&fc_vals[3], MTL::DataTypeUInt, 3},
            {&fc_vals[4], MTL::DataTypeUInt, 4},
            {&fc_vals[5], MTL::DataTypeUInt, 5},
            {&fc_vals[6], MTL::DataTypeUInt, 6},
        };

        std::ostringstream hash;
        hash << kernel_name << "_" << N_near_ << "_" << N_far_
             << "_" << H_q << "_" << H_kv_
             << "_" << split_size << "_" << num_splits;

        pipeline = d.get_kernel(kernel_name, lib, hash.str(), fc);
        pipelines_[cache_key] = pipeline;
    }

    auto q_flat = reshape(q, {total_bh_q, D_});

    FoveatedParams params;
    params.total_bh_q = total_bh_q;
    params.n_decode = n_decode;
    params.spike_margin = spike_margin_;

    auto prim = std::make_shared<TurboPrimitive>(
        default_stream(Device::gpu),
        pipeline,
        blob_buf_,
        blob_offset_,
        blob_offsets_,
        params,
        total_bh_q,
        num_splits);

    return array::make_arrays(
        {{B_, H_q, 1, D_}, {B_, H_q}, {B_, H_q}},
        {float16, int32, int32},
        prim,
        {q_flat, dk, dv, Pi_, S_mat_, centroids_});
}

BlobWriteInfo TurboFoveatedHandle::get_blob_info() const {
    BlobWriteInfo info;
    info.blob_ptr = const_cast<uint8_t*>(blob_.data<uint8_t>());
    info.near_k_offset = blob_offsets_.near_k;
    info.near_v_offset = blob_offsets_.near_v;
    info.near_valid_offset = blob_offsets_.near_valid;
    info.N_near_alloc = N_near_;
    info.H_kv = H_kv_;
    info.D = D_;
    return info;
}

} // namespace foveated
