// Foveated 2-tier compression C++ extension for Apple Silicon.
//
// Three compression modes:
//   Foveal:  gather non-contiguous fp16 segments into padded near buffer
//   Fp8:     fp16 → fp8 E4M3 with round-to-nearest-even (far K)
//   Int4:    fp16 → int4 per-token quantize + nibble pack (far V)

#include "foveated_compress.h"

#include <algorithm>

#include "mlx/backend/metal/device.h"

namespace foveated {

using namespace mlx::core;

// ============================================================================
// CompressPrimitive::eval_gpu
// ============================================================================

void CompressPrimitive::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    auto& s = stream();
    auto& d = metal::device(s.device);

    for (auto& o : outputs)
        o.set_data(allocator::malloc(o.nbytes()));

    auto& enc = d.get_command_encoder(s.index);
    enc.set_compute_pipeline_state(pipeline_);

    if (mode_ == CompressMode::Foveal) {
        enc.set_input_array(inputs[0], 0);
        enc.set_output_array(outputs[0], 1);
        enc.set_bytes(fov_params_, 2);

        enc.dispatch_threadgroups(
            MTL::Size(fov_params_.n_near_padded, B_times_H_, 1),
            MTL::Size(32, 1, 1));

    } else if (mode_ == CompressMode::Fp8) {
        enc.set_input_array(inputs[0], 0);
        enc.set_output_array(outputs[0], 1);
        enc.set_bytes(comp_params_, 2);

        enc.dispatch_threadgroups(
            MTL::Size(N_dispatch_, B_times_H_, 1),
            MTL::Size(32, 1, 1));

    } else {  // Int4
        enc.set_input_array(inputs[0], 0);
        enc.set_output_array(outputs[0], 1);  // packed nibbles
        enc.set_output_array(outputs[1], 2);  // scale
        enc.set_output_array(outputs[2], 3);  // zero
        enc.set_bytes(comp_params_, 4);

        enc.dispatch_threadgroups(
            MTL::Size(comp_params_.N, B_times_H_, 1),
            MTL::Size(32, 1, 1));
    }
}


// ============================================================================
// CompressHandle
// ============================================================================

CompressHandle::CompressHandle(
    const TierConfig& cfg,
    const std::string& metallib_path)
    : cfg_(cfg), metallib_path_(metallib_path) {}


MTL::ComputePipelineState* CompressHandle::get_pipeline(
    const std::string& kernel_name, int D)
{
    std::string cache_key = kernel_name + "_d" + std::to_string(D);

    auto it = pipelines_.find(cache_key);
    if (it != pipelines_.end())
        return it->second;

    auto& d = metal::device(Device::gpu);
    auto* lib = d.get_library("foveated_attn", metallib_path_);

    auto* pipeline = d.get_kernel(cache_key, lib, cache_key, {});
    pipelines_[cache_key] = pipeline;
    return pipeline;
}


CompressedLayer CompressHandle::compress_layer(
    const array& keys,
    const array& values)
{
    int B = keys.shape(0);
    int H = keys.shape(1);
    int S = keys.shape(2);
    int D = keys.shape(3);
    int BH = B * H;

    auto gpu_stream = default_stream(Device::gpu);

    // --- Tier boundary computation ---
    int n_sinks = std::min(cfg_.n_sinks, S);
    int window = std::min(cfg_.window_size, std::max(S - n_sinks, 0));
    int near_reserved = n_sinks + window;
    int R_total = std::max((int)(S * cfg_.near_pct), near_reserved);
    int far_total = S - R_total;
    if (far_total < 0) { far_total = 0; R_total = S; }

    int mid_start = n_sinks;
    int mid_end = window > 0 ? S - window : S;
    int mid_len = std::max(mid_end - mid_start, 0);
    int near_from_mid = std::min(std::max(R_total - near_reserved, 0), mid_len);

    int R_actual = n_sinks + near_from_mid + window;
    int headroom = std::max((int)(R_actual * cfg_.promo_headroom_pct),
                            cfg_.promo_headroom_min);
    int N_near_padded = R_actual + headroom;

    int near_mid_start = mid_end - near_from_mid;
    int actual_far_count = near_mid_start - mid_start;

    // --- Ensure fp16 + contiguous ---
    auto k = contiguous(astype(keys, float16));
    auto v = contiguous(astype(values, float16));

    // --- Build graph nodes (NO eval) ---

    // 1. Near K (gather non-contiguous segments)
    FovealParams fp{};
    fp.S = S;
    fp.n_sinks = n_sinks;
    fp.fov_mid_start = near_mid_start;
    fp.fov_mid_count = near_from_mid;
    fp.window_start = window > 0 ? S - window : S;
    fp.window_count = window;
    fp.n_near_padded = N_near_padded;
    fp.near_actual = R_actual;

    CompressParams dummy_cp{};
    auto* fov_pipeline = get_pipeline("compress_foveal", D);

    auto near_k_prim = std::make_shared<CompressPrimitive>(
        gpu_stream, fov_pipeline, CompressMode::Foveal,
        fp, dummy_cp, BH, N_near_padded);
    auto near_k_arr = array::make_arrays(
        {{B, H, N_near_padded, D}}, {float16}, near_k_prim, {k});

    // 2. Near V
    auto near_v_prim = std::make_shared<CompressPrimitive>(
        gpu_stream, fov_pipeline, CompressMode::Foveal,
        fp, dummy_cp, BH, N_near_padded);
    auto near_v_arr = array::make_arrays(
        {{B, H, N_near_padded, D}}, {float16}, near_v_prim, {v});

    // 3. Far K (fp16 → fp8 E4M3)
    array fk({0}, uint8);
    array fv({0}, uint8), fvs({0}, float16), fvz({0}, float16);

    if (actual_far_count > 0) {
        CompressParams cp{};
        cp.S = S;
        cp.src_offset = mid_start;
        cp.N = actual_far_count;

        auto* fp8_pipeline = get_pipeline("compress_fp8", D);
        auto fk_prim = std::make_shared<CompressPrimitive>(
            gpu_stream, fp8_pipeline, CompressMode::Fp8,
            fp, cp, BH, actual_far_count);
        auto fk_arr = array::make_arrays(
            {{B, H, actual_far_count, D}}, {uint8}, fk_prim, {k});
        fk = fk_arr[0];

        // 4. Far V (fp16 → int4 per-token packed)
        auto* int4_pipeline = get_pipeline("compress_int4", D);
        auto fv_prim = std::make_shared<CompressPrimitive>(
            gpu_stream, int4_pipeline, CompressMode::Int4,
            fp, cp, BH, actual_far_count);
        auto fv_arr = array::make_arrays(
            {{B, H, actual_far_count, D / 2}, {B, H, actual_far_count}, {B, H, actual_far_count}},
            {uint8, float16, float16}, fv_prim, {v});
        fv = fv_arr[0];
        fvs = fv_arr[1];
        fvz = fv_arr[2];
    } else {
        fk = zeros({B, H, 0, D}, uint8);
        fv = zeros({B, H, 0, D / 2}, uint8);
        fvs = zeros({B, H, 0}, float16);
        fvz = zeros({B, H, 0}, float16);
    }

    return CompressedLayer(
        near_k_arr[0], near_v_arr[0],
        fk, fv, fvs, fvz,
        full({H}, R_actual, int32), R_actual);
}


std::vector<CompressedLayer> CompressHandle::compress_all(
    const std::vector<array>& all_keys,
    const std::vector<array>& all_values)
{
    std::vector<CompressedLayer> layers;
    layers.reserve(all_keys.size());

    for (size_t i = 0; i < all_keys.size(); i++)
        layers.push_back(compress_layer(all_keys[i], all_values[i]));

    std::vector<array> all_outputs;
    for (auto& L : layers) {
        all_outputs.push_back(L.near_k);
        all_outputs.push_back(L.near_v);
        all_outputs.push_back(L.far_k);
        all_outputs.push_back(L.far_v);
        all_outputs.push_back(L.far_v_scale);
        all_outputs.push_back(L.far_v_zero);
        all_outputs.push_back(L.near_valid);
    }
    eval(all_outputs);

    return layers;
}

} // namespace foveated
