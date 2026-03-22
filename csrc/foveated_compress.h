#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace MTL { class ComputePipelineState; }

namespace foveated {

// Must match Metal kernel structs exactly.
struct FovealParams {
    uint32_t S;
    uint32_t n_sinks;
    uint32_t fov_mid_start;
    uint32_t fov_mid_count;
    uint32_t window_start;
    uint32_t window_count;
    uint32_t n_near_padded;
    uint32_t near_actual;
};

struct CompressParams {
    uint32_t S;
    uint32_t src_offset;
    uint32_t N;
};


// ============================================================================
// CompressPrimitive
// ============================================================================

enum class CompressMode { Foveal, Fp8, Int4 };

class CompressPrimitive : public mlx::core::Primitive {
 public:
    CompressPrimitive(
        mlx::core::Stream stream,
        MTL::ComputePipelineState* pipeline,
        CompressMode mode,
        FovealParams fov_params,
        CompressParams comp_params,
        int B_times_H,
        int N_dispatch)
        : Primitive(stream),
          pipeline_(pipeline),
          mode_(mode),
          fov_params_(fov_params),
          comp_params_(comp_params),
          B_times_H_(B_times_H),
          N_dispatch_(N_dispatch) {}

    void eval_cpu(
        const std::vector<mlx::core::array>&,
        std::vector<mlx::core::array>&) override {
        throw std::runtime_error("CompressPrimitive only runs on GPU");
    }

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    const char* name() const override { return "CompressPrimitive"; }

    bool is_equivalent(const Primitive& other) const override {
        auto* o = dynamic_cast<const CompressPrimitive*>(&other);
        return o && pipeline_ == o->pipeline_ && mode_ == o->mode_;
    }

 private:
    MTL::ComputePipelineState* pipeline_;
    CompressMode mode_;
    FovealParams fov_params_;
    CompressParams comp_params_;
    int B_times_H_;
    int N_dispatch_;
};


// ============================================================================
// CompressHandle: 2-tier compression
//
//   Near: gather sinks + high-importance mid + window → fp16
//   Far K: fp16 → fp8 E4M3 (register-level quantization)
//   Far V: fp16 → int4 per-token (nibble-packed + scale/zero)
// ============================================================================

struct TierConfig {
    float near_pct = 0.10f;
    int n_sinks = 4;
    int window_size = 32;
    float promo_headroom_pct = 0.5f;
    int promo_headroom_min = 4;
};

struct CompressedLayer {
    mlx::core::array near_k, near_v;       // fp16 (B, H, N_near_padded, D)
    mlx::core::array far_k;                // uint8 (B, H, N_far, D) — fp8 E4M3
    mlx::core::array far_v;                // uint8 (B, H, N_far, D/2) — int4 packed
    mlx::core::array far_v_scale;          // fp16 (B, H, N_far)
    mlx::core::array far_v_zero;           // fp16 (B, H, N_far)
    mlx::core::array near_valid;           // int32 (H,)
    int n_near_actual = 0;

    CompressedLayer(
        mlx::core::array near_k, mlx::core::array near_v,
        mlx::core::array far_k, mlx::core::array far_v,
        mlx::core::array far_v_scale, mlx::core::array far_v_zero,
        mlx::core::array near_valid, int n_near_actual)
        : near_k(std::move(near_k)), near_v(std::move(near_v)),
          far_k(std::move(far_k)), far_v(std::move(far_v)),
          far_v_scale(std::move(far_v_scale)), far_v_zero(std::move(far_v_zero)),
          near_valid(std::move(near_valid)), n_near_actual(n_near_actual) {}
};

class CompressHandle {
 public:
    CompressHandle(const TierConfig& cfg, const std::string& metallib_path);

    CompressedLayer compress_layer(
        const mlx::core::array& keys,
        const mlx::core::array& values);

    std::vector<CompressedLayer> compress_all(
        const std::vector<mlx::core::array>& all_keys,
        const std::vector<mlx::core::array>& all_values);

 private:
    MTL::ComputePipelineState* get_pipeline(
        const std::string& kernel_name, int D);

    TierConfig cfg_;
    std::string metallib_path_;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
};

} // namespace foveated
