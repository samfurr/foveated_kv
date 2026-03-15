#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/backend/metal/device.h"

namespace foveated {

// Direct-dispatch primitive. eval_gpu encodes Metal commands directly
// via CommandEncoder — no fast::metal_kernel overhead.
class FoveatedPrimitive : public mlx::core::Primitive {
 public:
    FoveatedPrimitive(
        mlx::core::Stream stream,
        int n_fov, int n_per, int n_far,
        int head_dim, int h_q, int h_kv,
        int split_size, int max_ov,
        float spike_margin,
        // Pre-bound static arrays (stored as inputs[0..14])
        // Dynamic arrays are inputs[15..21]
        int n_static_inputs);

    void eval_cpu(const std::vector<mlx::core::array>& inputs,
                  std::vector<mlx::core::array>& outputs) override {
        throw std::runtime_error("FoveatedPrimitive: GPU only");
    }

    void eval_gpu(const std::vector<mlx::core::array>& inputs,
                  std::vector<mlx::core::array>& outputs) override;

    DEFINE_NAME(FoveatedPrimitive);
    bool is_equivalent(const Primitive& other) const override;

 private:
    int n_fov_, n_per_, n_far_;
    int head_dim_, h_q_, h_kv_;
    int split_size_, max_ov_;
    float spike_margin_;
    int n_static_;

    // Ensure Metal pipelines are compiled (lazy, cached globally)
    void ensure_pipelines_() const;
    std::string kernel_key_() const;
};

// Python-facing handle using direct CommandEncoder dispatch.
class FoveatedHandleDirect {
 public:
    FoveatedHandleDirect(
        const mlx::core::array& foveal_k, const mlx::core::array& foveal_v,
        const mlx::core::array& periph_k, const mlx::core::array& periph_v,
        const mlx::core::array& periph_k_scale, const mlx::core::array& periph_k_zero,
        const mlx::core::array& periph_v_scale, const mlx::core::array& periph_v_zero,
        const mlx::core::array& far_k, const mlx::core::array& far_v,
        const mlx::core::array& far_k_scale, const mlx::core::array& far_k_zero,
        const mlx::core::array& far_v_scale, const mlx::core::array& far_v_zero,
        const mlx::core::array& foveal_valid,
        float spike_margin, int max_ov);

    std::vector<mlx::core::array> operator()(
        const mlx::core::array& query,
        const mlx::core::array& decode_k, const mlx::core::array& decode_v,
        const mlx::core::array& override_k, const mlx::core::array& override_v,
        const mlx::core::array& override_far_idx, const mlx::core::array& override_count);

 private:
    // Pre-stored static arrays (15 total, some pre-reshaped)
    std::vector<mlx::core::array> static_arrays_;
    int B_, H_kv_, D_, N_fov_, N_per_, N_far_, N_static_;
    float spike_margin_;
    int max_ov_;
};

} // namespace foveated
