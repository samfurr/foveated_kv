#pragma once

#include <string>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"

namespace foveated {

struct Config {
    int n_fov, n_per, n_far;
    int head_dim, h_q, h_kv;
    int split_size;
    int max_ov;
    float spike_margin;
};

// Build the Metal kernel source string with config constants injected.
// Returns (splitk_source, reduce_source).
std::pair<std::string, std::string> build_kernel_sources(const Config& cfg);

// Foveated attention: takes all inputs (static + dynamic), returns
// (output, spike_flags, spike_tokens).
// This is the Phase 1 stateless API — all 22 inputs passed each call.
std::vector<mlx::core::array> foveated_attention(
    // Static tier arrays (15)
    const mlx::core::array& foveal_k,
    const mlx::core::array& foveal_v,
    const mlx::core::array& periph_k,
    const mlx::core::array& periph_v,
    const mlx::core::array& periph_k_scale,
    const mlx::core::array& periph_k_zero,
    const mlx::core::array& periph_v_scale,
    const mlx::core::array& periph_v_zero,
    const mlx::core::array& far_k,
    const mlx::core::array& far_v,
    const mlx::core::array& far_k_scale,
    const mlx::core::array& far_k_zero,
    const mlx::core::array& far_v_scale,
    const mlx::core::array& far_v_zero,
    const mlx::core::array& foveal_valid,
    // Dynamic arrays (7)
    const mlx::core::array& query,
    const mlx::core::array& decode_k,
    const mlx::core::array& decode_v,
    const mlx::core::array& override_k,
    const mlx::core::array& override_v,
    const mlx::core::array& override_far_idx,
    const mlx::core::array& override_count,
    // Config
    float spike_margin = 0.5f,
    int split_size = 256);

} // namespace foveated
