# Roadmap: Native MLX Integration

## The Problem

The fused foveated kernel matches Apple's SDPA at the compute level and
reads 2-4x fewer bytes from memory. But end-to-end decode on a real model
is 4.4x slower because every attention layer goes through MLX's
`CustomKernel::eval_gpu` — a generic dispatch path designed for arbitrary
user kernels, not for a tight inner loop called 24 times per step.

Apple's `ScaledDotProductAttention` bypasses this entirely. It is a
first-class primitive in MLX's evaluator: direct buffer binding, no
contiguity loops, no source string caching, pre-compiled Metal pipeline
from the built-in metallib. That is the performance we need to match.

## What CustomKernel::eval_gpu Does (Per Call)

From the MLX source (`mlx/backend/metal/custom_kernel.cpp`):

1. **Output allocation or init_value fill.** Allocates Metal buffers for
   each output via `allocator::malloc`. If `init_value` is set, fills
   the buffer with that value (extra GPU dispatch). Our merged kernel
   uses `std::nullopt` so this is just malloc × 3 outputs.

2. **Contiguity check loop.** For each of N inputs: read the array's
   `flags().row_contiguous`. If false and `ensure_row_contiguous` is
   true, copy the entire array to a contiguous temporary. Even when all
   inputs are contiguous (ours are), the loop runs and checks every flag.

3. **Library cache validation.** Hash map lookup by kernel name. If
   found, compare the stored source string against the current source
   string (character-by-character, ~10KB for our kernel). If different,
   clear the cached library and recompile. This comparison happens every
   single `eval_gpu` call.

4. **Library + kernel retrieval.** `d.get_library(name, builder)` does
   a hash map lookup. If cached, returns immediately (builder lambda not
   called). Then `d.get_kernel(name, lib)` does another hash map lookup
   for the compiled pipeline state.

5. **Buffer binding loop.** For each input: `set_input_array(arr, idx)`
   which extracts the Metal buffer pointer, registers the buffer for
   dependency tracking (checks against `prev_outputs_` set for barrier
   insertion), and calls Metal's `setBuffer`. If `shape_infos_` flags
   are set for that input, also sends shape, strides, and ndim as
   additional buffer arguments (up to 3 extra bindings per input).

6. **Output binding.** For each output: `set_output_array(out, idx)`.

7. **Grid validation.** Checks threadgroup size against
   `maxTotalThreadsPerThreadgroup`. Computes adjusted grid/group dims.

8. **Dispatch.** `dispatch_threads(grid, group)` — note: `threads`,
   not `threadgroups`. Metal adjusts the grid at dispatch time.

9. **Temporary registration.** `add_temporaries(copies, stream_index)`
   to track the lifetime of any contiguity copies.

## What ScaledDotProductAttention::eval_gpu Does

From `mlx/backend/metal/scaled_dot_product_attention.cpp`:

1. **Smart contiguity check.** Checks only the specific stride pattern
   needed (last dim == 1, batch/head contiguous). Does NOT iterate all
   inputs generically. Copies only when truly needed. Can donate the
   query buffer to the output (zero-copy when possible).

2. **Kernel lookup.** Builds a short name string (e.g.,
   `sdpa_vector_float16_128_128`), looks up from the pre-compiled
   default metallib. No source string comparison. No JIT. The metallib
   is loaded once at startup.

3. **Direct buffer binding.** Hardcoded: `set_input_array(q, 0)`,
   `set_input_array(k, 1)`, `set_input_array(v, 2)`,
   `set_output_array(out, 3)`. Then `set_bytes` for scalar params
   (scale, gqa_factor, strides). No loops, no shape_info checks.

4. **Dispatch.** `dispatch_threadgroups(grid, group)` — note:
   `threadgroups`, not `threads`. No runtime grid adjustment.

5. **No library cache management.** The kernel comes from the metallib
   which is immutable.

## The Gap

Per layer, the extra cost of `CustomKernel::eval_gpu` vs SDPA's direct
path is approximately 4ms. Over 24 layers, that is ~100ms per decode
step. On a 0.5B model where the entire step takes 30ms with standard
SDPA, this 100ms dominates. On a 7B model at 32K context where the step
might take 200-400ms, the 100ms is still significant but the kernel's
bandwidth savings start to compete.

The gap comes from:
- Generic input validation loops vs hardcoded binding
- Source string caching vs pre-compiled metallib
- `dispatch_threads` vs `dispatch_threadgroups`
- No buffer donation optimization
- More Metal buffer arguments (9 in our blob path vs 4 for SDPA)

None of these are individually catastrophic. Combined over 24 layers
with per-call overhead, they add up.

## The Plan: Fork MLX

Fork MLX v0.31.1 (matching our installed version). Add `foveated_sdpa`
as a first-class primitive alongside `scaled_dot_product_attention`.
Pin the fork in our project so `uv sync` builds from source.

### Phase 1: Minimal Primitive

**New files:**

```
mlx/fast_primitives.h     — add FoveatedSDPA class declaration
mlx/fast.h                — add foveated_sdpa() function declaration
mlx/backend/metal/foveated_sdpa.cpp — eval_gpu implementation
mlx/backend/metal/kernels/foveated_sdpa.metal — kernel source
mlx/backend/metal/kernels/CMakeLists.txt — add to metallib build
python/mlx/fast.cpp       — nanobind binding for foveated_sdpa()
```

**FoveatedSDPA primitive class:**

```cpp
class FoveatedSDPA : public Custom {
public:
    FoveatedSDPA(
        Stream stream,
        int n_fov, int n_per, int n_far,
        int head_dim, int h_q, int h_kv,
        int split_size, int num_splits, int max_ov,
        float scale, float spike_margin);

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override;

private:
    // Compile-time tier config (no runtime lookup needed)
    int n_fov_, n_per_, n_far_;
    int head_dim_, h_q_, h_kv_;
    int split_size_, num_splits_, max_ov_;
    float scale_, spike_margin_;
};
```

**eval_gpu implementation** (mirrors `sdpa_vector` pattern):

```cpp
void FoveatedSDPA::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    auto& s = stream();
    auto& d = metal::device(s.device);

    // inputs: [query, foveal_k, foveal_v, periph_k, periph_v,
    //          periph_k_sz, periph_v_sz, far_k, far_v,
    //          far_k_sz, far_v_sz, foveal_valid,
    //          decode_k, decode_v,
    //          override_k, override_v, override_far_idx, override_count]
    //
    // OR: [query, blob, decode_k, decode_v,
    //       override_k, override_v, override_far_idx, override_count]

    auto& query = inputs[0];
    auto& out = outputs[0];

    // Donate query buffer to output if possible (like SDPA does)
    if (query.is_donatable() && query.flags().row_contiguous
        && query.size() == out.size()) {
        out.copy_shared_buffer(query);
    } else {
        out.set_data(allocator::malloc(out.nbytes()));
    }
    outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));
    outputs[2].set_data(allocator::malloc(outputs[2].nbytes()));

    // Build kernel name from compile-time config
    std::string kname = "foveated_sdpa_";
    kname += std::to_string(head_dim_);
    kname += "_" + std::to_string(num_splits_);

    // Get kernel from pre-compiled metallib (no JIT, no source caching)
    auto& enc = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname);
    enc.set_compute_pipeline_state(kernel);

    // Direct buffer binding — no loops, no shape_info
    int idx = 0;
    for (int i = 0; i < inputs.size(); i++)
        enc.set_input_array(inputs[i], idx++);
    for (auto& o : outputs)
        enc.set_output_array(o, idx++);

    // Scalar params as raw bytes
    int n_decode = inputs[/* decode_k idx */].shape(2);
    int total_bh_q = query.shape(0);
    uint32_t params[2] = {(uint32_t)total_bh_q, (uint32_t)n_decode};
    enc.set_bytes(params, 2, idx++);

    // dispatch_threadgroups (not dispatch_threads)
    enc.dispatch_threadgroups(
        MTL::Size(total_bh_q, 1, 1),
        MTL::Size(num_splits_ * 32, 1, 1));
}
```

**Metal kernel** (`foveated_sdpa.metal`):

Extract from our Python string constants into a proper `.metal` file.
The kernel body is identical to `_MERGED_SOURCE` + `_TIER_PROCESSING`.
Compile into the metallib at build time via `mlx_build_metallib` in
CMakeLists.txt. Use Metal function constants for `NUM_SPLITS` and
`HEAD_DIM` to specialize without separate source strings.

```metal
constant int NUM_SPLITS [[function_constant(0)]];
constant int HEAD_DIM   [[function_constant(1)]];
// ... other tier constants

[[kernel]] void foveated_sdpa_128_3(
    const device half* query [[buffer(0)]],
    const device uint8_t* blob [[buffer(1)]],
    // ... remaining inputs
    device half* out [[buffer(N)]],
    device int* spike_flags [[buffer(N+1)]],
    device int* spike_tokens [[buffer(N+2)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos [[thread_position_in_threadgroup]]
) {
    // Identical to current merged kernel body
}
```

**Python binding** (in `python/mlx/fast.cpp`):

```cpp
m.def(
    "foveated_sdpa",
    &fast::foveated_sdpa,
    "query"_a, "blob"_a,
    "decode_k"_a, "decode_v"_a,
    "override_k"_a, "override_v"_a,
    "override_far_idx"_a, "override_count"_a,
    "n_fov"_a, "n_per"_a, "n_far"_a,
    "head_dim"_a, "h_q"_a, "h_kv"_a,
    "split_size"_a, "num_splits"_a, "max_ov"_a,
    "scale"_a, "spike_margin"_a = 0.5f,
    nb::kw_only(), "stream"_a = nb::none());
```

### Phase 2: Integration

Replace the SDPA interceptor call with `mx.fast.foveated_sdpa()`.
Since it is a real `mx.fast` function (not a CustomKernel), the model's
forward pass creates `FoveatedSDPA` primitive nodes instead of
`CustomKernel` nodes. The evaluator processes them through the fast
path — no contiguity loops, no source caching, no `dispatch_threads`.

The interceptor becomes:

```python
def _fused_sdpa_interceptor(queries, keys, values, *, scale, **kwargs):
    if queries.shape[2] > 1:
        return _original_sdpa(queries, keys, values, scale=scale, **kwargs)
    layer_idx = state.layer_counter; state.layer_counter += 1
    fov_layer = state.fov_cache.layers[layer_idx]
    return mx.fast.foveated_sdpa(
        queries, fov_layer._blob,
        fov_layer.decode_k, fov_layer.decode_v,
        *fov_layer._override_arrays(),
        n_fov=fov_layer.N_fov, ...)
```

One C++ function call per layer (same as standard SDPA), creating one
graph node with the same evaluator privileges.

### Phase 3: Pre-Compiled Metallib

Move the kernel from JIT string to the pre-compiled metallib. Add to
`mlx/backend/metal/kernels/CMakeLists.txt`:

```cmake
build_kernel(foveated_sdpa foveated_sdpa.metal foveated_sdpa.h)
```

Use Metal function constants for specialization:

```cmake
# Generate variants for common HEAD_DIM values
foreach(HD IN ITEMS 64 128)
    foreach(NS IN ITEMS 1 2 3 4 8 16)
        # Function constants set at pipeline creation
    endforeach()
endforeach()
```

This eliminates ALL JIT compilation. The kernel is compiled once when
MLX is built, loaded from the metallib at startup, and dispatched
directly.

### Phase 4: Buffer Donation

Implement query buffer donation (like SDPA does):

```cpp
if (query.is_donatable() && query.flags().row_contiguous
    && query.size() == out.size()) {
    out.copy_shared_buffer(query);
}
```

This eliminates one `malloc` per layer per step (the output allocation).
With 24 layers, that is 24 fewer Metal buffer allocations per decode
step.

### Phase 5: Benchmark and Validate

With the native primitive in place:

1. Re-run the kernel benchmark at all context lengths. The SDPA cliff
   at 32K becomes our baseline (we cannot control Apple's kernel), but
   our kernel should now dispatch with the same per-layer overhead as
   SDPA — any remaining gap is pure GPU compute difference.

2. End-to-end decode on 0.5B, 3B, 7B models. Measure tok/s, not just
   kernel ms. The 4.4x overhead should collapse to near 1.0x at short
   context (same dispatch path), with bandwidth savings dominating at
   long context.

3. Memory comparison. At 32K+ context, show the KV cache size difference
   (fp16 vs foveated) and demonstrate fitting a model that would
   otherwise OOM.

4. Quality. Re-run needle, LongBench, PPL at multiple context lengths
   to confirm no regression from the native integration.

## Build System

Pin the MLX fork as a dependency:

```toml
# pyproject.toml
[tool.uv.sources]
mlx = { git = "https://github.com/your-username/mlx.git", branch = "foveated" }
```

Or for local development:

```toml
mlx = { path = "../mlx-fork" }
```

`uv sync` builds MLX from source with our additions. The C++ extension
(`csrc/`) becomes unnecessary — the kernel dispatches through MLX's own
evaluator.

## Expected Outcome

The native primitive eliminates the 100ms dispatch overhead. The
end-to-end performance becomes:

- **Short context (< 16K):** Near parity with standard SDPA. Same
  dispatch path, same evaluator treatment. The fused kernel reads a few
  more bytes (packed scale/zero arrays) but the compute is equivalent.

- **Long context (> 16K):** Bandwidth savings from INT8/INT4 dominate.
  The exact speedup depends on the model, context length, and hardware
  — the SDPA cliff issue needs investigation on larger machines to
  separate our savings from baseline degradation.

- **Memory:** 2.2x compression at all context lengths, enabling longer
  context or larger models on the same hardware.

The overhead gap between our kernel and Apple's SDPA becomes a pure
GPU compute comparison, not a dispatch path comparison. That is the
honest benchmark we need.

## Timeline Estimate

| Phase | Effort | Description |
|-------|--------|-------------|
| 1 | 2-3 days | Minimal primitive + JIT kernel in fork |
| 2 | 1 day | Python integration, replace interceptor |
| 3 | 1-2 days | Pre-compiled metallib, function constants |
| 4 | 0.5 day | Buffer donation |
| 5 | 1-2 days | Full benchmark suite on larger hardware |

Total: ~1 week of focused work, assuming familiarity with the MLX
codebase (which we now have from reading the source).
