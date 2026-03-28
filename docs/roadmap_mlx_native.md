# Roadmap: Eliminating the Dispatch Overhead

## No Fork Required

The original plan called for forking MLX to add foveated attention as a
built-in primitive. Research into MLX's internals revealed that isn't
necessary. The existing C++ extension mechanism provides everything we
need — we were just using it wrong.

## What Went Wrong

Our C++ `FoveatedHandle` calls `mx.fast.metal_kernel()` internally.
That function creates `CustomKernel` graph nodes regardless of whether
it's called from Python or C++. Every `CustomKernel::eval_gpu` runs the
generic dispatch path: source string comparison, contiguity loops, hash
map lookups with mutex contention, `dispatch_threads`.

The handle eliminated Python overhead but still created the same slow
graph nodes. Moving from Python to C++ changed who builds the nodes, not
what kind of nodes get built.

## What We Should Do Instead

Subclass `mlx::core::Primitive` directly. The `eval_gpu` loads a
precompiled `.metallib` from disk and dispatches via `CommandEncoder` —
the exact same code path that `ScaledDotProductAttention` uses. No
`CustomKernel` involved anywhere.

We already built `FoveatedPrimitive` and proved it produces correct
output (cosine 1.0 vs the Python path). It was slower at the time
because of per-call temporary allocation and a misunderstanding about
Metal buffer access. Both issues are now solved.

## The Three Fixes

### 1. Precompiled Metallib (Eliminates Source Caching)

`CustomKernel::eval_gpu` compares the full kernel source string on
every call (~10KB character comparison) and looks up the JIT-compiled
library through a mutex-protected hash map.

The fix: compile the kernel to a `.metallib` at build time. Load it
once via `d.get_library("foveated_attn", path_to_metallib)` — the
path-based overload reads a binary file, no string comparison, no JIT.
Cached after first load.

```cmake
# In csrc/CMakeLists.txt — use MLX's build macro
mlx_build_metallib(
    TARGET foveated_metallib
    TITLE foveated_attn
    OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/foveated_attn.metal
    INCLUDE_DIRS ${MLX_INCLUDE_DIRS}
)
```

The kernel source moves from Python string constants to a proper
`.metal` file. Metal function constants replace the string-templated
compile-time constants:

```metal
constant int HEAD_DIM [[function_constant(0)]];
constant int NUM_SPLITS [[function_constant(1)]];
// ... tier sizes passed as function constants at pipeline creation
```

This lets one `.metallib` serve all tier configurations. Pipeline
specialization happens at `get_kernel` time via `MTLFCList`, same as
SDPA.

### 2. Blob as Tracked Input Array (Correct Dependency Tracking)

The original plan was to bypass `set_input_array` for static tier arrays
by using raw `set_buffer` with `MTL::Buffer*` pointers. In practice,
the blob is now passed as a tracked input array via `set_input_array`
to ensure correct MLX graph dependency tracking. This is simpler and
avoids subtle bugs with buffer lifetime management.

The 7 static tier arrays (near_k, near_v, far_k, far_v, far_v_scale,
far_v_zero, near_valid) are pre-packed into a single uint8 blob at
compression time. The blob is passed to `FoveatedPrimitive` as an input
array alongside the 3 dynamic inputs (query, decode_k, decode_v).

Override buffers have been eliminated; promotions go directly into the
blob's near-tier headroom via the C++ PromotionPipeline.

### 3. Merged Kernel + dispatch_threadgroups

The merged kernel (Split-K + Reduce in one dispatch via shared memory)
is already implemented and working. The `FoveatedPrimitive` uses
`dispatch_threadgroups` directly — no runtime grid adjustment from
`dispatch_threads`.

Research confirmed that `dispatch_threads` vs `dispatch_threadgroups`
has minimal overhead difference (sub-microsecond). But using
`dispatch_threadgroups` matches the SDPA pattern exactly and avoids
any potential Metal runtime adjustments.

## Implementation

### Files to Modify

```
csrc/
  CMakeLists.txt              — add mlx_build_metallib, link metallib
  foveated_attn.h             — update FoveatedPrimitive with static bufs
  foveated_attn.cpp           — eval_gpu with precompiled lib + set_buffer
  kernels/
    foveated_attn.metal       — NEW: kernel source extracted from Python
    foveated_attn.h           — NEW: shared constants/structs for Metal
```

### FoveatedPrimitive::eval_gpu (Target Implementation)

```cpp
void FoveatedPrimitive::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& enc = d.get_command_encoder(s.index);

    // Allocate outputs (3 arrays: out, spike_flags, spike_tokens)
    for (auto& o : outputs)
        o.set_data(allocator::malloc(o.nbytes()));

    // Precompiled kernel — loaded once from .metallib, cached by name
    // Function constants specialize for HEAD_DIM, NUM_SPLITS, etc.
    auto kernel = d.get_kernel(kernel_name_, metallib_, hash_name_, func_consts_);
    enc.set_compute_pipeline_state(kernel);

    // All inputs — tracked via set_input_array for dependency tracking
    // inputs[0]: blob (packed statics), [1]: query, [2]: decode_k, [3]: decode_v
    int idx = 0;
    for (int i = 0; i < inputs.size(); i++)
        enc.set_input_array(inputs[i], idx++);

    // Outputs
    for (auto& o : outputs)
        enc.set_output_array(o, idx++);

    // Runtime params as raw bytes (no mx.array allocation)
    int n_decode = inputs[1].shape(2);
    int total_bh_q = inputs[0].shape(0);
    uint32_t params[2] = {(uint32_t)total_bh_q, (uint32_t)n_decode};
    enc.set_bytes(params, 2, idx);

    // Single dispatch — merged kernel, shared memory reduce
    enc.dispatch_threadgroups(
        MTL::Size(total_bh_q, 1, 1),
        MTL::Size(num_splits_ * 32, 1, 1));
}
```

This is structurally identical to `sdpa_vector` in MLX's own source.
Same `get_command_encoder`, same `set_compute_pipeline_state`, same
`dispatch_threadgroups`. No contiguity loops, no source caching, no
hash map mutex contention.

### Python Integration

The `FoveatedHandle` nanobind class creates `FoveatedPrimitive`
graph nodes via `array::make_arrays`. The interceptor calls it:

```python
# In _fused_sdpa_interceptor:
out, flags, tokens = handle(query, decode_k, decode_v)
```

One nanobind call per layer (3 dynamic inputs, no override arrays).
The primitive's `eval_gpu` runs on the same path as SDPA. The evaluator
processes `FoveatedPrimitive` nodes the same way it processes
`ScaledDotProductAttention` nodes — through
the generic `arr.primitive().eval_gpu()` virtual dispatch with no
special-casing either way.

### Build System

```cmake
# csrc/CMakeLists.txt

# Find MLX
find_package(MLX CONFIG REQUIRED)
include(${MLX_CMAKE_DIR}/extension.cmake)

# Compile Metal kernel to metallib
mlx_build_metallib(
    TARGET foveated_metallib
    TITLE foveated_attn
    OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/foveated_attn.metal
    INCLUDE_DIRS ${MLX_INCLUDE_DIRS}
)

# C++ extension module
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(foveated_ext
    NB_STATIC STABLE_ABI LTO NOMINSIZE
    NB_DOMAIN mlx
    bindings.cpp
    foveated_attn.cpp
)
target_link_libraries(foveated_ext PRIVATE mlx)
add_dependencies(foveated_ext foveated_metallib)

# Install metallib alongside the extension
install(FILES ${CMAKE_BINARY_DIR}/foveated_attn.metallib
        DESTINATION ${CMAKE_INSTALL_PREFIX})
```

Build: `uv sync --extra ext && uv run python setup.py build_ext --inplace`

The `.metallib` file is compiled once at build time. No JIT compilation
at runtime. No source strings stored in memory.

## What This Eliminates

| CustomKernel overhead | FoveatedPrimitive |
|-----------------------|-------------------|
| 10KB source string comparison per call | None (precompiled metallib) |
| Hash map lookup with shared_lock | Direct pointer to cached pipeline |
| Contiguity check loop (N inputs) | Blob packs 7 statics into 1; only 4 inputs total |
| Shape/stride/ndim conditional binding | Not used (no shape_infos) |
| dispatch_threads | dispatch_threadgroups |
| Lambda construction for builder | None (path-based library load) |
| checked_inputs vector allocation | No intermediate vector |

## Actual Outcome

The dispatch overhead optimizations, combined with fixing a dtype mismatch
(fp16 cache vs bf16 model) and replacing the SDPA monkey-patch with direct
attention module patching (`install_fused_attention`), eliminated the
end-to-end slowdown entirely:

**End-to-end decode performance (tok/s):**

| Model | Fused | Standard | Speedup |
|-------|-------|----------|---------|
| 4-bit (Qwen2.5-7B-Instruct-4bit) | 150 tok/s | 130-146 tok/s | 1.03-1.45x |
| bf16 (Qwen2.5-0.5B-Instruct-bf16) | 67-69 tok/s | 60-66 tok/s | 1.04-1.14x |

**Kernel microbenchmarks (7B shapes):**

| Context | fp16 SDPA | Fused | Speedup |
|---------|-----------|-------|---------|
| 1K | 0.84 ms | 1.00 ms | 0.84x |
| 4K | 2.07 ms | 1.20 ms | 1.72x |
| 8K | 4.15 ms | 1.68 ms | 2.47x |
| 16K | 9.67 ms | 2.90 ms | 3.34x |
| 32K | 15.19 ms | 5.18 ms | 2.93x |

Key changes that unlocked the speedup:
- Direct attention module patching replaced SDPA monkey-patch (eliminates interceptor overhead)
- Blob passed as tracked input array (`set_input_array`) instead of raw pointer (`set_buffer`)
- Lazy decode buffer concatenation (flat list + lazy concat vs O(n) chained concat)
- Closure-based interceptor for fallback path
- Conditional `astype` in C++ to skip no-op dtype casts
- Fixed dtype mismatch: fp16 cache vs bf16 model was causing silent conversion overhead

## Phases

| Phase | What | Status |
|-------|------|--------|
| 1 | Extract kernel to .metal file, compile with mlx_build_metallib | **Done** |
| 2 | FoveatedPrimitive eval_gpu: precompiled metallib, set_buffer for statics | **Done** |
| 3 | Metal function constants for tier specialization | **Done** |
| 4 | Benchmark: verify dispatch overhead reduction | **Done** |
| 5 | Remove old CustomKernel path, make FoveatedPrimitive the sole pipeline | **Done** |
| 6 | End-to-end validation on 0.5B + larger models | **Done** |

The old CustomKernel path (fast::metal_kernel) has been removed. The C++
extension now uses FoveatedPrimitive exclusively. The Python Metal kernel
path in `metal_foveated.py` remains as a fallback when the C++ extension
is not built.
