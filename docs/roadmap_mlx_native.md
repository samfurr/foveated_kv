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

### 2. Static Buffer Bypass (Eliminates Input Validation)

`CustomKernel::eval_gpu` iterates all inputs checking contiguity and
binding buffers through `set_input_array` (which does dependency
tracking against the command encoder's output set).

The fix: the 11 static tier arrays never change after compression.
Bind them via `set_buffer` using their raw `MTL::Buffer*` pointer,
extracted once at construction:

```cpp
// In FoveatedPrimitive constructor (once):
for (auto& arr : static_arrays) {
    static_bufs_.push_back(static_cast<const MTL::Buffer*>(arr.buffer().ptr()));
    static_offsets_.push_back(arr.offset());
}

// In eval_gpu (every call):
for (int i = 0; i < 11; i++)
    enc.set_buffer(static_bufs_[i], buffer_idx + i, static_offsets_[i]);
```

Critical detail from the research: `buffer().ptr()` returns the
`MTL::Buffer*`. Our earlier attempt used `buffer().raw_ptr()` which
returns the data address — wrong level of indirection. This is confirmed
in MLX's allocator source where the Metal buffer is obtained via
`static_cast<MTL::Buffer*>(buffer.ptr())`.

Only the 7 dynamic inputs (query, decode K/V, override arrays) go
through `set_input_array` for proper dependency tracking. The statics
have no dependencies — they were evaluated during compression and never
change.

Graph inputs drop from 19 (or 9 with blob) to 7. The evaluator only
tracks 7 edges per node instead of 19.

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

    // Static tier arrays — raw Metal buffer binding, no graph tracking
    for (int i = 0; i < n_static_; i++)
        enc.set_buffer(static_bufs_[i], i, static_offsets_[i]);

    // Dynamic inputs — standard set_input_array for dependency tracking
    // inputs[0]: query, [1]: decode_k, [2]: decode_v,
    // [3-6]: override arrays
    int idx = n_static_;
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

The `FoveatedHandleDirect` nanobind class creates `FoveatedPrimitive`
graph nodes via `array::make_arrays`. The interceptor calls it:

```python
# In _fused_sdpa_interceptor:
out, flags, tokens = handle(query, decode_k, decode_v,
                            ov_k, ov_v, ov_idx, ov_cnt)
```

One nanobind call per layer. The primitive's `eval_gpu` runs on the
same path as SDPA. The evaluator processes `FoveatedPrimitive` nodes
the same way it processes `ScaledDotProductAttention` nodes — through
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
| Contiguity check loop (N inputs) | Static bufs bypass graph; 7 dynamic only |
| Shape/stride/ndim conditional binding | Not used (no shape_infos) |
| dispatch_threads | dispatch_threadgroups |
| Lambda construction for builder | None (path-based library load) |
| checked_inputs vector allocation | No intermediate vector |

## Expected Outcome

The per-layer dispatch overhead should drop from ~5ms (CustomKernel) to
~1ms (matching SDPA). Over 24 layers, that is 96ms saved — bringing
end-to-end decode from ~130ms to ~35ms on the 0.5B model.

At that point, the fused kernel adds approximately 5ms of total overhead
over standard SDPA (from the 7 dynamic input bindings + runtime params).
The remaining comparison is pure GPU compute: our merged kernel reading
INT8/INT4 vs SDPA reading fp16. Break-even at short context, bandwidth
wins at long context.

## Phases

| Phase | What | Effort |
|-------|------|--------|
| 1 | Extract kernel to .metal file, compile with mlx_build_metallib | 0.5 day |
| 2 | Update FoveatedPrimitive eval_gpu: load metallib, set_buffer for statics | 1 day |
| 3 | Metal function constants for tier specialization | 0.5 day |
| 4 | Benchmark: verify SDPA-equivalent dispatch overhead | 0.5 day |
| 5 | End-to-end validation on 0.5B + larger models | 1 day |

Total: ~3 days of focused work. No MLX fork. Same `csrc/` directory,
same build system, same nanobind module.
