---
name: CUDA kernel has unfixed bug
description: The CUDA Split-K kernel in csrc/ has a bug and hasn't been able to run. User considering removing CUDA and Triton code entirely.
type: project
---

The CUDA foveated decode kernel (`csrc/foveated_attn/foveated_decode.cu`) has an unresolved bug preventing it from running. The Triton kernel (`triton_foveated.py`) was also ~10x slower than FA2 on A100.

**Why:** MLX Metal Split-K kernel now works and delivers 2.25-3.28x speedup on Apple Silicon. CUDA/Triton code may no longer be needed if the project targets Apple Silicon.

**How to apply:** Don't try to fix/use the CUDA kernel. The working Metal Split-K kernel in `metal_foveated.py` is the production path. If user decides to remove CUDA/Triton, clean up `csrc/`, `cuda_foveated.py`, `triton_foveated.py`, and related references.
