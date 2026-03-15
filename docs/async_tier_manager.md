# Async Promotion System

Last updated: 2026-03-14

## Architecture

The MLX async promotion system (`mlx_async_promoter.py`) uses 2 background worker
threads to handle tier management without blocking the main decode loop.

### Why 2 workers

1. **Spike worker**: Processes raw spike events from the Metal kernel. Maps far-local
   indices to archive indices. Queues promotion updates. Handles the urgent path.

2. **Disk worker**: Reads fp16 originals from NVMe mmap archives. Handles the I/O
   path separately so spike processing is never blocked by disk reads.

### Key constraint: no MLX in workers

MLX's computation graph is not thread-safe. All worker threads use **numpy only**.
MLX operations (updating cache tensors, converting numpy arrays to mx.array) happen
on the main thread when updates are drained and applied.

This is a hard constraint, not a performance choice.

## Unified Memory Advantage

On Apple Silicon, there is no CPU-GPU memory transfer bottleneck:

- No PCIe transfers to schedule or overlap
- Worker threads access the same physical memory as the GPU
- Disk mmap archives read directly into numpy arrays
- Updated tier data is visible to the GPU after mx.array conversion on main thread

This eliminates the entire "layer-spread PCIe transfer" design from the earlier
architecture. Promotion is: read from disk -> convert on main thread -> done.

## Disk mmap Archives (`disk_archive.py`)

NVMe-backed numpy.memmap files store exact fp16 originals for all non-foveal tokens.

- **One file per layer.** Separate mmap files avoid lock contention between layers.
- **~50us per token read.** NVMe random read latency, not sequential throughput.
- **Written once after prefill.** New tokens during decode are appended.
- **numpy.memmap**: OS handles page caching. Recently accessed regions stay hot.

Archive format:
```
archive_layer_0.npy  ->  memmap shape (S, 2, H_kv, D) dtype=float16
archive_layer_1.npy      [token_idx, 0=key/1=value, head, dim]
...
```

## GPU -> Main Thread Handoff

### Fire-and-forget spike handoff

The Metal kernel writes spike flags and token indices during attention. The main
thread collects these after each layer's attention call and hands them to the spike
worker via a simple queue:

```python
# Main thread (after Metal kernel returns)
spikes = collect_spikes(layer_idx)
if spikes:
    promoter.submit_spikes(layer_idx, spikes)  # non-blocking

# Later in the same decode step
updates = promoter.drain(layer_idx)  # O(1) dict lookup
if updates:
    apply_updates(cache, layer_idx, updates)  # MLX operations here
```

### O(1) drain via dict

Ready updates are stored in a dict keyed by layer index. The main thread checks
for its current layer and applies any pending updates. No scanning, no priority
queue traversal.

## Worker Model

### Spike worker

```
Loop:
  1. Block on spike queue (with timeout)
  2. Map far-local index -> archive index
  3. Submit read request to disk worker
  4. Store pending promotion in ready dict
```

### Disk worker

```
Loop:
  1. Block on read queue (with timeout)
  2. Read fp16 K,V from mmap archive
  3. Store result in ready dict (keyed by layer)
```

Both workers shut down cleanly when signaled. No daemon threads.

## Safe Mutation Points

Updates are applied only on the main thread, only between layer calls:

- After draining updates for a layer
- Before the next layer's attention call
- Never during a Metal kernel execution

This is simpler than the earlier design because unified memory eliminates the
need to overlap PCIe transfers with MLP compute.

## Stats and Instrumentation

The promoter tracks:
- Spikes submitted
- Spikes processed
- Disk reads completed
- Updates applied
- Queue depths (current and peak)

These are available via `promoter.stats()` for debugging and benchmarking.

## Comparison with Earlier Design

| Aspect | Earlier (PyTorch/CUDA) | Current (MLX) |
|--------|----------------------|---------------|
| Workers | 1 urgent + 1 background | 1 spike + 1 disk |
| Archive | CPU RAM tensors | NVMe mmap files |
| Transfer | PCIe GPU<->CPU | None (unified memory) |
| Worker ops | numpy scoring | numpy only (no MLX) |
| Update apply | GPU-side at safe points | Main thread mx.array conversion |
| Handoff | publish_query + publish_spike | fire-and-forget submit_spikes |
| Drain | drain_ready_updates(layer) | O(1) dict[layer] lookup |

The MLX version is simpler because unified memory removes the transfer scheduling
problem entirely. The hard constraint is thread safety of MLX's computation graph,
which is solved by keeping all MLX operations on the main thread.

## Implementation

- `src/mipmap_kv/mlx_async_promoter.py` — async promoter with 2 workers
- `src/mipmap_kv/disk_archive.py` — numpy.memmap archive management
- `tests/test_disk_archive.py` — 8 tests for archive operations
- `tests/test_mlx_foveated.py` — MLX cache tests including promotion paths
