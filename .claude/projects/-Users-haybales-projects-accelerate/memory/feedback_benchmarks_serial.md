---
name: Run benchmarks serially on 8GB Mac
description: Don't run multiple benchmarks in parallel — they compete for memory on 8GB machine
type: feedback
---

Run benchmarks one at a time, not in parallel. User's machine has 8GB RAM. Parallel model benchmarks cause OOM or memory pressure.

**Why:** User pointed out benchmarks take 12+ minutes and should not overlap.

**How to apply:** Always run model benchmarks sequentially. Use `run_in_background` only for non-memory-intensive tasks (code agents, file edits). Max context for Qwen2.5-0.5B bf16 on 8GB is ~8K (16K OOMs on Metal's 4GB buffer limit from vocab logits).
