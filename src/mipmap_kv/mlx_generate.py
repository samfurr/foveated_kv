"""
mlx-lm integration for foveated KV cache quality benchmarks.

Provides cache wrappers that plug into mlx-lm's generation pipeline,
enabling real model evaluation with foveated mixed-precision attention.

Strategy:
  1. Prefill with standard mlx-lm cache → extract K,V
  2. Compress into MLXFoveatedKVCache
  3. Replace cache objects with FoveatedCacheWrapper
  4. Decode normally — model sees dequanted K,V from foveated tiers
  5. Quality degradation from quantization is what we measure

For speed benchmarks, use benchmark_mlx.py (synthetic, measures fused kernel).
For quality benchmarks, use this module (real model, measures end-to-end accuracy).

FusedCacheWrapper + SDPA monkey-patch (speed path):
  Instead of materializing fp16 K,V intermediates for the model's own SDPA,
  we intercept mx.fast.scaled_dot_product_attention and route decode attention
  through the fused Split-K Metal kernel that operates directly on quantized
  data — 3.28x faster than fp16 SDPA at 32K context.
"""

import math
from typing import Optional

import mlx.core as mx

from .mlx_foveated import MLXFoveatedKVCache, MLXFoveatedLayer, MLXTierConfig
from .mlx_quantize import (
    dequantize_int4_per_token,
    dequantize_int8_per_channel,
    dequantize_int8_per_token,
    quantize_int8_per_channel,
    quantize_int8_per_token,
    quantize_int4_per_token,
)


def _log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Numerically stable log-softmax."""
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


class FoveatedCacheWrapper:
    """Drop-in replacement for mlx-lm's KVCache that uses foveated storage.

    On update_and_fetch(), adds the new token to the foveal tier and returns
    all tiers dequanted to fp16. The model's SDPA runs on the dequanted K,V,
    so the attention output reflects quantization error — which is exactly
    what quality benchmarks need to measure.
    """

    def __init__(self, fov_layer: MLXFoveatedLayer):
        self.fov_layer = fov_layer
        self.offset = fov_layer.total_tokens

    @property
    def state(self):
        """mlx-lm checks this to determine if cache has data."""
        return (self.fov_layer.foveal_k, self.fov_layer.foveal_v)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Add new token and return all K,V for SDPA.

        Args:
            keys: (B, H_kv, 1, D) — new decode token's K (post-RoPE)
            values: (B, H_kv, 1, D) — new decode token's V

        Returns:
            all_keys: (B, H_kv, S_total, D) float16
            all_values: (B, H_kv, S_total, D) float16
        """
        self.fov_layer.add_token(keys, values)
        self.offset += 1
        return self._get_all_kv()

    def _get_all_kv(self) -> tuple[mx.array, mx.array]:
        """Dequant all tiers and concatenate for standard SDPA."""
        layer = self.fov_layer

        periph_k = dequantize_int8_per_channel(
            layer.periph_k, layer.periph_k_scale, layer.periph_k_zero
        )
        periph_v = dequantize_int8_per_token(
            layer.periph_v, layer.periph_v_scale, layer.periph_v_zero
        )
        far_k = dequantize_int8_per_channel(
            layer.far_k, layer.far_k_scale, layer.far_k_zero
        )
        far_v = dequantize_int4_per_token(
            layer.far_v, layer.far_v_scale, layer.far_v_zero
        )

        all_k = mx.concatenate([layer.effective_foveal_k, periph_k, far_k], axis=2)
        all_v = mx.concatenate([layer.effective_foveal_v, periph_v, far_v], axis=2)
        return all_k, all_v


class AsyncCacheWrapper:
    """Cache wrapper with non-blocking async promotion.

    Flow per decode step:
      1. Apply ready promotions from background worker (non-blocking drain)
      2. Add new token to foveal tier
      3. Detect far-tier spikes (fast — score comparison in main thread)
      4. Queue spikes to async promoter (background reads from disk mmap)
      5. Return dequanted K,V for model's SDPA

    Only far-tier (INT4 V) tokens are promotion candidates.
    Peripheral (INT8) has low enough error — not worth promoting.
    Margin = 0: if any far token outscores weakest foveal, promote.
    """

    def __init__(self, fov_layer: MLXFoveatedLayer, layer_idx: int, promoter):
        self.fov_layer = fov_layer
        self.layer_idx = layer_idx
        self.promoter = promoter  # AsyncPromoter
        self.offset = fov_layer.total_tokens

    @property
    def state(self):
        return (self.fov_layer.foveal_k, self.fov_layer.foveal_v)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Add token, async spike detection, return K,V."""
        # 1. Apply ready promotions from previous step (non-blocking)
        self._apply_ready_promotions()

        # 2. Add new token to foveal
        self.fov_layer.add_token(keys, values)
        self.offset += 1

        # 3. Detect far-tier spikes and queue to async worker
        self.promoter.detect_and_queue(
            self.layer_idx, keys, self.fov_layer
        )

        # 4. Return dequanted K,V for model's SDPA
        return self._get_all_kv()

    def _apply_ready_promotions(self):
        """Grow-only promotion: add promoted token to foveal, never evict.

        Quality can only improve — we're adding exact fp16 for a token
        the model is paying attention to. No risk of evicting important tokens.
        Cap at 1 per layer per step to bound overhead.
        """
        ready = self.promoter.drain_ready(self.layer_idx)
        if not ready:
            return
        promo = ready[-1]
        layer = self.fov_layer
        H = layer.foveal_k.shape[1]
        D = layer.foveal_k.shape[-1]
        n_fov = layer.foveal_k.shape[2]
        h = promo.head_idx
        if n_fov == 0:
            return
        # Swap: replace weakest foveal in THIS head only (mx.where, head-specific)
        pk = mx.array(promo.promoted_k_np).reshape(1, 1, 1, D).astype(mx.float16)
        pv = mx.array(promo.promoted_v_np).reshape(1, 1, 1, D).astype(mx.float16)
        norms = mx.sum(layer.foveal_k[promo.batch_idx, h].astype(mx.float32) ** 2, axis=-1)
        weakest = mx.argmin(norms)
        slot_mask = (mx.arange(n_fov) == weakest).reshape(1, 1, n_fov, 1)
        h_mask = (mx.arange(H).reshape(1, H, 1, 1) == h)
        mask = slot_mask & h_mask
        layer.foveal_k = mx.where(mask, pk, layer.foveal_k)
        layer.foveal_v = mx.where(mask, pv, layer.foveal_v)
        self.promoter.stats.promotions_applied += 1

    def _get_all_kv(self) -> tuple[mx.array, mx.array]:
        layer = self.fov_layer
        periph_k = dequantize_int8_per_channel(
            layer.periph_k, layer.periph_k_scale, layer.periph_k_zero)
        periph_v = dequantize_int8_per_token(
            layer.periph_v, layer.periph_v_scale, layer.periph_v_zero)
        far_k = dequantize_int8_per_channel(
            layer.far_k, layer.far_k_scale, layer.far_k_zero)
        far_v = dequantize_int4_per_token(
            layer.far_v, layer.far_v_scale, layer.far_v_zero)
        all_k = mx.concatenate([layer.effective_foveal_k, periph_k, far_k], axis=2)
        all_v = mx.concatenate([layer.effective_foveal_v, periph_v, far_v], axis=2)
        return all_k, all_v


# ---------------------------------------------------------------------------
# Fused SDPA monkey-patch: route decode attention through the Metal kernel
# ---------------------------------------------------------------------------

# Global state for the SDPA interceptor. Using a simple namespace rather than
# threading.local() because MLX decode is single-threaded (the async promoter
# worker never calls SDPA). A namespace keeps things readable.

class _FusedSDPAState:
    """Global mutable state for the fused SDPA interceptor."""
    original_sdpa = None
    installed: bool = False
    fov_cache: Optional[MLXFoveatedKVCache] = None
    promoter = None
    layer_counter: int = 0
    n_layers: int = 0
    decode_step: int = 0
    spike_check_interval: int = 4  # Check spikes every N steps (not every step)
    pending_spikes: list = []
    _far_idx_np_cache: dict = {}

_fused_state = _FusedSDPAState()


class FusedCacheWrapper:
    """Cache wrapper for the fused SDPA path.

    Unlike FoveatedCacheWrapper which returns dequanted fp16 K,V for the
    model's own SDPA, this wrapper returns *dummy* K,V tensors because the
    actual attention is handled by the intercepted SDPA function calling
    directly into the fused Metal kernel on quantized data.

    update_and_fetch() still captures new K,V and adds them to the foveal
    tier so the cache stays up-to-date.
    """

    def __init__(self, fov_layer: MLXFoveatedLayer, layer_idx: int):
        self.fov_layer = fov_layer
        self.layer_idx = layer_idx
        self.offset = fov_layer.total_tokens

    @property
    def state(self):
        """mlx-lm checks this to determine if cache has data."""
        return (self.fov_layer.foveal_k, self.fov_layer.foveal_v)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Add new token to foveal tier; return dummy K,V for shape compliance.

        The model will pass these K,V into mx.fast.scaled_dot_product_attention,
        but our interceptor ignores them and uses the fused kernel instead.
        We still need correct shapes so the model's reshaping/assertions pass.

        Args:
            keys: (B, H_kv, 1, D) new decode token's K
            values: (B, H_kv, 1, D) new decode token's V

        Returns:
            dummy_keys: (B, H_kv, 1, D) — single-token dummy (minimal memory)
            dummy_values: (B, H_kv, 1, D)
        """
        # Just capture new token. Promotions are applied in reset_fused_layer_counter()
        # at the start of each step — not here in the hot path.
        self.fov_layer.add_token(keys, values)
        self.offset += 1
        return keys, values


def _fused_sdpa_interceptor(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask=None,
    **kwargs,
) -> mx.array:
    """Drop-in replacement for mx.fast.scaled_dot_product_attention.

    For prefill (seq_len > 1): passes through to the original SDPA.
    For decode (seq_len == 1): routes through the fused Metal kernel
    operating directly on quantized KV data.
    """
    seq_len = queries.shape[2]

    # Prefill: pass through to original SDPA
    if seq_len > 1:
        return _fused_state.original_sdpa(
            queries, keys, values, scale=scale, mask=mask, **kwargs
        )

    # Decode: use fused Metal kernel
    state = _fused_state
    layer_idx = state.layer_counter
    state.layer_counter += 1

    # Wrap around if we exceed total layers (safety)
    if layer_idx >= state.n_layers:
        layer_idx = layer_idx % state.n_layers

    fov_layer = state.fov_cache.layers[layer_idx]
    if fov_layer is None:
        # Layer not in foveated cache — fall back to original
        return state.original_sdpa(
            queries, keys, values, scale=scale, mask=mask, **kwargs
        )

    # Route through fused kernel with spike detection on ALL layers.
    # Spike flags are a free kernel side-output (zero extra compute).
    # We store references here (zero overhead). The data lands in unified
    # memory when the kernel completes. At the start of the NEXT step,
    # reset_fused_layer_counter() reads them — by then mx.eval(logits) has
    # guaranteed all kernels finished. Pure fire-and-forget.
    if state.promoter is not None:
        output, spike_flags, spike_tokens = fov_layer.attend_fused_with_spikes(queries)
        if spike_flags is not None:
            state.pending_spikes.append((layer_idx, spike_flags, spike_tokens))
        return output
    else:
        return fov_layer.attend_fused(queries)


def install_fused_sdpa(
    fov_cache: MLXFoveatedKVCache,
    promoter=None,
) -> None:
    """Monkey-patch mx.fast.scaled_dot_product_attention for fused decode.

    Installs an interceptor that routes decode-time attention (seq_len=1)
    through the fused Split-K Metal kernel, bypassing fp16 intermediate
    materialization for a ~3.3x speedup.

    Args:
        fov_cache: The foveated cache with all layers populated.
        promoter: Optional AsyncPromoter for spike-driven promotion.
    """
    if _fused_state.installed:
        raise RuntimeError("Fused SDPA already installed — call uninstall_fused_sdpa() first")

    _fused_state.original_sdpa = mx.fast.scaled_dot_product_attention
    _fused_state.fov_cache = fov_cache
    _fused_state.promoter = promoter
    _fused_state.layer_counter = 0
    _fused_state.n_layers = len(fov_cache.layers)
    _fused_state.pending_spikes = []
    _fused_state._far_idx_np_cache = {}
    _fused_state.decode_step = 0
    _fused_state.installed = True

    mx.fast.scaled_dot_product_attention = _fused_sdpa_interceptor


def uninstall_fused_sdpa() -> None:
    """Restore the original mx.fast.scaled_dot_product_attention."""
    if not _fused_state.installed:
        return

    mx.fast.scaled_dot_product_attention = _fused_state.original_sdpa
    _fused_state.original_sdpa = None
    _fused_state.fov_cache = None
    _fused_state.promoter = None
    _fused_state.layer_counter = 0
    _fused_state.n_layers = 0
    _fused_state.installed = False


def reset_fused_layer_counter() -> None:
    """Reset layer counter + flush spikes + apply promotions. Call per step.

    This is THE sync point — everything happens here, once per decode step:
    1. Batch-eval all pending spike arrays (ONE GPU sync for all layers)
    2. Hand resolved spikes to async worker (non-blocking)
    3. Drain and apply any ready promotions from previous steps
    4. Reset counter for the new forward pass
    """
    state = _fused_state
    state.layer_counter = 0
    state.decode_step += 1

    if state.promoter is None:
        state.pending_spikes.clear()
        return

    # --- Flush pending spikes (batch eval, one GPU sync) ---
    if state.pending_spikes:
        _flush_pending_spikes()

    # --- Apply ready promotions in batch (one rebuild per layer) ---
    _apply_promotions_batched(state.fov_cache, state.promoter)


def _flush_pending_spikes():
    """Batch-eval all layers' spike arrays and hand to async worker.

    Called once per decode step. Converts lazy MLX arrays to numpy in one
    batch, then enqueues to the raw spike worker for dedup + disk reads.
    """
    import numpy as np
    state = _fused_state
    pending = state.pending_spikes
    state.pending_spikes = []

    if not pending:
        return

    # One GPU sync for all layers' spike arrays. The kernels already ran
    # (logits eval synced the GPU), so this just ensures MLX buffers stay valid.
    all_arrays = []
    for _, flags, tokens in pending:
        all_arrays.extend([flags, tokens])
    mx.eval(*all_arrays)

    for layer_idx, flags, tokens in pending:
        flags_np = np.array(flags)
        tokens_np = np.array(tokens)

        # Cache far_idx numpy per layer (stable between promotions)
        if layer_idx not in state._far_idx_np_cache:
            fov_layer = state.fov_cache.layers[layer_idx]
            if fov_layer is not None:
                mx.eval(fov_layer.far_idx)
                state._far_idx_np_cache[layer_idx] = np.array(fov_layer.far_idx)

        far_idx_np = state._far_idx_np_cache.get(layer_idx)
        if far_idx_np is None:
            continue

        try:
            state.promoter._raw_spike_queue.put_nowait(
                (layer_idx, flags_np, tokens_np, far_idx_np)
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# End fused SDPA monkey-patch
# ---------------------------------------------------------------------------


def _apply_promotions_batched(cache: MLXFoveatedKVCache, promoter):
    """Grow-only promotion: add promoted tokens to foveal, never evict.

    Quality can only improve — we're adding exact fp16 for tokens the
    model is paying attention to. No eviction means no risk of removing
    important tokens. Max 1 promotion per step total to bound overhead.
    """
    # Collect all ready promotions across all layers
    all_ready = []
    for layer_idx in range(len(cache.layers)):
        if cache.layers[layer_idx] is None:
            continue
        ready = promoter.drain_ready(layer_idx)
        for p in ready:
            all_ready.append((layer_idx, p))

    if not all_ready:
        return

    # Hard cap: max 1 promotion per 10 steps (avoid cumulative swap noise)
    if hasattr(promoter, '_last_promo_step') and \
       (promoter.stats.promotions_applied > 0) and \
       (promoter._apply_counter - promoter._last_promo_step < 10):
        return
    if not hasattr(promoter, '_apply_counter'):
        promoter._apply_counter = 0
        promoter._last_promo_step = -10
    promoter._apply_counter += 1

    # Apply at most 1 (most recent = most relevant)
    layer_idx, promo = all_ready[-1]
    layer = cache.layers[layer_idx]
    H = layer.foveal_k.shape[1]
    D = layer.foveal_k.shape[-1]
    n_fov = layer.foveal_k.shape[2]
    h = promo.head_idx

    if n_fov > 0:
        pk = mx.array(promo.promoted_k_np).reshape(1, 1, 1, D).astype(mx.float16)
        pv = mx.array(promo.promoted_v_np).reshape(1, 1, 1, D).astype(mx.float16)
        norms = mx.sum(layer.foveal_k[promo.batch_idx, h].astype(mx.float32) ** 2, axis=-1)
        weakest = mx.argmin(norms)
        slot_mask = (mx.arange(n_fov) == weakest).reshape(1, 1, n_fov, 1)
        h_mask = (mx.arange(H).reshape(1, H, 1, 1) == h)
        mask = slot_mask & h_mask
        layer.foveal_k = mx.where(mask, pk, layer.foveal_k)
        layer.foveal_v = mx.where(mask, pv, layer.foveal_v)

    promoter._last_promo_step = promoter._apply_counter
    promoter.stats.promotions_applied += 1


def generate_with_promotion(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    cfg: Optional[MLXTierConfig] = None,
    disk_archive_dir: Optional[str] = None,
    temp: float = 0.0,
) -> tuple[str, dict]:
    """Generate with async promotion: spike detection → background disk read → tier update.

    The promotion pipeline is fully non-blocking:
    - Main thread: decode step + fast spike detection (score comparison)
    - Background thread: reads fp16 from disk mmap, prepares promotions
    - Promotions applied at start of next decode step (safe mutation point)

    Only far-tier tokens (INT4 V) are promoted. Margin = 0.
    """
    from .disk_archive import offload_cache_to_disk
    from .mlx_async_promoter import AsyncPromoter

    tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    fov_cache, prefill_logits, _ = prefill_and_compress(model, tokens, cfg)

    mem_before = fov_cache.memory_bytes()

    # Offload archives to disk
    if disk_archive_dir is None:
        import tempfile
        disk_archive_dir = tempfile.mkdtemp(prefix="foveated_archive_")
    disk_archives = offload_cache_to_disk(fov_cache, disk_archive_dir)
    mem_after = fov_cache.memory_bytes()

    # Start async promoter
    promoter = AsyncPromoter(fov_cache, disk_archives)

    # Wrap with async cache wrappers
    wrappers = []
    for i, layer in enumerate(fov_cache.layers):
        if layer is None:
            wrappers.append(None)
            continue
        wrappers.append(AsyncCacheWrapper(layer, i, promoter))

    # Decode
    generated = []
    next_logits = prefill_logits[:, -1, :]

    for _ in range(max_tokens):
        if temp == 0:
            next_token = mx.argmax(next_logits, axis=-1)
        else:
            next_token = mx.random.categorical(next_logits / temp)

        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)

        next_input = next_token.reshape(1, 1)
        next_logits = model(next_input, cache=wrappers)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits)

    promoter.stop()
    stats = promoter.get_stats()
    stats.update({
        "prompt_tokens": tokens.shape[1],
        "generated_tokens": len(generated),
        "disk_offloaded": True,
        "mem_quantized_mb": mem_before["total_quantized"] / (1024 * 1024),
        "mem_archive_before_mb": mem_before["archive"] / (1024 * 1024),
        "mem_archive_after_mb": mem_after["archive"] / (1024 * 1024),
        "mem_saved_mb": (mem_before["archive"] - mem_after["archive"]) / (1024 * 1024),
    })

    return tokenizer.decode(generated), stats


def generate_fused(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    cfg: Optional[MLXTierConfig] = None,
    disk_archive_dir: Optional[str] = None,
    temp: float = 0.0,
    enable_promotion: bool = True,
) -> tuple[str, dict]:
    """Generate text using fused Metal kernel for maximum decode speed.

    Instead of materializing fp16 K,V intermediates and running the model's
    own SDPA, this intercepts mx.fast.scaled_dot_product_attention and routes
    decode attention through the fused Split-K Metal kernel that operates
    directly on quantized data — ~3.3x faster than fp16 SDPA at 32K context.

    Pipeline:
      1. Prefill with standard mlx-lm cache
      2. Build foveated cache + offload archives to disk
      3. Install fused SDPA interceptor (monkey-patches mx.fast)
      4. Optionally start async promoter for spike-driven tier updates
      5. Decode loop — model calls flow through fused kernel automatically
      6. Uninstall interceptor + cleanup

    Args:
        model: mlx-lm model
        tokenizer: mlx-lm tokenizer
        prompt: input text
        max_tokens: max tokens to generate
        cfg: tier configuration
        disk_archive_dir: directory for disk-backed archives (temp dir if None)
        temp: sampling temperature (0 = greedy)
        enable_promotion: if True, run async spike detection + promotion

    Returns:
        generated_text: decoded output
        stats: dict with timing and promotion info
    """
    import time

    from .disk_archive import offload_cache_to_disk
    from .mlx_async_promoter import AsyncPromoter

    tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)

    # 1. Prefill with standard cache, then compress into foveated tiers
    t0 = time.perf_counter()
    fov_cache, prefill_logits, _ = prefill_and_compress(model, tokens, cfg)
    t_prefill = time.perf_counter() - t0

    mem_before = fov_cache.memory_bytes()

    # 2. Offload archives to disk
    if disk_archive_dir is None:
        import tempfile
        disk_archive_dir = tempfile.mkdtemp(prefix="foveated_fused_")
    disk_archives = offload_cache_to_disk(fov_cache, disk_archive_dir)
    mem_after = fov_cache.memory_bytes()

    # 3. Start async promoter (optional)
    promoter = None
    if enable_promotion:
        promoter = AsyncPromoter(fov_cache, disk_archives)

    # 4. Wrap with FusedCacheWrapper (returns dummy K,V; SDPA is intercepted)
    fused_wrappers = []
    for i, layer in enumerate(fov_cache.layers):
        if layer is None:
            fused_wrappers.append(None)
            continue
        fused_wrappers.append(FusedCacheWrapper(layer, i))

    # 5. Install fused SDPA interceptor
    install_fused_sdpa(fov_cache, promoter=promoter)

    # 6. Decode loop
    generated = []
    next_logits = prefill_logits[:, -1, :]
    t_decode_start = time.perf_counter()

    try:
        for step in range(max_tokens):
            if temp == 0:
                next_token = mx.argmax(next_logits, axis=-1)
            else:
                next_token = mx.random.categorical(next_logits / temp)

            token_id = next_token.item()
            if token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)

            # Reset layer counter at start of each forward pass
            reset_fused_layer_counter()

            next_input = next_token.reshape(1, 1)
            next_logits = model(next_input, cache=fused_wrappers)
            next_logits = next_logits[:, -1, :]
            mx.eval(next_logits)

    finally:
        # 7. Always uninstall, even on error
        uninstall_fused_sdpa()
        if promoter is not None:
            promoter.stop()

    t_decode = time.perf_counter() - t_decode_start

    # Collect stats
    stats = {
        "prompt_tokens": tokens.shape[1],
        "generated_tokens": len(generated),
        "fused_kernel": True,
        "disk_offloaded": True,
        "prefill_time_s": t_prefill,
        "decode_time_s": t_decode,
        "tokens_per_second": len(generated) / max(t_decode, 1e-6),
        "mem_quantized_mb": mem_before["total_quantized"] / (1024 * 1024),
        "mem_archive_before_mb": mem_before["archive"] / (1024 * 1024),
        "mem_archive_after_mb": mem_after["archive"] / (1024 * 1024),
        "mem_saved_mb": (mem_before["archive"] - mem_after["archive"]) / (1024 * 1024),
    }
    if promoter is not None:
        stats.update(promoter.get_stats())

    return tokenizer.decode(generated), stats


def prefill_and_compress(
    model,
    tokens: mx.array,
    cfg: Optional[MLXTierConfig] = None,
) -> tuple[MLXFoveatedKVCache, mx.array, list]:
    """Prefill a model and compress its KV cache into foveated tiers.

    Args:
        model: mlx-lm model (nn.Module with model.layers)
        tokens: (1, S) input token IDs
        cfg: tier configuration

    Returns:
        fov_cache: compressed MLXFoveatedKVCache
        prefill_logits: (1, S, vocab) logits from prefill
        std_caches: the original standard caches (for reference comparison)
    """
    from mlx_lm.models.cache import make_prompt_cache

    cfg = cfg or MLXTierConfig()

    # Prefill with standard cache
    std_caches = make_prompt_cache(model)
    prefill_logits = model(tokens, cache=std_caches)
    mx.eval(prefill_logits)

    # Extract K,V from standard caches and build foveated cache
    fov_cache = MLXFoveatedKVCache(cfg)
    for layer_idx, cache_entry in enumerate(std_caches):
        k, v = cache_entry.state
        if k is not None:
            fov_cache.update(k, v, layer_idx)

    fov_cache.compress()
    return fov_cache, prefill_logits, std_caches


def wrap_cache_for_model(
    fov_cache: MLXFoveatedKVCache,
) -> list[FoveatedCacheWrapper]:
    """Create mlx-lm-compatible cache wrappers for each layer."""
    return [
        FoveatedCacheWrapper(layer) if layer is not None else None
        for layer in fov_cache.layers
    ]


def generate_foveated(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    cfg: Optional[MLXTierConfig] = None,
    temp: float = 0.0,
) -> tuple[str, dict]:
    """Generate text using foveated KV cache with a real mlx-lm model.

    Args:
        model: mlx-lm model
        tokenizer: mlx-lm tokenizer
        prompt: input text
        max_tokens: max tokens to generate
        cfg: tier configuration
        temp: sampling temperature (0 = greedy)

    Returns:
        generated_text: the full output (prompt + generated)
        stats: dict with compression info
    """
    tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)

    # Prefill + compress
    fov_cache, prefill_logits, _ = prefill_and_compress(model, tokens, cfg)
    stats = {
        "prompt_tokens": tokens.shape[1],
        "tiers": {
            "foveal": fov_cache.layers[0].foveal_k.shape[2],
            "peripheral": fov_cache.layers[0].periph_k.shape[2],
            "far": fov_cache.layers[0].far_k.shape[2],
        },
    }

    # Wrap for model compatibility
    fov_wrappers = wrap_cache_for_model(fov_cache)

    # Greedy decode
    generated = []
    next_logits = prefill_logits[:, -1, :]

    for _ in range(max_tokens):
        if temp == 0:
            next_token = mx.argmax(next_logits, axis=-1)
        else:
            next_token = mx.random.categorical(next_logits / temp)

        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)

        # Decode step
        next_input = next_token.reshape(1, 1)
        next_logits = model(next_input, cache=fov_wrappers)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits)

    stats["generated_tokens"] = len(generated)
    output_text = tokenizer.decode(generated)
    return output_text, stats


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    context_len: int = 4096,
    eval_len: int = 256,
    cfg: Optional[MLXTierConfig] = None,
) -> tuple[float, float]:
    """Compute perplexity with foveated cache vs standard cache.

    Prefills context_len tokens, then evaluates log-likelihood on
    the next eval_len tokens with both standard and foveated caches.

    Args:
        model: mlx-lm model
        tokenizer: tokenizer
        text: evaluation text (should be longer than context_len + eval_len)
        context_len: number of tokens to prefill as context
        eval_len: number of tokens to evaluate PPL over
        cfg: tier config

    Returns:
        standard_ppl: perplexity with full fp16 cache
        foveated_ppl: perplexity with foveated cache
    """
    from mlx_lm.models.cache import make_prompt_cache

    cfg = cfg or MLXTierConfig()

    all_tokens = mx.array(tokenizer.encode(text))
    total = context_len + eval_len
    if all_tokens.shape[0] < total:
        raise ValueError(
            f"Text too short: {all_tokens.shape[0]} tokens, need {total}"
        )

    context = all_tokens[:context_len].reshape(1, -1)
    eval_tokens = all_tokens[context_len:total]

    # --- Standard PPL ---
    std_cache = make_prompt_cache(model)
    logits = model(context, cache=std_cache)
    mx.eval(logits)

    std_nll = 0.0
    for i in range(eval_len):
        tok = eval_tokens[i : i + 1].reshape(1, 1)
        logits = model(tok, cache=std_cache)
        log_probs = _log_softmax(logits[:, -1, :])
        mx.eval(log_probs)
        if i + 1 < eval_len:
            target = eval_tokens[i + 1].item()
            std_nll -= log_probs[0, target].item()

    std_ppl = math.exp(std_nll / max(eval_len - 1, 1))

    # --- Foveated PPL ---
    fov_cache, _, _ = prefill_and_compress(model, context, cfg)
    fov_wrappers = wrap_cache_for_model(fov_cache)

    fov_nll = 0.0
    for i in range(eval_len):
        tok = eval_tokens[i : i + 1].reshape(1, 1)
        logits = model(tok, cache=fov_wrappers)
        log_probs = _log_softmax(logits[:, -1, :])
        mx.eval(log_probs)
        if i + 1 < eval_len:
            target = eval_tokens[i + 1].item()
            fov_nll -= log_probs[0, target].item()

    fov_ppl = math.exp(fov_nll / max(eval_len - 1, 1))

    return std_ppl, fov_ppl


def needle_test(
    model,
    tokenizer,
    context_len: int = 4096,
    needle_depth: float = 0.5,
    cfg: Optional[MLXTierConfig] = None,
) -> tuple[bool, bool, dict]:
    """Needle-in-a-haystack test: can the model retrieve a passkey?

    Inserts a random 5-digit passkey at needle_depth within filler text,
    then asks the model to retrieve it. Tests both standard and foveated.

    Args:
        model: mlx-lm model
        tokenizer: tokenizer
        context_len: total context length
        needle_depth: 0.0 = beginning, 1.0 = end
        cfg: tier config

    Returns:
        standard_found: did standard cache find the passkey?
        foveated_found: did foveated cache find the passkey?
        info: dict with passkey, generated text, etc.
    """
    import random

    cfg = cfg or MLXTierConfig()
    passkey = str(random.randint(10000, 99999))

    # Build prompt with needle
    needle = f"The secret passkey is {passkey}. Remember it."
    filler_sentence = "This is a document about various topics in science and technology. "
    retrieval_prompt = "\nWhat is the secret passkey mentioned in the text above? The passkey is: "

    # Estimate tokens per filler repeat
    filler_tokens = len(tokenizer.encode(filler_sentence))
    needle_tokens = len(tokenizer.encode(needle))
    retrieval_tokens = len(tokenizer.encode(retrieval_prompt))
    available = context_len - needle_tokens - retrieval_tokens - 10

    n_filler_before = int((available * needle_depth) / filler_tokens)
    n_filler_after = int((available * (1 - needle_depth)) / filler_tokens)

    prompt = (
        filler_sentence * n_filler_before
        + needle
        + filler_sentence * n_filler_after
        + retrieval_prompt
    )

    # Truncate to context_len tokens
    prompt_tokens = tokenizer.encode(prompt)[:context_len]
    prompt = tokenizer.decode(prompt_tokens)

    # Standard generation
    std_text, _ = _generate_short(model, tokenizer, prompt, max_tokens=20)
    std_found = passkey in std_text

    # Foveated generation
    fov_text, _ = generate_foveated(model, tokenizer, prompt, max_tokens=20, cfg=cfg)
    fov_found = passkey in fov_text

    return std_found, fov_found, {
        "passkey": passkey,
        "context_len": len(prompt_tokens),
        "needle_depth": needle_depth,
        "standard_output": std_text,
        "foveated_output": fov_text,
    }


def _generate_short(
    model, tokenizer, prompt: str, max_tokens: int = 20
) -> tuple[str, dict]:
    """Generate with standard (non-foveated) cache for baseline comparison."""
    from mlx_lm.models.cache import make_prompt_cache

    tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    cache = make_prompt_cache(model)
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    generated = []
    next_logits = logits[:, -1, :]

    for _ in range(max_tokens):
        next_token = mx.argmax(next_logits, axis=-1)
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        next_input = next_token.reshape(1, 1)
        next_logits = model(next_input, cache=cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits)

    return tokenizer.decode(generated), {"generated_tokens": len(generated)}
