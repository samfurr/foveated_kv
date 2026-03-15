"""
mlx-lm integration for foveated KV cache.

Provides the fused SDPA monkey-patch and generation loops that plug into
mlx-lm's pipeline, routing decode attention through the fused Split-K Metal
kernel operating directly on quantized data.

Pipeline:
  1. Prefill with standard mlx-lm cache → extract K,V
  2. Compress into MLXFoveatedKVCache
  3. Install fused SDPA interceptor (monkey-patches mx.fast)
  4. Decode loop — model calls flow through fused kernel automatically
  5. Uninstall interceptor + cleanup
"""

import math
from typing import Optional

import mlx.core as mx

from .mlx_foveated import MLXFoveatedKVCache, MLXFoveatedLayer, MLXTierConfig


def _log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Numerically stable log-softmax."""
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


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
    pending_spikes: list = []
    _far_idx_np_cache: dict = {}
    _fused_wrappers: list = None

_fused_state = _FusedSDPAState()


class FusedCacheWrapper:
    """Cache wrapper for the fused SDPA path.

    Returns *dummy* K,V tensors because the actual attention is handled by
    the intercepted SDPA function calling directly into the fused Metal kernel
    on quantized data.

    update_and_fetch() still captures new K,V and adds them to the foveal
    tier so the cache stays up-to-date.
    """

    def __init__(self, fov_layer: MLXFoveatedLayer, layer_idx: int):
        self.fov_layer = fov_layer
        self.layer_idx = layer_idx
        self.offset = fov_layer.total_tokens
        self.latest_k = None  # captured for spike detection
        self.latest_q = None  # actual Q from SDPA (set by interceptor)

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
        # Capture new token + save K for spike detection (used by promoter)
        self.fov_layer.add_token(keys, values)
        self.latest_k = keys  # query proxy for spike detection
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

    # Use attend_fused_with_spikes — spike detection is a free byproduct of
    # the kernel's online softmax (just tracks max far score vs min foveal).
    # Zero additional cost vs attend_fused.
    out, flags, tokens = fov_layer.attend_fused_with_spikes(queries)

    # Stash spike arrays on the wrapper. They're part of the same computation
    # graph as `out`, so mx.eval(logits) materializes them for free. We
    # process them in reset_fused_layer_counter (after eval, before next step).
    wrappers = state._fused_wrappers
    if wrappers is not None and layer_idx < len(wrappers):
        w = wrappers[layer_idx]
        if w is not None:
            w.latest_q = queries
            w._spike_flags = flags
            w._spike_tokens = tokens

    return out


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
    _fused_state._fused_wrappers = None  # set by generate_fused after creating wrappers
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
    """Reset layer counter + run spike detection. Call per step.

    Spike detection scores one layer per step (rotating). Detected spikes
    are queued for the background worker, which writes promoted fp16 K,V
    to the shared-memory override buffer. The Metal kernel reads this
    buffer directly — no main-thread promotion application needed.
    """
    state = _fused_state
    state.layer_counter = 0
    state.decode_step += 1

    if state.promoter is None:
        return

    # --- Process kernel spike outputs from last step ---
    # The spike flags/tokens were computed as a FREE byproduct of the fused
    # kernel's online softmax. Batch-eval all layers' spikes in ONE sync,
    # then hand numpy copies to the raw spike worker.
    import numpy as _np
    wrappers_with_spikes = []
    all_spike_arrays = []
    for w in (state._fused_wrappers or []):
        if w is None:
            continue
        flags = getattr(w, '_spike_flags', None)
        tokens = getattr(w, '_spike_tokens', None)
        if flags is not None and tokens is not None:
            all_spike_arrays.extend([flags, tokens])
            wrappers_with_spikes.append(w)

    if all_spike_arrays:
        mx.eval(*all_spike_arrays)  # ONE GPU sync for all 24 layers

        for w in wrappers_with_spikes:
            flags_np = _np.array(w._spike_flags)
            tokens_np = _np.array(w._spike_tokens)
            far_idx_np = _np.array(w.fov_layer.far_idx)
            try:
                state.promoter._raw_spike_queue.put_nowait(
                    (w.layer_idx, flags_np, tokens_np, far_idx_np)
                )
            except Exception:
                pass
            w._spike_flags = None
            w._spike_tokens = None


# ---------------------------------------------------------------------------
# End fused SDPA monkey-patch
# ---------------------------------------------------------------------------


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

    # 3. Start async promoter (optional) + wire override buffers to layers
    promoter = None
    if enable_promotion:
        promoter = AsyncPromoter(fov_cache, disk_archives)
        for i, layer in enumerate(fov_cache.layers):
            if layer is not None:
                layer.overrides = promoter.overrides_for_layer(i)

    # 4. Wrap with FusedCacheWrapper (returns dummy K,V; SDPA is intercepted)
    fused_wrappers = []
    for i, layer in enumerate(fov_cache.layers):
        if layer is None:
            fused_wrappers.append(None)
            continue
        fused_wrappers.append(FusedCacheWrapper(layer, i))

    # 5. Install fused SDPA interceptor
    install_fused_sdpa(fov_cache, promoter=promoter)
    _fused_state._fused_wrappers = fused_wrappers

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
        # Clean up disk archive temp dir
        if disk_archive_dir is not None:
            import shutil
            shutil.rmtree(disk_archive_dir, ignore_errors=True)

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

    # --- Foveated PPL (fused path) ---
    fov_cache, _, _ = prefill_and_compress(model, context, cfg)
    fov_wrappers = [
        FusedCacheWrapper(layer, i) if layer is not None else None
        for i, layer in enumerate(fov_cache.layers)
    ]
    install_fused_sdpa(fov_cache)

    fov_nll = 0.0
    try:
        for i in range(eval_len):
            reset_fused_layer_counter()
            tok = eval_tokens[i : i + 1].reshape(1, 1)
            logits = model(tok, cache=fov_wrappers)
            log_probs = _log_softmax(logits[:, -1, :])
            mx.eval(log_probs)
            if i + 1 < eval_len:
                target = eval_tokens[i + 1].item()
                fov_nll -= log_probs[0, target].item()
    finally:
        uninstall_fused_sdpa()

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

    # Foveated generation (fused path)
    fov_text, _ = generate_fused(
        model, tokenizer, prompt, max_tokens=20, cfg=cfg, enable_promotion=False,
    )
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
