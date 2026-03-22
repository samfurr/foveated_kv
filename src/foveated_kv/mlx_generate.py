"""
mlx-lm integration for foveated KV cache.

Provides the fused SDPA interceptor and generation loops that plug into
mlx-lm's pipeline, routing decode attention through the fused Metal kernel
operating directly on quantized data.

Architecture:
  The SDPA interceptor monkey-patches mx.fast.scaled_dot_product_attention.
  During decode (seq_len=1), it routes attention through the fused Metal
  kernel which operates on the blob (near fp16 + far fp8/int4) directly.
  The kernel produces spike_flags/tokens as a free byproduct of attention.

  Between steps, reset_fused_layer_counter() drains spikes into the C++
  PromotionPipeline, which reads fp16 from disk mmap and writes into the
  blob's near-tier headroom. The kernel sees promoted tokens on next
  dispatch via near_valid[h] — zero overhead, zero kernel changes.

  When the C++ extension isn't available, the fallback reconstructs fp16
  from compressed tiers and uses standard SDPA. No promotion — just
  compressed, frozen tiers. The C++ path is for speed; the fallback is
  for correctness.
"""

import math
from typing import Optional

import mlx.core as mx

from .mlx_foveated import MLXFoveatedKVCache, MLXFoveatedLayer, MLXTierConfig


def _log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Numerically stable log-softmax."""
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Fused SDPA interceptor
# ---------------------------------------------------------------------------

class _FusedSDPAState:
    """Minimal state for the SDPA interceptor."""
    original_sdpa = None
    _installed: bool = False
    _fov_cache: Optional[MLXFoveatedKVCache] = None
    _cpp_pipeline_handle = None
    _layer_counter: int = 0
    _n_layers: int = 0
    _decode_step: int = 0
    _fused_wrappers: list = None
    _fused_disabled: bool = False

_fused_state = _FusedSDPAState()


class FusedCacheWrapper:
    """Cache wrapper that routes attention through the fused Metal kernel.

    Returns dummy K,V tensors (the model expects them for shape compliance)
    while actual attention is handled by the SDPA interceptor calling into
    the fused kernel on quantized data.
    """

    def __init__(self, fov_layer: MLXFoveatedLayer, layer_idx: int):
        self.fov_layer = fov_layer
        self.layer_idx = layer_idx
        self.offset = fov_layer.total_tokens
        self._spike_flags = None
        self._spike_tokens = None

    @property
    def state(self):
        """mlx-lm checks this to determine if cache has data."""
        return self.fov_layer.near_k, self.fov_layer.near_v

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Add new token to decode buffer; return K,V for shape compliance."""
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

    Prefill (seq_len > 1): passes through to the original SDPA.
    Decode (seq_len == 1): routes through the fused Metal kernel.
    """
    seq_len = queries.shape[2]

    if seq_len > 1:
        return _fused_state.original_sdpa(
            queries, keys, values, scale=scale, mask=mask, **kwargs
        )

    state = _fused_state
    layer_idx = state._layer_counter
    state._layer_counter += 1

    if layer_idx >= state._n_layers:
        layer_idx = layer_idx % state._n_layers

    fov_layer = state._fov_cache.layers[layer_idx]
    if fov_layer is None or state._fused_disabled:
        return state.original_sdpa(
            queries, keys, values, scale=scale, mask=mask, **kwargs
        )

    try:
        out, flags, tokens = fov_layer.attend_fused_with_spikes(queries)
    except Exception:
        import logging
        logging.getLogger("foveated_kv").warning(
            "Fused Metal kernel failed — falling back to standard SDPA "
            "for the rest of this session"
        )
        state._fused_disabled = True
        return state.original_sdpa(
            queries, keys, values, scale=scale, mask=mask, **kwargs
        )

    # Stash spike arrays on the wrapper for drain at end of step.
    # They're part of the same computation graph as `out`, so
    # mx.eval(logits) materializes them for free.
    wrappers = state._fused_wrappers
    if wrappers is not None and layer_idx < len(wrappers):
        w = wrappers[layer_idx]
        if w is not None:
            w._spike_flags = flags
            w._spike_tokens = tokens

    return out


def install_fused_sdpa(fov_cache: MLXFoveatedKVCache) -> None:
    """Install the fused SDPA interceptor.

    Monkey-patches mx.fast.scaled_dot_product_attention so decode-time
    attention (seq_len=1) routes through the fused Metal kernel.
    """
    if _fused_state._installed:
        raise RuntimeError("Fused SDPA already installed — call uninstall_fused_sdpa() first")

    _fused_state.original_sdpa = mx.fast.scaled_dot_product_attention
    _fused_state._fov_cache = fov_cache
    _fused_state._cpp_pipeline_handle = None
    _fused_state._layer_counter = 0
    _fused_state._n_layers = len(fov_cache.layers)
    _fused_state._decode_step = 0
    _fused_state._fused_wrappers = None
    _fused_state._fused_disabled = False
    _fused_state._installed = True

    mx.fast.scaled_dot_product_attention = _fused_sdpa_interceptor


def uninstall_fused_sdpa() -> None:
    """Restore the original mx.fast.scaled_dot_product_attention."""
    if not _fused_state._installed:
        return

    mx.fast.scaled_dot_product_attention = _fused_state.original_sdpa
    _fused_state.original_sdpa = None
    _fused_state._fov_cache = None
    _fused_state._cpp_pipeline_handle = None
    _fused_state._layer_counter = 0
    _fused_state._n_layers = 0
    _fused_state._decode_step = 0
    _fused_state._fused_wrappers = None
    _fused_state._fused_disabled = False
    _fused_state._installed = False


def reset_fused_layer_counter() -> None:
    """Reset layer counter and drain spikes into the C++ pipeline.

    Called once per decode step, after mx.eval(logits) has materialized
    the spike arrays. The C++ pipeline reads spike data zero-copy from
    unified memory, filters (cooldown, dedup, budget), and queues
    promotions for the background worker.
    """
    state = _fused_state
    state._layer_counter = 0
    state._decode_step += 1

    pipeline = state._cpp_pipeline_handle
    if pipeline is None:
        return

    for w in (state._fused_wrappers or []):
        if w is None:
            continue
        flags = w._spike_flags
        tokens = w._spike_tokens
        if flags is not None and tokens is not None:
            pipeline.drain_spikes(
                w.layer_idx, flags, tokens,
                w.fov_layer.far_idx, state._decode_step)
        w._spike_flags = None
        w._spike_tokens = None


# ---------------------------------------------------------------------------
# Generation
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
    """Generate text using the fused Metal kernel.

    Pipeline:
      1. Prefill with standard mlx-lm cache → extract K,V
      2. Compress into foveated tiers + offload archives to disk
      3. Set up C++ promotion pipeline (if available)
      4. Install fused SDPA interceptor
      5. Decode loop — model calls flow through fused kernel automatically
      6. Cleanup

    Returns:
        generated_text: decoded output
        stats: dict with timing and promotion info
    """
    import time

    from .disk_archive import offload_cache_to_disk
    from .mlx_foveated import _cpp_available, _PromotionPipeline

    tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)

    # 1. Prefill + compress
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

    # 3. Set up C++ promotion pipeline
    cpp_pipeline = None
    if enable_promotion and _cpp_available and _PromotionPipeline is not None:
        import numpy as _np
        n_layers = len(fov_cache.layers)
        cpp_pipeline = _PromotionPipeline(n_layers)

        for i, layer in enumerate(fov_cache.layers):
            if layer is None:
                continue
            layer._ensure_kcache()
            handle = layer._kcache.get("cpp_handle")
            if handle is not None:
                cpp_pipeline.register_blob(i, handle.get_blob_info())
            if i < len(disk_archives) and disk_archives[i] is not None:
                archive = disk_archives[i]
                archive_idx = _np.array(archive.idx).flatten().tolist()
                cpp_pipeline.register_archive(
                    i, archive.path_k, archive.path_v,
                    archive.H_kv, archive.S_arc, archive.D,
                    archive_idx)

    # 4. Wrap layers + install interceptor
    fused_wrappers = [
        FusedCacheWrapper(layer, i) if layer is not None else None
        for i, layer in enumerate(fov_cache.layers)
    ]
    install_fused_sdpa(fov_cache)
    _fused_state._fused_wrappers = fused_wrappers
    _fused_state._cpp_pipeline_handle = cpp_pipeline

    # 5. Decode loop
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

            reset_fused_layer_counter()

            next_input = next_token.reshape(1, 1)
            next_logits = model(next_input, cache=fused_wrappers)
            next_logits = next_logits[:, -1, :]
            mx.eval(next_logits)

    finally:
        uninstall_fused_sdpa()
        if cpp_pipeline is not None:
            cpp_pipeline.stop()
        import shutil
        shutil.rmtree(disk_archive_dir, ignore_errors=True)

    t_decode = time.perf_counter() - t_decode_start

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
    if cpp_pipeline is not None:
        stats.update(cpp_pipeline.get_stats())

    return tokenizer.decode(generated), stats


def prefill_and_compress(
    model,
    tokens: mx.array,
    cfg: Optional[MLXTierConfig] = None,
) -> tuple[MLXFoveatedKVCache, mx.array, list]:
    """Prefill a model and compress its KV cache into foveated tiers.

    Returns:
        fov_cache: compressed MLXFoveatedKVCache
        prefill_logits: (1, S, vocab) logits from prefill
        std_caches: the original standard caches (for reference)
    """
    from mlx_lm.models.cache import make_prompt_cache

    cfg = cfg or MLXTierConfig()

    std_caches = make_prompt_cache(model)
    prefill_logits = model(tokens, cache=std_caches)
    mx.eval(prefill_logits)

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

    Tests both standard and foveated caches.

    Returns:
        standard_found: did standard cache find the passkey?
        foveated_found: did foveated cache find the passkey?
        info: dict with passkey, generated text, etc.
    """
    import random

    cfg = cfg or MLXTierConfig()
    passkey = str(random.randint(10000, 99999))

    needle = f"The secret passkey is {passkey}. Remember it."
    filler_sentence = "This is a document about various topics in science and technology. "
    retrieval_prompt = "\nWhat is the secret passkey mentioned in the text above? The passkey is: "

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

    prompt_tokens = tokenizer.encode(prompt)[:context_len]
    prompt = tokenizer.decode(prompt_tokens)

    std_text, _ = _generate_short(model, tokenizer, prompt, max_tokens=20)
    std_found = passkey in std_text

    fov_text, _ = generate_fused(
        model, tokenizer, prompt, max_tokens=20, cfg=cfg, enable_promotion=True,
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
