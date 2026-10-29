"""ref_boost for MOSS-Image edit: pure-torch core, no comfy imports (the tests load
this file on its own).

MOSS-Image is single stream like Krea 2, but its sequence is [image | text | refs]
(target first), and with a kv_cache edit LoRA the references never ride in the
per-step sequence: their keys/values are precomputed once at t=0 and appended as
extra keys (comfy/ldm/moss_image/model.py `attention`, ostris nodes.py `_attention`).
So the DiT block attention is

    queries [image | text] (+ refs when kv_cache is off)
    keys    [image | text | refs]

and the ref_boost of comfyui-krea2edit -- an additive log(b) logit bias on
target rows x reference columns -- maps onto it unchanged: rows = image tokens,
columns = each reference's keys. The bias never touches the reference-only pass,
so the cached K/V stay exactly what the LoRA was trained with.
"""
import math

import torch
import torch.nn.functional as F


def boost_list(nref, ref_boost=1.0, ref_boost_a=1.0):
    """krea2edit convention: the LAST reference gets ref_boost, earlier ones ref_boost_a."""
    return [ref_boost_a] * (nref - 1) + [ref_boost]


def is_neutral(ref_boost=1.0, ref_boost_a=1.0, ref_boost_out=1.0):
    return ref_boost == 1.0 and ref_boost_a == 1.0 and ref_boost_out == 1.0


def ref_bias(boosts, boost_mask, n_img, n_txt, ref_lens, ref_hw, n_q, device, dtype,
             boost_out=1.0):
    """Additive attention-logit bias (1, 1, n_q, n_k) for MOSS [image | text | refs].

    boosts:     per-ref factor on image->ref attention, aligned with ref_lens.
    boost_mask: ComfyUI MASK (ref pixel space) restricting the LAST ref's boost to a
                region; needs ref_hw (the refs' latent grids), ignored without it.
    boost_out:  factor for the LAST ref outside boost_mask (krea2edit ref_boost_out).
    n_q:        n_img + n_txt with cached ref K/V, n_k when the refs are in the sequence.
    Only image (target) rows are biased: text rows and reference rows keep plain
    attention, as in krea2edit. Returns None when nothing is biased.
    """
    nref = len(ref_lens)
    n_k = n_img + n_txt + sum(ref_lens)
    offs = [n_img + n_txt]
    for sl in ref_lens:
        offs.append(offs[-1] + sl)
    bias = None
    for i, b in enumerate(boosts):
        masked = boost_mask is not None and i == nref - 1 and ref_hw is not None
        b_out = boost_out if masked else 1.0
        # b == 1.0 alone is not enough to skip: with a mask the outside may be biased
        if b == 1.0 and b_out == 1.0:
            continue
        if bias is None:
            bias = torch.zeros(1, 1, n_q, n_k, device=device, dtype=dtype)
        off, sl = offs[i], ref_lens[i]
        if masked:
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(mask[None].float(), size=tuple(ref_hw[i]), mode="area")[0, 0]
            cols = off + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
            if b_out != 1.0:
                # whole ref first, then the masked region overwrites it below
                bias[:, :, :n_img, off:off + sl] = math.log(max(b_out, 1e-4))
        else:
            cols = torch.arange(off, off + sl, device=device)
        if b != 1.0 or masked:
            bias[:, :, :n_img, cols] = math.log(max(b, 1e-4))
    return bias


def ref_layout(x, context, kwargs):
    """(n_img, n_txt, ref_lens, ref_hw) of a MOSS apply_model call, or None without refs.

    ref_latents (ostris patch, or native with kv_cache off) carry one grid per ref;
    native ai-toolkit/kv_cache formats only pass ref_kv (B, layers, 2, Lr, D), whose
    per-ref split is gone: then ref_lens is one block and ref_hw is None.
    """
    n_img = x.shape[-2] * x.shape[-1]
    n_txt = context.shape[1]
    refs = kwargs.get("ref_latents")
    if refs:
        hw = [tuple(r.shape[-2:]) for r in refs]
        return n_img, n_txt, [h * w for h, w in hw], hw
    ref_kv = kwargs.get("ref_kv")
    if ref_kv is not None:
        return n_img, n_txt, [ref_kv.shape[3]], None
    return None


def seq_len(t, skip_reshape):
    """Sequence length of q/k as optimized_attention receives them."""
    return t.shape[2] if skip_reshape else t.shape[1]


def make_attention_override(boosts, boost_mask, n_img, n_txt, ref_lens, ref_hw,
                            boost_out=1.0, prev_override=None, preferred=None):
    """`optimized_attention_override` (comfy wrap_attn protocol) adding ref_bias to the
    DiT block calls only. Every other call (the Qwen text blocks, a reference-only
    K/V pass) goes through untouched.

    preferred(transformer_options, skip_reshape, lq, lk) -> attention fn or None:
    native MOSS blocks pass their own ComfyAttention, which wrap_attn drops once an
    override is installed; this hands it back so the backend does not change.
    """
    n_k = n_img + n_txt + sum(ref_lens)
    cache = {}

    def override(func, q, k, v, heads, mask=None, skip_reshape=False,
                 skip_output_reshape=False, **kwargs):
        lq, lk = seq_len(q, skip_reshape), seq_len(k, skip_reshape)
        target = func
        if preferred is not None:
            target = preferred(kwargs.get("transformer_options"), skip_reshape, lq, lk) or func
        if lk == n_k and lq in (n_img + n_txt, n_k):
            dtype = mask.dtype if mask is not None and torch.is_floating_point(mask) else q.dtype
            key = (lq, q.device, dtype)
            if key not in cache:
                cache[key] = ref_bias(boosts, boost_mask, n_img, n_txt, ref_lens, ref_hw,
                                      lq, q.device, dtype, boost_out=boost_out)
            bias = cache[key]
            if bias is not None:
                if mask is not None and mask.dtype == torch.bool:
                    mask = torch.zeros(mask.shape, device=q.device, dtype=dtype).masked_fill(~mask, float("-inf"))
                mask = bias if mask is None else mask + bias
        if prev_override is not None:
            return prev_override(target, q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape, **kwargs)
        return target(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kwargs)
    return override


def native_preferred(dit, n_img, n_txt, ref_lens):
    """preferred() for comfy's MOSSImage: block i -> layers[i].comfy_attention.function,
    the 2 text blocks -> theirs. The ostris patch calls optimized_attention with
    skip_reshape=True and no preferred backend, so those calls keep the default."""
    def fn(module):
        return getattr(getattr(module, "comfy_attention", None), "function", None)

    layers = [fn(b) for b in getattr(dit, "layers", [])]
    text = {fn(b) for b in getattr(getattr(dit, "text_embedder", None), "blocks", [])}
    text_fn = next(iter(text)) if len(text) == 1 else None
    n_k = n_img + n_txt + sum(ref_lens)

    def preferred(transformer_options, skip_reshape, lq, lk):
        if skip_reshape:
            return None
        if lk == n_k and lq in (n_img + n_txt, n_k):
            i = (transformer_options or {}).get("block_index")
            return layers[i] if isinstance(i, int) and 0 <= i < len(layers) else None
        if lq == lk == n_txt:
            return text_fn
        return None
    return preferred


def make_apply_model_wrapper(ref_boost=1.0, ref_boost_a=1.0, ref_boost_mask=None,
                             ref_boost_out=1.0, log=print):
    """APPLY_MODEL wrapper: reads the layout of each call and installs the override in
    a COPY of transformer_options (never the caller's dict: cond/uncond and later
    runs must not inherit it)."""
    warned = set()

    def warn_once(msg):
        if msg not in warned:
            warned.add(msg)
            log(f"[moss_ref_boost] WARNING: {msg}")

    def wrapper(executor, x, t, c_concat=None, c_crossattn=None, control=None,
                transformer_options={}, **kwargs):
        layout = None
        if c_crossattn is not None and not is_neutral(ref_boost, ref_boost_a, ref_boost_out):
            layout = ref_layout(x, c_crossattn, kwargs)
        if layout is None:
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        n_img, n_txt, ref_lens, ref_hw = layout
        mask = ref_boost_mask
        if ref_hw is None:
            # cached native refs: one block, per-ref split and grid unknown
            if ref_boost_a != 1.0:
                warn_once("native kv_cache refs arrive as one block: ref_boost applies to "
                          "all of them, ref_boost_a is ignored (use the ostris patch or "
                          "edit_format kv_cache off for per-ref boosts)")
            if mask is not None:
                warn_once("ref_boost_mask needs the reference grid, which native kv_cache "
                          "refs do not carry: mask and ref_boost_out ignored")
                mask = None
            boosts = [ref_boost]
        else:
            boosts = boost_list(len(ref_lens), ref_boost, ref_boost_a)
        if mask is None and ref_boost_out != 1.0:
            warn_once(f"ref_boost_out={ref_boost_out} has NO EFFECT without ref_boost_mask")
        if all(b == 1.0 for b in boosts) and (mask is None or ref_boost_out == 1.0):
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        dit = getattr(getattr(executor, "class_obj", None), "diffusion_model", None)
        to = transformer_options.copy()
        to["optimized_attention_override"] = make_attention_override(
            boosts, mask, n_img, n_txt, ref_lens, ref_hw, boost_out=ref_boost_out,
            prev_override=transformer_options.get("optimized_attention_override"),
            preferred=None if dit is None else native_preferred(dit, n_img, n_txt, ref_lens))
        return executor(x, t, c_concat, c_crossattn, control, to, **kwargs)
    return wrapper
