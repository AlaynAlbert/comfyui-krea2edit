"""MOSS-Image Ref Boost: the comfyui-krea2edit reference-fidelity dials (ref_boost,
ref_boost_a, ref_boost_mask + ref_boost_out) for MOSS-Image edit.

Works on top of either edit path, wired AFTER it:
  - native: TextEncodeMOSSImageEdit (ComfyUI master, PR #17296) on the stock model;
  - ostris: MOSSImageOstrisEditModelPatch (kv_cache on or off).
SkipLayerGuidanceDiTSimple (single_layers 22) can sit before or after it.

With every dial at 1.0 the node returns the model it was given, untouched: no
clone, no wrapper, so the output is bit-identical to not having the node.
"""
from . import core

TIP_OUT = ("companion to ref_boost_mask: the dial for the LAST reference OUTSIDE the mask "
           "(ref_boost applies inside it). 1.0 = off. No effect without ref_boost_mask, "
           "and none on native kv_cache refs (their grid is not available)")


class MOSSImageRefBoost:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                        "tooltip": "attention bias target->LAST reference (the only one in single-reference edits): >1 pulls harder toward it, <1 loosens. 1.0 = off"}),
                "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                          "tooltip": "the same dial for every EARLIER reference (the scene in two-reference edits). Ignored on native kv_cache refs, which arrive as one block"}),
            },
            "optional": {
                "ref_boost_mask": ("MASK", {"tooltip": "optional region on the LAST reference (e.g. the face) where ref_boost applies; empty = whole reference"}),
                "ref_boost_out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                            "tooltip": TIP_OUT}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches/moss_image"
    DESCRIPTION = ("Reference-fidelity dials for MOSS-Image edit (port of comfyui-krea2edit "
                   "ref_boost): an additive log(b) bias on image->reference attention in "
                   "every DiT block. The reference K/V cache is never touched. All 1.0 = "
                   "the model passes through untouched.")

    def patch(self, model, ref_boost=1.0, ref_boost_a=1.0, ref_boost_mask=None,
              ref_boost_out=1.0, **_future):
        if _future:
            print(f"[moss_ref_boost] WARNING: unknown inputs ({', '.join(sorted(_future))}) "
                  f"ignored; update the node", flush=True)
        if core.is_neutral(ref_boost, ref_boost_a, ref_boost_out):
            return (model,)
        import comfy.patcher_extension
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "moss_ref_boost",
                               core.make_apply_model_wrapper(ref_boost, ref_boost_a, ref_boost_mask,
                                                             ref_boost_out,
                                                             log=lambda s: print(s, flush=True)))
        return (m,)


NODE_CLASS_MAPPINGS = {"MOSSImageRefBoost": MOSSImageRefBoost}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSImageRefBoost": "MOSS-Image Ref Boost (krea2edit port)"}
