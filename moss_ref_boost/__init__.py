"""MOSS-Image Ref Boost — prototype ComfyUI node, separate from the krea2edit node.

Install: copy or link this folder into ComfyUI/custom_nodes/ (the repo root's
__init__.py does not load it). See README.md.
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
