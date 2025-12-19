"""
MetaHub Save Image - ComfyUI Custom Node
Entry point for node registration
"""

from .metahub_save_node import MetaHubSaveNode

NODE_CLASS_MAPPINGS = {
    "MetaHubSaveNode": MetaHubSaveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaHubSaveNode": "MetaHub Save Image"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
