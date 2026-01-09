"""
MetaHub Save Image - ComfyUI Custom Node
Entry point for node registration
"""

try:
    from .metahub_save_node import MetaHubSaveNode
    from .timer_node import MetaHubTimerNode
except ImportError:
    from metahub_save_node import MetaHubSaveNode
    from timer_node import MetaHubTimerNode

NODE_CLASS_MAPPINGS = {
    "MetaHubSaveNode": MetaHubSaveNode,
    "MetaHubTimerNode": MetaHubTimerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaHubSaveNode": "MetaHub Save Image",
    "MetaHubTimerNode": "MetaHub Timer"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
