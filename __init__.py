"""
MetaHub Save Image/Video - ComfyUI Custom Node
Entry point for node registration
"""

try:
    from .metahub_save_node import MetaHubSaveImage, MetaHubSaveNode
    from .metahub_save_video_node import MetaHubSaveVideoNode
    from .timer_node import MetaHubTimerNode
except ImportError:
    from metahub_save_node import MetaHubSaveImage, MetaHubSaveNode
    from metahub_save_video_node import MetaHubSaveVideoNode
    from timer_node import MetaHubTimerNode

NODE_CLASS_MAPPINGS = {
    "MetaHubSaveImage": MetaHubSaveImage,
    "MetaHubSaveNode": MetaHubSaveNode,
    "MetaHubSaveVideoNode": MetaHubSaveVideoNode,
    "MetaHubTimerNode": MetaHubTimerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaHubSaveImage": "MetaHub Save Image",
    "MetaHubSaveNode": "MetaHub Save Image Advanced",
    "MetaHubSaveVideoNode": "MetaHub Save Video",
    "MetaHubTimerNode": "MetaHub Timer"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
