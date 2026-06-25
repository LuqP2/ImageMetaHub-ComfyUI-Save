"""
MetaHub Save Image/Video - ComfyUI Custom Node
Entry point for node registration
"""

try:
    from .metahub_input_node import MetaHubInputNode
    from .metahub_save_node import MetaHubSaveImage, MetaHubSaveNode
    from .metahub_save_video_node import MetaHubSaveVideoNode
    from .timer_node import MetaHubTimerNode
except ImportError:
    from metahub_input_node import MetaHubInputNode
    from metahub_save_node import MetaHubSaveImage, MetaHubSaveNode
    from metahub_save_video_node import MetaHubSaveVideoNode
    from timer_node import MetaHubTimerNode

NODE_CLASS_MAPPINGS = {
    "MetaHubInputNode": MetaHubInputNode,
    "MetaHubSaveImage": MetaHubSaveImage,
    "MetaHubSaveNode": MetaHubSaveNode,
    "MetaHubSaveVideoNode": MetaHubSaveVideoNode,
    "MetaHubTimerNode": MetaHubTimerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaHubInputNode": "MetaHub Input",
    "MetaHubSaveImage": "MetaHub Save Image",
    "MetaHubSaveNode": "MetaHub Save Image Advanced",
    "MetaHubSaveVideoNode": "MetaHub Save Video",
    "MetaHubTimerNode": "MetaHub Timer"
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
