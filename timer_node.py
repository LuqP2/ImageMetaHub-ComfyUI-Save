"""
MetaHub Timer Node for ComfyUI
Measures workflow execution time from start to finish
"""

import time


class MetaHubTimerNode:
    """
    Timer node that records a timestamp for workflow execution timing.

    Usage:
    1. Place this node where you want to START measuring time
    2. Connect any input (clip, image, latent, conditioning) to trigger it
    3. Connect the 'elapsed_time' output to MetaHub Save Node's 'generation_time_override'

    How it works:
    - This node records a timestamp when it executes
    - The Save Node calculates elapsed time from this timestamp
    - Place multiple timers to measure different workflow stages
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "conditioning": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CLIP", "IMAGE", "LATENT", "CONDITIONING", "FLOAT")
    RETURN_NAMES = ("clip", "image", "latent", "conditioning", "elapsed_time")
    FUNCTION = "measure_time"
    CATEGORY = "image/save"
    DESCRIPTION = "Measures workflow execution time for MetaHub Save Node"
    OUTPUT_NODE = False

    def measure_time(self, clip=None, image=None, latent=None, conditioning=None):
        """
        Records the current timestamp when this node executes.
        The Save Node will calculate elapsed time from this timestamp.

        Args:
            clip: Optional CLIP input (passed through unchanged)
            image: Optional IMAGE input (passed through unchanged)
            latent: Optional LATENT input (passed through unchanged)
            conditioning: Optional CONDITIONING input (passed through unchanged)

        Returns:
            (clip, image, latent, conditioning, start_timestamp): Original inputs and start timestamp
        """
        # Return current timestamp - Save Node will calculate elapsed time
        start_timestamp = time.time()
        print(f"[MetaHub Timer] Recording start timestamp: {start_timestamp}")

        # Passthrough the inputs unchanged + return start timestamp
        return (clip, image, latent, conditioning, start_timestamp)


NODE_CLASS_MAPPINGS = {
    "MetaHubTimerNode": MetaHubTimerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaHubTimerNode": "MetaHub Timer"
}
