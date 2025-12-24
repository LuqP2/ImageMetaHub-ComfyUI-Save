"""
MetaHub Timer Node for ComfyUI
Measures workflow execution time from start to finish
"""

import time

# Global storage for workflow timers
_TIMER_START_TIMES = {}


class MetaHubTimerNode:
    """
    Timer node that measures workflow execution time.

    Usage:
    1. Place this node at the START of your workflow
    2. Connect any input (image, latent, conditioning, etc.) to trigger it early
    3. Connect the 'elapsed_time' output to MetaHub Save Node's 'generation_time_override'

    The timer starts when this node first executes in a workflow run.
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
        Measures elapsed time since this timer node was first executed.
        Each timer node instance tracks its own independent timing.

        Args:
            clip: Optional CLIP input (passed through unchanged)
            image: Optional IMAGE input (passed through unchanged)
            latent: Optional LATENT input (passed through unchanged)
            conditioning: Optional CONDITIONING input (passed through unchanged)

        Returns:
            (clip, image, latent, conditioning, elapsed_time): Original inputs and elapsed time in seconds
        """
        current_time = time.time()

        # Use this node instance's ID as unique timer identifier
        # This allows multiple independent timers in the same workflow
        timer_id = id(self)

        # Start timer on first execution of this specific timer instance
        if timer_id not in _TIMER_START_TIMES:
            _TIMER_START_TIMES[timer_id] = current_time
            print(f"[MetaHub Timer] Started timer #{timer_id}")

            # Cleanup old timers (keep only last 50 to support multiple timers)
            if len(_TIMER_START_TIMES) > 50:
                oldest_key = min(_TIMER_START_TIMES.keys(), key=lambda k: _TIMER_START_TIMES[k])
                del _TIMER_START_TIMES[oldest_key]

        # Calculate elapsed time
        elapsed = current_time - _TIMER_START_TIMES[timer_id]
        print(f"[MetaHub Timer] Elapsed time: {elapsed:.2f}s")

        # Passthrough the inputs unchanged + return elapsed time
        return (clip, image, latent, conditioning, elapsed)


NODE_CLASS_MAPPINGS = {
    "MetaHubTimerNode": MetaHubTimerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaHubTimerNode": "MetaHub Timer"
}
