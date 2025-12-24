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
            "required": {
                # Accept any type as passthrough to ensure early execution
                "trigger": ("*",),
            },
        }

    RETURN_TYPES = ("*", "FLOAT")
    RETURN_NAMES = ("passthrough", "elapsed_time")
    FUNCTION = "measure_time"
    CATEGORY = "image/save"
    DESCRIPTION = "Measures workflow execution time for MetaHub Save Node"

    def measure_time(self, trigger):
        """
        Measures elapsed time since workflow start.

        Args:
            trigger: Any input to trigger early execution (passed through unchanged)

        Returns:
            (passthrough, elapsed_time): Original input and elapsed time in seconds
        """
        current_time = time.time()

        # Use trigger object ID as workflow identifier
        # This works because ComfyUI reuses the same objects within a workflow execution
        workflow_id = id(trigger)

        # Start timer on first execution
        if workflow_id not in _TIMER_START_TIMES:
            _TIMER_START_TIMES[workflow_id] = current_time
            print(f"[MetaHub Timer] Started timer for workflow")

            # Cleanup old timers (keep only last 20)
            if len(_TIMER_START_TIMES) > 20:
                oldest_key = min(_TIMER_START_TIMES.keys(), key=lambda k: _TIMER_START_TIMES[k])
                del _TIMER_START_TIMES[oldest_key]

        # Calculate elapsed time
        elapsed = current_time - _TIMER_START_TIMES[workflow_id]

        # Passthrough the input unchanged + return elapsed time
        return (trigger, elapsed)


NODE_CLASS_MAPPINGS = {
    "MetaHubTimerNode": MetaHubTimerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaHubTimerNode": "MetaHub Timer"
}
