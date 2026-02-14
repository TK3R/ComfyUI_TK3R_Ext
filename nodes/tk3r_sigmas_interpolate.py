"""
Interpolate between two sigma schedules by averaging each pair
"""

import torch
import logging

logger = logging.getLogger(__name__)

class TK3RSigmasInterpolate:
    """Interpolate between two sigma schedules by averaging corresponding values"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"tooltip": "First sigma schedule"}),
                "sigmas_2": ("SIGMAS", {"tooltip": "Second sigma schedule"}),
            },
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "interpolate_sigmas"
    CATEGORY = "TK3R/Utility"
    DESCRIPTION = "Interpolate between two sigma schedules by averaging each pair. If lengths don't match, returns first sigma schedule unchanged."
    
    def interpolate_sigmas(self, sigmas_1, sigmas_2):
        # Check if lengths match
        if len(sigmas_1) != len(sigmas_2):
            logger.error(
                f"TK3RSigmasInterpolate: Sigma length mismatch! "
                f"sigmas_1 has {len(sigmas_1)} values, sigmas_2 has {len(sigmas_2)} values. "
                f"Returning sigmas_1 unchanged."
            )
            return (sigmas_1,)
        
        # Average the two sigma schedules
        interpolated_sigmas = (sigmas_1 + sigmas_2) / 2.0
        
        return (interpolated_sigmas,)


NODE_CLASS_MAPPINGS = {
    "TK3RSigmasInterpolate": TK3RSigmasInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK3RSigmasInterpolate": "TK3R Sigmas Interpolate",
}
