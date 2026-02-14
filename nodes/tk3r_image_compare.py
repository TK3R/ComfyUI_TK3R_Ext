import torch


class TK3RImageCompare:
    """
    Node that outputs both current and previous image inputs.
    Useful for comparing consecutive frames or tracking image changes.
    """
    # Class-level cache to store the last image
    _last_image = {}
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("current", "previous")
    FUNCTION = "compare"

    CATEGORY = "TK3R/Utility"

    def compare(self, image):
        # Generate a unique key for this node instance
        # Using id(self) to track per-instance
        instance_id = id(self)
        
        # Get the previous image for this instance, if it exists
        if instance_id in TK3RImageCompare._last_image:
            previous = TK3RImageCompare._last_image[instance_id]
        else:
            # First run - return current image for both outputs
            previous = image
        
        # Store current image as the new "last" image for next run
        TK3RImageCompare._last_image[instance_id] = image.clone()
        
        return (image, previous)


NODE_CLASS_MAPPINGS = {
    "TK3R Image Compare": TK3RImageCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK3R Image Compare": "TK3R Image Compare",
}
