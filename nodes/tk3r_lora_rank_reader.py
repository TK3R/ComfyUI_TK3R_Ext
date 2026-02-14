"""
LoRA Rank Reader Node

A ComfyUI node that reads and displays the rank of a LoRA safetensors file.
"""

from safetensors import safe_open
import os

import folder_paths


class TK3RLoRARankReader:
    """
    Reads a LoRA .safetensors file and outputs information about its rank.
    
    The rank (r) is the bottleneck dimension in LoRA matrices, which determines
    the model's capacity and file size.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "read_rank"
    CATEGORY = "TK3R/Utility"
    DESCRIPTION = "Reads and displays the rank of a LoRA safetensors file."
    OUTPUT_NODE = True

    def read_rank(self, lora_name):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        
        try:
            with safe_open(lora_path, framework="pt") as f:
                # Search for the first 'A' or 'down' matrix to identify the rank
                # These matrices have the shape (rank, original_dimension)
                keys = f.keys()
                target_key = next((k for k in keys if "lora_A" in k or "lora_down" in k), None)

                if target_key:
                    tensor_shape = f.get_slice(target_key).get_shape()
                    # The rank 'r' is the smaller bottleneck dimension, typically the first index
                    rank = tensor_shape[0]
                    
                    info_lines = [
                        f"=== LoRA Analysis ===",
                        f"File: {os.path.basename(lora_path)}",
                        f"Rank (r): {rank}",
                        f"Sample key: {target_key}",
                        f"Tensor shape: {tensor_shape}"
                    ]
                    
                    # Check for Alpha in metadata if it exists
                    metadata = f.metadata()
                    if metadata and 'ss_network_alpha' in metadata:
                        info_lines.append(f"Network Alpha: {metadata['ss_network_alpha']}")
                    
                    # Count total LoRA layers
                    lora_layer_count = len([k for k in keys if "lora_A" in k or "lora_down" in k])
                    info_lines.append(f"Total LoRA layers: {lora_layer_count}")
                    
                    info_text = "\n".join(info_lines)
                    print(info_text)
                    return (info_text,)
                else:
                    error_msg = f"Error: Could not find LoRA weight matrices in {lora_path}"
                    print(error_msg)
                    return (error_msg,)
                    
        except Exception as e:
            error_msg = f"Error opening file: {str(e)}"
            print(error_msg)
            return (error_msg,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "TK3RLoRARankReader": TK3RLoRARankReader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK3RLoRARankReader": "LoRA Rank Reader",
}
