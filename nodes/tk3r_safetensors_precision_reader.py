"""
Safetensors Precision Reader Node

A ComfyUI node that reads and displays the floating-point precision of a safetensors file.
"""

import os
import logging
import traceback

import folder_paths


class TK3RSafetensorsPrecisionReader:
    """
    Reads a safetensors file and reports the floating-point precision used (fp8, fp16, bf16, fp32, etc.)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Full path to safetensors file"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PRECISION_INFO",)
    FUNCTION = "read_precision"
    CATEGORY = "TK3R/Utility"
    OUTPUT_NODE = True

    def read_precision(self, file_path):
        import safetensors.torch
        
        try:
            # If relative path provided, try to resolve from models folder
            if not os.path.isabs(file_path):
                models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                models_dir = os.path.dirname(models_dir)  # Go up to models folder
                file_path = os.path.join(models_dir, file_path)
            
            logging.info(f"\n[TK3R Sft Precision Reader] Reading file: {file_path}")
            
            dtype_counts = {}
            total_tensors = 0
            
            with safetensors.torch.safe_open(file_path, framework="pt") as f:
                keys = list(f.keys())
                total_tensors = len(keys)
                
                logging.info(f"  Total tensors: {total_tensors}")
                
                for key in keys:
                    tensor = f.get_tensor(key)
                    dtype_str = str(tensor.dtype).replace('torch.', '')
                    
                    if dtype_str not in dtype_counts:
                        dtype_counts[dtype_str] = 0
                    dtype_counts[dtype_str] += 1
            
            # Determine primary precision
            if not dtype_counts:
                return ("ERROR: No tensors found in file",)
            
            # Sort by count to find most common dtype
            sorted_dtypes = sorted(dtype_counts.items(), key=lambda x: x[1], reverse=True)
            primary_dtype = sorted_dtypes[0][0]
            primary_count = sorted_dtypes[0][1]
            
            # Map torch dtypes to common names
            precision_map = {
                'float32': 'fp32',
                'float16': 'fp16',
                'bfloat16': 'bf16',
                'float8_e4m3fn': 'fp8 (e4m3)',
                'float8_e5m2': 'fp8 (e5m2)',
                'int8': 'int8',
                'uint8': 'uint8',
            }
            
            primary_precision = precision_map.get(primary_dtype, primary_dtype)
            
            # Build detailed info
            info_lines = [
                f"File: {os.path.basename(file_path)}",
                f"Primary Precision: {primary_precision}",
                f"Total Tensors: {total_tensors}",
                "\nPrecision Breakdown:"
            ]
            
            for dtype, count in sorted_dtypes:
                precision_name = precision_map.get(dtype, dtype)
                percentage = (count / total_tensors) * 100
                info_lines.append(f"  {precision_name}: {count} tensors ({percentage:.1f}%)")
            
            result = "\n".join(info_lines)
            logging.info(f"\n[TK3R Sft Precision Reader] Result:\n{result}")
            
            return (result,)
            
        except Exception as e:
            error_message = f"ERROR: {str(e)}\n{traceback.format_exc()}"
            logging.error(f"\n[TK3R Sft Precision Reader] {error_message}")
            return (error_message,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "TK3R Safetensors Precision Reader": TK3RSafetensorsPrecisionReader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK3RSafetensorsPrecisionReader": "TK3R Safetensors Precision Reader",
}
