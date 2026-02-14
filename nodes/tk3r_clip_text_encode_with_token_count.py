import torch


class TK3RCLIPTextEncodeWithTokenCount:
    """
    CLIPTextEncode with added token count output.
    Encodes text and returns both the conditioning and token count.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": True,
                    "tooltip": "The text to be encoded."
                }),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "system_prompt": (list(cls.SYSTEM_PROMPT.keys()), {"default": "none", "tooltip": "System prompt for Lumina2 models. Use 'none' for other models."}),
            }
        }
    
    SYSTEM_PROMPT = {
        "none": "",
        "lumina2: superior": "You are an assistant designed to generate superior images with the superior "\
            "degree of image-text alignment based on textual prompts or user prompts.",
        "lumina2: alignment": "You are an assistant designed to generate high-quality images with the "\
            "highest degree of image-text alignment based on textual prompts."        
    }
    
    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "token_count", "info")
    FUNCTION = "encode"
    CATEGORY = "TK3R/Utility"
    OUTPUT_NODE = True
    DESCRIPTION = "Encodes text prompt and outputs the conditioning along with token count information."

    def encode(self, clip, text, system_prompt="none"):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        
        # Apply system prompt if selected
        if system_prompt != "none":
            sys_prompt_text = self.SYSTEM_PROMPT[system_prompt]
            text = f'{sys_prompt_text} <Prompt Start> {text}'

        # Tokenize the text
        tokens = clip.tokenize(text)
        
        # Count tokens
        token_counts = {}
        total_tokens = 0
        
        for key, token_data in tokens.items():
            count = 0
            
            if isinstance(token_data, torch.Tensor):
                # Direct tensor: count non-padding tokens
                # For CLIP tokenizers, structure is: [start_token, ...content..., end_token, pad, pad, ...]
                # Common values: start=49406, end=49407, pad=49407
                flat_tokens = token_data.flatten()
                
                # Skip the start token (first token)
                tokens_to_check = flat_tokens[1:] if len(flat_tokens) > 1 else flat_tokens
                
                for i, token_id in enumerate(tokens_to_check):
                    tid = token_id.item()
                    
                    # Check if this is an end/pad token by looking ahead
                    # If we see the same token repeated to the end, it's padding
                    if i < len(tokens_to_check) - 1:
                        # Check if all remaining tokens are the same (padding pattern)
                        remaining = tokens_to_check[i:]
                        if len(remaining) > 1 and all(t.item() == tid for t in remaining):
                            # This is the start of padding, include current token as end token
                            count += 1
                            break
                    
                    count += 1
                
                # Add 1 for the start token
                count += 1
            
            elif isinstance(token_data, list):
                # Handle nested list structures
                if len(token_data) > 0:
                    inner_data = token_data[0]
                    
                    if isinstance(inner_data, list):
                        # Nested list structure (SDXL format: [[token_ids], [weights]])
                        # Count non-padding tokens by looking for repeating end token pattern
                        for i, token_id in enumerate(inner_data):
                            # Check if remaining tokens are all the same (padding pattern)
                            if i > 0 and i < len(inner_data) - 1:
                                remaining = inner_data[i:]
                                if len(remaining) > 1 and all(t == token_id for t in remaining):
                                    # Found padding start, include current token as end token
                                    count = i + 1
                                    break
                        
                        # If no padding pattern found, use full length
                        if count == 0:
                            count = len(inner_data)
                    elif isinstance(inner_data, torch.Tensor):
                        # List containing tensor
                        flat_tokens = inner_data.flatten()
                        
                        # Skip start token
                        tokens_to_check = flat_tokens[1:] if len(flat_tokens) > 1 else flat_tokens
                        
                        for i, token_id in enumerate(tokens_to_check):
                            tid = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
                            
                            # Check for padding pattern
                            if i < len(tokens_to_check) - 1:
                                remaining = tokens_to_check[i:]
                                if len(remaining) > 1:
                                    remaining_vals = [t.item() if isinstance(t, torch.Tensor) else t for t in remaining]
                                    if all(v == tid for v in remaining_vals):
                                        count += 1
                                        break
                            
                            count += 1
                        
                        # Add 1 for start token
                        count += 1
                    else:
                        # Direct list of tokens
                        count = len(token_data)
            
            token_counts[key] = count
            total_tokens = max(total_tokens, count)
        
        # Build info string
        info_lines = [f"Total tokens: {total_tokens}"]
        info_lines.append(f"Text length: {len(text)} characters")
        
        if len(token_counts) > 1:
            info_lines.append(f"\nTokenizer breakdown:")
            for key, count in token_counts.items():
                info_lines.append(f"  {key}: {count} tokens")
        
        info_string = "\n".join(info_lines)
        
        # Encode from tokens (same as CLIPTextEncode)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        return {"ui": {"text": [info_string]}, "result": (conditioning, total_tokens, info_string)}


NODE_CLASS_MAPPINGS = {
    "TK3R CLIP Text Encode With Token Count": TK3RCLIPTextEncodeWithTokenCount,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK3R CLIP Text Encode With Token Count": "TK3R CLIP Text Encode With Token Count",
}
