import node_helpers
import comfy.utils
import math
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

class TK3RTextEncodeQwenImageEditPlus:
    """
    This an enhanced version of the TextEncodeQwenImageEditPlus node, it is NOT my code, I originally got it from some repo and it was supposed to just overwrite the original file but this is not a good approach with all the constant Comfy updates, so I have added it as a separate node in my repo.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE", ),
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "image4": ("IMAGE", ),
                "target_latent": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "execute"
    CATEGORY = "TK3R/Advanced"

    def execute(self, clip, prompt, vae=None, image1=None, image2=None, image3=None, image4=None, target_latent=None):
        ref_latents = []
        images = [image1, image2, image3, image4]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe key details of the input image (including any objects, characters, poses, facial features, clothing, setting, textures and style), then explain how the user's text instruction should alter, modify or recreate the image. Generate a new image that meets the user's requirements, which can vary from a small change to a completely new image using inputs as a guide.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    if target_latent is not None:
                        twidth = target_latent["samples"].shape[-1] * 8
                        theight = target_latent["samples"].shape[-2] * 8
                        s = comfy.utils.common_upscale(samples, twidth, theight, "lanczos", "center")       
                    else:
                        s = samples
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return (conditioning,)

NODE_CLASS_MAPPINGS = {
    "TK3RTextEncodeQwenImageEditPlusExt": TK3RTextEncodeQwenImageEditPlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK3RTextEncodeQwenImageEditPlusExt": "TK3R TextEncodeQwenImageEditPlus Ext"
}