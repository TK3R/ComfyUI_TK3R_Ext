"""
Custom ControlNet nodes with start/stop percent and decay functionality.
This file contains modified versions of QwenImageDiffsynthControlnet and related classes
that support temporal control with decay modes.

Extracted from comfy_extras/nodes_model_patch.py to preserve custom modifications
after ComfyUI updates.
"""

import math
import torch
import comfy.utils
import comfy.latent_formats
import comfy.model_management
import comfy.ldm.lumina.controlnet


# ============================================================================
# Sigma Window and Decay Utilities
# ============================================================================

SIGMA_EPS = 1e-6


def _clamp_percent(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigma_from_percent(model_sampling, percent: float) -> float:
    sigma = model_sampling.percent_to_sigma(percent)
    if isinstance(sigma, torch.Tensor):
        sigma = float(sigma.detach().cpu())
    return float(sigma)


def resolve_sigma_window(model_sampling, start_percent: float, end_percent: float):
    start = _clamp_percent(start_percent)
    end = _clamp_percent(end_percent)
    sigma_start = _sigma_from_percent(model_sampling, start)
    sigma_end = _sigma_from_percent(model_sampling, end)
    sigma_high = max(sigma_start, sigma_end)
    sigma_low = min(sigma_start, sigma_end)
    if math.isclose(sigma_high, sigma_low, rel_tol=1e-6, abs_tol=SIGMA_EPS):
        sigma_high += SIGMA_EPS
    return sigma_low, sigma_high


def current_sigma(transformer_options):
    if not isinstance(transformer_options, dict):
        return None
    sigma_tensor = transformer_options.get("sigmas")
    if not isinstance(sigma_tensor, torch.Tensor) or sigma_tensor.numel() == 0:
        return None
    return float(sigma_tensor.reshape(-1)[0].item())


def sigma_within_window(transformer_options, sigma_low: float, sigma_high: float, sigma_val=None) -> bool:
    if sigma_val is None:
        sigma_val = current_sigma(transformer_options)
    if sigma_val is None:
        return True
    return (sigma_low - SIGMA_EPS) <= sigma_val <= (sigma_high + SIGMA_EPS)


DECAY_MODES = ["none", "linear", "cosine", "exponential", "inverse_exponential"]


def calculate_strength_multiplier(sigma_val, sigma_low, sigma_high, decay_mode="none"):
    """Calculate strength multiplier based on sigma position within window.
    
    Args:
        sigma_val: Current sigma value
        sigma_low: Low bound of sigma window (corresponds to stop_percent)
        sigma_high: High bound of sigma window (corresponds to start_percent)
        decay_mode: One of "none", "linear", "cosine", "exponential", "inverse_exponential"
    
    Returns:
        Strength multiplier between 0.0 and 1.0
    """
    if sigma_val is None:
        return 1.0
    
    # Outside window check
    if sigma_val < sigma_low - SIGMA_EPS:
        return 0.0
    if sigma_val > sigma_high + SIGMA_EPS:
        return 0.0 if decay_mode != "none" else 1.0
    
    # No decay - instant cutoff at stop_percent
    if decay_mode == "none":
        return 1.0
    
    if sigma_high <= sigma_low:
        return 1.0
    
    # Calculate normalized position: 1.0 at sigma_high (start), 0.0 at sigma_low (stop)
    t = (sigma_val - sigma_low) / (sigma_high - sigma_low)
    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
    
    if decay_mode == "linear":
        return t
    elif decay_mode == "cosine":
        # Smooth S-curve using cosine interpolation
        return (1.0 - math.cos(t * math.pi)) / 2.0
    elif decay_mode == "exponential":
        # Fast decay at start, slow at end (strength drops quickly)
        return 1.0 - math.exp(-3.0 * t) / (1.0 - math.exp(-3.0)) if t > 0 else 0.0
    elif decay_mode == "inverse_exponential":
        # Slow decay at start, fast at end (strength holds longer)
        return (math.exp(3.0 * t) - 1.0) / (math.exp(3.0) - 1.0)
    else:
        return 1.0


# ============================================================================
# Patch Classes with Decay Support
# ============================================================================

class DiffSynthCnetPatch:
    def __init__(self, model_patch, vae, image, strength, mask=None, sigma_low: float = float("-inf"), sigma_high: float = float("inf"),
                 start_percent: float = 0.0, stop_percent: float = 1.0, decay_mode: str = "none"):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask
        self.sigma_low = min(sigma_low, sigma_high)
        self.sigma_high = max(sigma_low, sigma_high)
        self.start_percent = start_percent
        self.stop_percent = stop_percent
        self.decay_mode = decay_mode
        self.encoded_image = model_patch.model.process_input_latent_image(self.encode_latent_cond(image))
        self.encoded_image_size = (image.shape[1], image.shape[2])

    def encode_latent_cond(self, image):
        latent_image = self.vae.encode(image)
        if self.model_patch.model.additional_in_dim > 0:
            if self.mask is None:
                mask_ = torch.ones_like(latent_image)[:, :self.model_patch.model.additional_in_dim // 4]
            else:
                mask_ = comfy.utils.common_upscale(self.mask.mean(dim=1, keepdim=True), latent_image.shape[-1], latent_image.shape[-2], "bilinear", "none")

            return torch.cat([latent_image, mask_], dim=1)
        else:
            return latent_image

    def __call__(self, kwargs):
        transformer_options = kwargs.get("transformer_options")
        block_index = kwargs.get("block_index")
        sigma_val = current_sigma(transformer_options)
        
        # Calculate strength multiplier based on position in sigma window
        strength_mult = calculate_strength_multiplier(sigma_val, self.sigma_low, self.sigma_high, self.decay_mode)
        effective_strength = self.strength * strength_mult
        
        if strength_mult <= 0:
            return kwargs

        x = kwargs.get("x")
        img = kwargs.get("img")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.model_patch.model.process_input_latent_image(self.encode_latent_cond(image_scaled.movedim(1, -1)))
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        img[:, :self.encoded_image.shape[1]] += (self.model_patch.model.control_block(img[:, :self.encoded_image.shape[1]], self.encoded_image.to(img.dtype), block_index) * effective_strength)
        kwargs['img'] = img
        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self):
        return [self.model_patch]


class ZImageControlPatch:
    def __init__(self, model_patch, vae, image, strength, inpaint_image=None, mask=None, sigma_low: float = float("-inf"), sigma_high: float = float("inf"),
                 start_percent: float = 0.0, stop_percent: float = 1.0, decay_mode: str = "none"):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.inpaint_image = inpaint_image
        self.mask = mask
        self.strength = strength
        self.sigma_low = min(sigma_low, sigma_high)
        self.sigma_high = max(sigma_low, sigma_high)
        self.start_percent = start_percent
        self.stop_percent = stop_percent
        self.decay_mode = decay_mode
        
        # Logic from nodes_model_patch.py to handle optional image/inpaint_image
        skip_encoding = False
        if self.image is not None and self.inpaint_image is not None:
            if self.image.shape != self.inpaint_image.shape:
                skip_encoding = True

        if skip_encoding:
            self.encoded_image = None
        else:
            self.encoded_image = self.encode_latent_cond(self.image, self.inpaint_image)
            if self.image is None:
                self.encoded_image_size = (self.inpaint_image.shape[1], self.inpaint_image.shape[2])
            else:
                self.encoded_image_size = (self.image.shape[1], self.image.shape[2])
        self.temp_data = None

    def encode_latent_cond(self, control_image=None, inpaint_image=None):
        latent_image = None
        if control_image is not None:
            latent_image = comfy.latent_formats.Flux().process_in(self.vae.encode(control_image))

        if self.model_patch.model.additional_in_dim > 0:
            if inpaint_image is None:
                inpaint_image = torch.ones_like(control_image) * 0.5
            
            if self.mask is not None:
                # Upscale mask for inpaint image
                mask_inpaint = comfy.utils.common_upscale(self.mask.view(self.mask.shape[0], -1, self.mask.shape[-2], self.mask.shape[-1]).mean(dim=1, keepdim=True), inpaint_image.shape[-2], inpaint_image.shape[-3], "bilinear", "center")
                inpaint_image = ((inpaint_image - 0.5) * mask_inpaint.movedim(1, -1).round()) + 0.5

            inpaint_image_latent = comfy.latent_formats.Flux().process_in(self.vae.encode(inpaint_image))

            if self.mask is None:
                mask_ = torch.zeros_like(inpaint_image_latent)[:, :1]
            else:
                mask_ = comfy.utils.common_upscale(self.mask.view(self.mask.shape[0], -1, self.mask.shape[-2], self.mask.shape[-1]).mean(dim=1, keepdim=True).to(device=inpaint_image_latent.device), inpaint_image_latent.shape[-1], inpaint_image_latent.shape[-2], "nearest", "center")
            
            if latent_image is None:
                 latent_image = comfy.latent_formats.Flux().process_in(self.vae.encode(torch.ones_like(inpaint_image) * 0.5))

            return torch.cat([latent_image, mask_, inpaint_image_latent], dim=1)
        else:
            return latent_image

    def __call__(self, kwargs):
        transformer_options = kwargs.get("transformer_options")
        block_index = kwargs.get("block_index")
        block_type = kwargs.get("block_type", "")
        sigma_val = current_sigma(transformer_options)
        
        # Calculate strength multiplier based on position in sigma window
        strength_mult = calculate_strength_multiplier(sigma_val, self.sigma_low, self.sigma_high, self.decay_mode)
        effective_strength = self.strength * strength_mult
        
        if strength_mult <= 0:
            # Still need to pop img and txt as the original code does
            kwargs.pop("img", None)
            kwargs.pop("txt", None)
            return kwargs

        x = kwargs.get("x")
        img = kwargs.get("img")
        img_input = kwargs.get("img_input")
        txt = kwargs.get("txt")
        pe = kwargs.get("pe")
        vec = kwargs.get("vec")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = None
            if self.image is not None:
                image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center").movedim(1, -1)
                self.encoded_image_size = (image_scaled.shape[-3], image_scaled.shape[-2])

            inpaint_scaled = None
            if self.inpaint_image is not None:
                inpaint_scaled = comfy.utils.common_upscale(self.inpaint_image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center").movedim(1, -1)
                if image_scaled is None:
                     self.encoded_image_size = (inpaint_scaled.shape[-3], inpaint_scaled.shape[-2])

            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.encode_latent_cond(image_scaled, inpaint_scaled)
            comfy.model_management.load_models_gpu(loaded_models)

        cnet_blocks = self.model_patch.model.n_control_layers
        div = round(30 / cnet_blocks)

        cnet_index = (block_index // div)
        cnet_index_float = (block_index / div)

        kwargs.pop("img")  # we do ops in place
        kwargs.pop("txt")

        if cnet_index_float > (cnet_blocks - 1):
            self.temp_data = None
            return kwargs

        if self.temp_data is None or self.temp_data[0] > cnet_index:
            if block_type == "noise_refiner":
                self.temp_data = (-3, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))
            else:
                self.temp_data = (-1, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))

        if block_type == "noise_refiner":
            next_layer = self.temp_data[0] + 1
            self.temp_data = (next_layer, self.model_patch.model.forward_noise_refiner_block(block_index, self.temp_data[1][1], img_input[:, :self.temp_data[1][1].shape[1]], None, pe, vec))
            if self.temp_data[1][0] is not None:
                img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * effective_strength)
        else:
            while self.temp_data[0] < cnet_index and (self.temp_data[0] + 1) < cnet_blocks:
                next_layer = self.temp_data[0] + 1
                self.temp_data = (next_layer, self.model_patch.model.forward_control_block(next_layer, self.temp_data[1][1], img_input[:, :self.temp_data[1][1].shape[1]], None, pe, vec))

            if cnet_index_float == self.temp_data[0]:
                img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * effective_strength)
                if cnet_blocks == self.temp_data[0] + 1:
                    self.temp_data = None

        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
            self.temp_data = None
        return self

    def models(self):
        return [self.model_patch]


# ============================================================================
# Main Node Class
# ============================================================================

class TK3RQwenImageDiffsynthControlnetAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "model_patch": ("MODEL_PATCH",),
                              "vae": ("VAE",),
                              "image": ("IMAGE",),
                              "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "stop_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "decay": (DECAY_MODES, {"default": "none", "tooltip": "Strength decay curve: none=instant cutoff, linear=even decay, cosine=smooth S-curve, exponential=fast initial decay, inverse_exponential=slow initial decay"}),
                              },
                "optional": {"mask": ("MASK",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "diffsynth_controlnet"

    CATEGORY = "TK3R/Advanced"

    def diffsynth_controlnet(self, model, model_patch, vae, image=None, strength=1.0, inpaint_image=None, start_percent=0.0, stop_percent=1.0, decay="none", mask=None):
        model_patched = model.clone()
        if image is not None:
            image = image[:, :, :, :3]
        if inpaint_image is not None:
            inpaint_image = inpaint_image[:, :, :, :3]
        model_sampling = model.get_model_object("model_sampling")
        if model_sampling is None:
            sigma_low, sigma_high = float("-inf"), float("inf")
        else:
            sigma_low, sigma_high = resolve_sigma_window(model_sampling, start_percent, stop_percent)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.ndim == 4:
                mask = mask.unsqueeze(2)
            mask = 1.0 - mask

        if isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control):
            patch = ZImageControlPatch(model_patch, vae, image, strength, inpaint_image=inpaint_image, mask=mask, sigma_low=sigma_low, sigma_high=sigma_high, start_percent=start_percent, stop_percent=stop_percent, decay_mode=decay)
            model_patched.set_model_noise_refiner_patch(patch)
            model_patched.set_model_double_block_patch(patch)
        else:
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask, sigma_low, sigma_high, start_percent, stop_percent, decay))
        return (model_patched,)
    
class TK3RZImageFunControlnet(TK3RQwenImageDiffsynthControlnetAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "model_patch": ("MODEL_PATCH",),
                              "vae": ("VAE",),
                              "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "stop_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "decay": (DECAY_MODES, {"default": "none", "tooltip": "Strength decay curve: none=instant cutoff, linear=even decay, cosine=smooth S-curve, exponential=fast initial decay, inverse_exponential=slow initial decay"}),
                              },
                "optional": {"image": ("IMAGE",), "inpaint_image": ("IMAGE",), "mask": ("MASK",)}}

    CATEGORY = "TK3R/Advanced"


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "TK3RQwenImageDiffsynthControlnetAdvanced": TK3RQwenImageDiffsynthControlnetAdvanced,
    "TK3RZImageFunControlnet": TK3RZImageFunControlnet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK3RQwenImageDiffsynthControlnetAdvanced": "TK3R Qwen Image DiffSynth ControlNet Advanced (Custom)",
    "TK3RZImageFunControlnet": "TK3R ZImage Fun ControlNet (Custom)",
}
