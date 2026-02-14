import logging
import math
import torch

logger = logging.getLogger(__name__)


DECAY_MODES = ["none", "linear", "cosine", "exponential", "inverse_exponential"]


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
    if math.isclose(sigma_high, sigma_low, rel_tol=1e-6, abs_tol=1e-9):
        sigma_high += 1e-9
    return sigma_low, sigma_high


def current_sigma(transformer_options):
    if not isinstance(transformer_options, dict):
        return None
    sigma_tensor = transformer_options.get("sigmas")
    if not isinstance(sigma_tensor, torch.Tensor) or sigma_tensor.numel() == 0:
        return None
    return float(sigma_tensor.reshape(-1)[0].item())


def calculate_strength_multiplier(sigma_val, sigma_low, sigma_high, decay_mode="none"):
    if sigma_val is None:
        return 1.0

    if sigma_val < sigma_low - 1e-9:
        return 0.0
    if sigma_val > sigma_high + 1e-9:
        return 0.0 if decay_mode != "none" else 1.0

    if decay_mode == "none":
        return 1.0

    if sigma_high <= sigma_low:
        return 1.0

    t = (sigma_val - sigma_low) / (sigma_high - sigma_low)
    t = max(0.0, min(1.0, t))

    if decay_mode == "linear":
        return t
    if decay_mode == "cosine":
        return (1.0 - math.cos(t * math.pi)) / 2.0
    if decay_mode == "exponential":
        return 1.0 - math.exp(-3.0 * t) / (1.0 - math.exp(-3.0)) if t > 0 else 0.0
    if decay_mode == "inverse_exponential":
        return (math.exp(3.0 * t) - 1.0) / (math.exp(3.0) - 1.0)
    return 1.0


class DecayControlNetWrapper:
    """Scales control outputs per step using sigma-window decay.
    
    Transparently wraps a controlnet and delegates all attributes/methods to it,
    except get_control which applies decay-based strength scaling.
    """

    def __init__(self, inner, start_percent: float, end_percent: float, decay_mode: str):
        object.__setattr__(self, 'inner', inner)
        object.__setattr__(self, 'start_percent', start_percent)
        object.__setattr__(self, 'end_percent', end_percent)
        object.__setattr__(self, 'decay_mode', decay_mode)
        object.__setattr__(self, 'sigma_low', float("-inf"))
        object.__setattr__(self, 'sigma_high', float("inf"))

    def __getattr__(self, name):
        """Delegate attribute access to inner controlnet."""
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        """Delegate attribute setting to inner controlnet, except internal state."""
        if name in ('inner', 'start_percent', 'end_percent', 'decay_mode', 'sigma_low', 'sigma_high'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def pre_run(self, model, percent_to_timestep_function):
        try:
            sampling = getattr(model, "model_sampling", None)
            if sampling is not None:
                self.sigma_low, self.sigma_high = resolve_sigma_window(sampling, self.start_percent, self.end_percent)
        except Exception as e:
            logger.warning("DecayControlNetWrapper: failed to resolve sigma window (%s)", e)
        return self.inner.pre_run(model, percent_to_timestep_function)

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        ctrl = self.inner.get_control(x_noisy, t, cond, batched_number, transformer_options)
        if ctrl is None:
            return None

        sigma_val = current_sigma(transformer_options)
        mult = calculate_strength_multiplier(sigma_val, self.sigma_low, self.sigma_high, self.decay_mode)
        if mult == 1.0:
            return ctrl
        if mult == 0.0:
            return None

        try:
            for key in ctrl:
                arr = ctrl[key]
                for i, x in enumerate(arr):
                    if x is not None:
                        arr[i] = x * mult
        except Exception as e:
            logger.warning("DecayControlNetWrapper: scaling failed (%s)", e)
        return ctrl

    def cleanup(self):
        return self.inner.cleanup()

    def get_models(self):
        return self.inner.get_models()

    def get_extra_hooks(self):
        return self.inner.get_extra_hooks()

    def inference_memory_requirements(self, dtype):
        return self.inner.inference_memory_requirements(dtype)

    def copy(self):
        return DecayControlNetWrapper(self.inner.copy(), self.start_percent, self.end_percent, self.decay_mode)

    def set_previous_controlnet(self, prev):
        self.inner.set_previous_controlnet(prev)
        return self

    def set_cond_hint(self, *args, **kwargs):
        return self.inner.set_cond_hint(*args, **kwargs)

class TK3RControlNetApplyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "decay": (DECAY_MODES, {"default": "none", "tooltip": "Strength decay: none, linear, cosine, exponential, inverse_exponential."}),
            },
            "optional": {
                "vae": ("VAE", ),
            },
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "TK3R/Advanced"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, decay="none", vae=None, extra_concat=[]):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat)
                    c_net.set_previous_controlnet(prev_cnet)
                    if decay != "none":
                        c_net = DecayControlNetWrapper(c_net, start_percent, end_percent, decay)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


NODE_CLASS_MAPPINGS = {
    "TK3R ControlNet Apply Advanced": TK3RControlNetApplyAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK3R ControlNet Apply Advanced": "TK3R ControlNet Apply Advanced",
}
