import logging
import math
import torch
import comfy.samplers

logger = logging.getLogger(__name__)


class Guider_CFGSchedule(comfy.samplers.CFGGuider):
    def set_cfg_schedule(self, schedule):
        if isinstance(schedule, list):
            schedule = torch.tensor(schedule, dtype=torch.float32)
        self.cfg_schedule = schedule
        self.last_sigma_idx = None
        # Set cfg to the first value for compatibility
        if len(schedule) > 0:
            self.cfg = float(schedule[0].item())

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        sched_len = int(self.cfg_schedule.shape[-1]) if hasattr(self, "cfg_schedule") else 0
        if sched_len == 0:
            cond_scale = self.cfg
        else:
            current_sigma = timestep[0].item() if isinstance(timestep, torch.Tensor) else timestep[0]
            
            # Find which step range this sigma falls in
            try:
                sample_sigmas = model_options["transformer_options"]["sample_sigmas"]
                # Sigmas are in descending order. Find where current_sigma fits
                sigma_idx = 0
                for i in range(len(sample_sigmas) - 1):
                    if current_sigma >= sample_sigmas[i + 1]:
                        sigma_idx = i
                        break
                    sigma_idx = i + 1
            except Exception as e:
                logger.warning(f"Could not get sample_sigmas: {e}, using index 0")
                sigma_idx = 0
            
            # Only log when sigma_idx changes (new actual step)
            if self.last_sigma_idx is None or sigma_idx != self.last_sigma_idx:
                self.last_sigma_idx = sigma_idx
                step_idx = min(sigma_idx, sched_len - 1)
                logger.info(f"[CFG Schedule] Step {sigma_idx}: CFG = {float(self.cfg_schedule[step_idx].item()):.1f} (sigma={current_sigma:.4f})")
            
            step_idx = min(sigma_idx, sched_len - 1)
            cond_scale = float(self.cfg_schedule[step_idx].item())
        
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)
        return comfy.samplers.sampling_function(self.inner_model, x, timestep, negative_cond, positive_cond, cond_scale, model_options=model_options, seed=seed)

class TK3RCFGScheduleGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg_schedule": ("STRING", {"default": "3.0, 2.0, 1.5, 1.0", "multiline": False}),
                "interpolate_steps": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("GUIDER", "SIGMAS", "SIGMAS")
    RETURN_NAMES = ("guider", "schedule", "normalized_schedule")
    FUNCTION = "execute"
    CATEGORY = "TK3R/Advanced"

    def execute(self, model, positive, negative, cfg_schedule, interpolate_steps):
        # Parse the string into a list of floats
        try:
            cfg_values = [float(x.strip()) for x in cfg_schedule.split(',')]
            logger.info(f"[CFG Schedule] Parsed CFG schedule: {[f'{x:.1f}' for x in cfg_values]}")
            
            # Interpolate if steps > 0
            if interpolate_steps > 0:
                cfg_values = self._interpolate_schedule(cfg_values, interpolate_steps)
                logger.info(f"[CFG Schedule] Interpolated to {interpolate_steps} steps: {[f'{x:.1f}' for x in cfg_values]}")
            
            cfg_tensor = torch.tensor(cfg_values, dtype=torch.float32)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse cfg_schedule '{cfg_schedule}': {e}. Using default 8.0")
            cfg_tensor = torch.tensor([8.0], dtype=torch.float32)
        
        guider = Guider_CFGSchedule(model)
        guider.set_conds(positive, negative)
        guider.set_cfg_schedule(cfg_tensor)
        
        # Create normalized schedule (0-1 range)
        min_cfg = cfg_tensor.min()
        max_cfg = cfg_tensor.max()
        if max_cfg > min_cfg:
            normalized_schedule = (cfg_tensor - min_cfg) / (max_cfg - min_cfg)
        else:
            # All values are the same, normalize to 1.0
            normalized_schedule = torch.ones_like(cfg_tensor)
        
        return (guider, cfg_tensor, normalized_schedule)
    
    def _interpolate_schedule(self, values, steps):
        """Interpolate the schedule values over the specified number of steps."""
        if len(values) == 1:
            # Single value, repeat it
            return values * steps
        
        if steps == len(values):
            # Already the right length
            return values
        
        # Create interpolated values
        input_indices = torch.linspace(0, len(values) - 1, len(values))
        output_indices = torch.linspace(0, len(values) - 1, steps)
        
        # Linear interpolation
        values_tensor = torch.tensor(values, dtype=torch.float32)
        interpolated = torch.zeros(steps, dtype=torch.float32)
        
        for i, out_idx in enumerate(output_indices):
            # Find the two surrounding input indices
            left_idx = int(torch.floor(out_idx).item())
            right_idx = min(left_idx + 1, len(values) - 1)
            
            # Calculate interpolation weight
            if left_idx == right_idx:
                interpolated[i] = values_tensor[left_idx]
            else:
                weight = out_idx - left_idx
                interpolated[i] = values_tensor[left_idx] * (1 - weight) + values_tensor[right_idx] * weight
        
        return interpolated.tolist()


class Guider_CFGSigmaInterpolate(comfy.samplers.CFGGuider):
    def set_sigma_schedule(self, start_cfg, end_cfg, linear_blend):
        self.start_cfg = start_cfg
        self.end_cfg = end_cfg
        self.linear_blend = linear_blend
        # Set cfg for compatibility
        self.cfg = start_cfg

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        current_sigma = timestep[0].item() if isinstance(timestep, torch.Tensor) else timestep[0]
        
        # Sigma-based interpolation (follows sigma schedule)
        sigma_progress = 1.0 - max(0.0, min(1.0, current_sigma))  # Clamp to [0, 1]
        
        # Linear-based interpolation (based on position in sampling schedule)
        linear_progress = sigma_progress  # Default to same as sigma
        try:
            sample_sigmas = model_options["transformer_options"]["sample_sigmas"]
            # Find position in the sigma schedule
            sigma_idx = 0
            for i in range(len(sample_sigmas) - 1):
                if current_sigma >= sample_sigmas[i + 1]:
                    sigma_idx = i
                    break
                sigma_idx = i + 1
            linear_progress = sigma_idx / max(1, len(sample_sigmas) - 1)
        except Exception as e:
            logger.warning(f"Could not get sample_sigmas for linear progress: {e}")
        
        # Blend between linear and sigma schedules
        # linear_blend = 0.0 means full linear schedule
        # linear_blend = 1.0 means full sigma schedule
        blended_progress = (sigma_progress * self.linear_blend) + (linear_progress * (1.0 - self.linear_blend))
        cond_scale = self.start_cfg + (self.end_cfg - self.start_cfg) * blended_progress
        
        logger.info(f"[CFG Sigma Interpolate] Sigma={current_sigma:.4f}, SigmaProgress={sigma_progress:.4f}, LinearProgress={linear_progress:.4f}, Blend={self.linear_blend:.2f}, CFG={cond_scale:.4f}")
        
        negative_cond = self.conds.get("negative", None)
        positive_cond = self.conds.get("positive", None)
        return comfy.samplers.sampling_function(self.inner_model, x, timestep, negative_cond, positive_cond, cond_scale, model_options=model_options, seed=seed)


class TK3RCFGSigmaInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg_start": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "cfg_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "linear_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }
    
    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "execute"
    CATEGORY = "TK3R/Advanced"

    def execute(self, model, positive, negative, cfg_start, cfg_end, linear_blend):
        logger.info(f"[CFG Sigma Interpolate] Creating guider with start_cfg={cfg_start}, end_cfg={cfg_end}, linear_blend={linear_blend}")
        
        guider = Guider_CFGSigmaInterpolate(model)
        guider.set_conds(positive, negative)
        guider.set_sigma_schedule(cfg_start, cfg_end, linear_blend)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "TK3R Scheduled CFG Guider": TK3RCFGScheduleGuider,
    "TK3R CFG Sigma Interpolate": TK3RCFGSigmaInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TK3R Scheduled CFG Guider": "TK3R Scheduled CFG Guider",
    "TK3R CFG Sigma Interpolate": "TK3R CFG Sigma Interpolate",
}
