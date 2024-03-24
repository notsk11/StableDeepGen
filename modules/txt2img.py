#@title modules/txt2img.py

from diffusers import DiffusionPipeline
from diffusers.schedulers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, DPMSolverMultistepScheduler
import torch
import numpy as np
from PIL import Image
import random
import sys

def load_diffusion_pipeline(models_id):
    pipeline = DiffusionPipeline.from_pretrained(models_id, torch_dtype=torch.float16).to("cuda")
    return pipeline

def load_scheduler(scheduler_choice, pipeline):
    scheduler_choices = {
        "Euler": EulerDiscreteScheduler.from_config,
        "Euler a": EulerAncestralDiscreteScheduler.from_config,
        "LMS": LMSDiscreteScheduler.from_config,
        "DPM++ 2M": DPMSolverMultistepScheduler.from_config,
        "DPM++ SDE": DPMSolverSinglestepScheduler.from_config,
        "LMS Karras": lambda config, use_karras_sigmas=True: LMSDiscreteScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas),
        "DPM2 Karras": lambda config, use_karras_sigmas=True: KDPM2DiscreteScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas),
        "DPM++ 2M Karras": lambda config, use_karras_sigmas=True: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas),
        "DPM++ SDE Karras": lambda config, use_karras_sigmas=True: DPMSolverSinglestepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas),
        "DPM++ 2M SDE": lambda config, use_karras_sigmas=False: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas, algorithm_type='sde-dpmsolver++'),
        "DPM++ 2M SDE Karras": lambda config, use_karras_sigmas=True: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=use_karras_sigmas, algorithm_type='sde-dpmsolver++'),
    }

    scheduler_input = scheduler_choices[scheduler_choice](pipeline.scheduler.config if hasattr(pipeline, 'scheduler') else None, use_karras_sigmas=True)
    pipeline.scheduler = scheduler_input
    return scheduler_input

def generate_images(pipeline, prompt_str, neg_prompt_str, height_int, width_int, num_steps_int, guid_scale_float, num_images_int, seed_int, scheduler_choice):
    # Set a default value for seed
    if seed_int == "":
        seed = random.randint(0, sys.maxsize)
    else:
        try:
            seed = int(seed_int)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return None

    torch.manual_seed(seed)
    height_int = int(height_int)
    width_int = int(width_int)
    num_steps_int = int(num_steps_int)
    guid_scale_float = float(guid_scale_float)
    num_images_int = int(num_images_int)
    scheduler = load_scheduler(scheduler_choice, pipeline)

    images = pipeline(prompt=prompt_str, negative_prompt=neg_prompt_str, height=height_int, width=width_int, num_inference_steps=num_steps_int, guidance_scale=guid_scale_float, num_images_per_prompt=num_images_int).images
    images_np = [np.array(img) for img in images]
    images_pil = [Image.fromarray(img_np) for img_np in images_np]
    return images_pil, seed, scheduler
