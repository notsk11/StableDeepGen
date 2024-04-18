# /content/modules/scheduler.py
import torch
from modules import pipeline

from diffusers import (
    PNDMScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler,
)

def update_scheduler(scheduler_name):
    if scheduler_name in scheduler_constructors:
        # Get the constructor function for the selected scheduler
        constructor = scheduler_constructors[scheduler_name]
        # Load the scheduler using the constructor
        pipeline.scheduler = constructor("notsk007/" + scheduler_name)
        return f"Successfully updated scheduler to {scheduler_name}. Scheduler value: {pipeline.scheduler}"
    else:
        return f"Scheduler '{scheduler_name}' not found."


# Define a dictionary to map scheduler names to their constructors
scheduler_constructors = {
    "PNDM": PNDMScheduler.from_pretrained,
    "DEIS": DEISMultistepScheduler.from_pretrained,
    "UniPC": UniPCMultistepScheduler.from_pretrained,
    "Euler": EulerDiscreteScheduler.from_pretrained,
    "Euler A": EulerAncestralDiscreteScheduler.from_pretrained,
    "LMS": LMSDiscreteScheduler.from_pretrained,
    "LMS-Karras": LMSDiscreteScheduler.from_pretrained,
    "DPM2": KDPM2DiscreteScheduler.from_pretrained,
    "DPM2-Karras": KDPM2DiscreteScheduler.from_pretrained,
    "DPM2-A": KDPM2AncestralDiscreteScheduler.from_pretrained,
    "DPM2-A-Karras": KDPM2AncestralDiscreteScheduler.from_pretrained,
    "DPM-SDE": DPMSolverSinglestepScheduler.from_pretrained,
    "DPM-SDE-Karras": DPMSolverSinglestepScheduler.from_pretrained,
    "DPM-2M": DPMSolverMultistepScheduler.from_pretrained,
    "DPM-2M-Karras": DPMSolverMultistepScheduler.from_pretrained,
    "DPM-2M-SDE": DPMSolverMultistepScheduler.from_pretrained,
    "DPM-2M-SDE-Karras": DPMSolverMultistepScheduler.from_pretrained,
}
