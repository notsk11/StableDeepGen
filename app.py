# /content/app.py

import gradio as gr
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
import torch
from modules import txt2img
from modules.txt2img import txt2img
from modules import pipeline
from modules.pipeline import load_pipeline_global
from modules import style
from modules.style import css
from modules.scheduler import update_scheduler  # Import update_scheduler from scheduler module
from modules.scheduler import scheduler_constructors
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


with gr.Blocks(css=style.css) as demo:
  gr.Markdown("Stable Diffusion Checkpoint", elem_classes="stable-diff")
  with gr.Row():
    model_global = gr.Textbox(elem_classes="model", container=False, value="SG161222/Realistic_Vision_V6.0_B1_noVAE")
    load_model_global = gr.Button(value="Load", elem_classes="load-model")
  with gr.Tab("Txt2Img", elem_classes="tab"):
    with gr.Row():
      with gr.Column():
        prompt_t2i = gr.Textbox(elem_classes="prompt", container=False, lines=3, placeholder="Place Promps here....")
        negative_prompt_t2i = gr.Textbox(elem_classes="negative-prompt", container=False, lines=3, placeholder="Place Negative Promps here....")
      with gr.Column():
        generate_t2i = gr.Button(value="Generate", elem_classes="generate-t2i")
    with gr.Tab("Generation", elem_classes="tab2"):
      with gr.Row():
        with gr.Column():
          gr.Markdown("Sampling Methods", elem_classes="samp-meth-mark")
          scheduler = gr.Dropdown(elem_classes="samp-meth", choices=list(scheduler_constructors.keys()), value="DPM-SDE-Karras", container=False)
        with gr.Column():
          load_scheduler_t2i = gr.Button(value="Load", elem_classes="load-scheduler-t2i")
        with gr.Column():
          gr.Markdown("Sampling Steps", elem_classes="samp-steps-mark")
          num_inference_steps_t2i = gr.Slider(elem_classes="samp-steps", minimum=1, maximum=100, value=10, step=1, container=False)
        with gr.Column():
          image_out_t2i = gr.Gallery(elem_classes="image-output-t2i")
      with gr.Row():
        with gr.Column():
          gr.Markdown("Height", elem_classes="height-t2i-mark")
          height_t2i = gr.Slider(elem_classes="height-t2i", minimum=100, maximum=1600, value=408, step=8, container=False)
          gr.Markdown("Width", elem_classes="width-t2i-mark")
          width_t2i = gr.Slider(elem_classes="width-t2i", minimum=100, maximum=1600, value=408, step=8, container=False)
        with gr.Column():
          gr.Markdown("Batch Count", elem_classes="batch-count-t2i-mark")
          batch_count_t2i = gr.Slider(elem_classes="batch-count-t2i", minimum=1, maximum=10, step=1, value=1, container=False)
          gr.Markdown("Batch Size", elem_classes="batch-size-t2i-mark")
          batch_size = gr.Slider(elem_classes="batch-size-t2i", minimum=1, maximum=10, value=1, step=1, container=False)
      with gr.Column():
        gr.Markdown("CFG Scale", elem_classes="guidance-scale-t2i-mark")
        guidance_scale_t2i = gr.Slider(elem_classes="guidance-scale-t2i", minimum=0, maximum=10, value=7.5, step=0.1, container=False)
        gr.Markdown("Seed", elem_classes="seed-input-t2i-mark")
        seed_input_t2i = gr.Textbox(elem_classes="seed-input-t2i", container=False)
        metadata_t2i = gr.Textbox(elem_classes="metadata-t2i", container=False, lines=3)
      load_model_global.click(fn=load_pipeline_global, inputs=[model_global])
      load_scheduler_t2i.click(fn=update_scheduler, inputs=[scheduler])
      generate_t2i.click(fn=txt2img, inputs=[prompt_t2i, negative_prompt_t2i, height_t2i, width_t2i, num_inference_steps_t2i, guidance_scale_t2i, batch_count_t2i, seed_input_t2i], outputs=[image_out_t2i, metadata_t2i])


demo.launch(share=True, debug=True)
