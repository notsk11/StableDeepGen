#@title main.py
import gradio as gr
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, DPMSolverMultistepScheduler
from modules.txt2img import load_diffusion_pipeline, generate_images
from modules import style
from modules.style import css
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

modelnames = [
    ("digiplay/Realisian_v5", "digiplay/Realisian_v5"),
    ("SG161222/Realistic_Vision_V6.0_B1_noVAE", "SG161222/Realistic_Vision_V6.0_B1_noVAE"),
    ("segmind/SSD-1B", "segmind/SSD-1B"),
    ("digiplay/RealEpicMajicRevolution_v1", "digiplay/RealEpicMajicRevolution_v1"),
    ("imagepipeline/Realities-Edge-XL", "imagepipeline/Realities-Edge-XL"),
    ("stablediffusionapi/realisian111", "stablediffusionapi/realisian111")
]

pipeline = None  # Initialize pipeline globally

def load_model(models_id):
    global pipeline
    pipeline = load_diffusion_pipeline(models_id)
    pipeline.safety_checker = None

image_info_textbox = gr.Textbox(label="Image Info")  # Create the Textbox component
textbox_holder = [image_info_textbox]

def create_callback():
    def generate_images_callback(prompt_str, neg_prompt_str, height_int, width_int, num_steps_int, guid_scale_float, num_images_int, seed_int, scheduler_choice):
        global pipeline
        images, seed, scheduler = generate_images(pipeline, prompt_str, neg_prompt_str, height_int, width_int, num_steps_int, guid_scale_float, num_images_int, seed_int, scheduler_choice)
        image_data = {
            "prompt": prompt_str,
            "negative_prompt": neg_prompt_str,
            "seed": seed,
            "scheduler": pipeline.scheduler
        }
        image_info_textbox = update_image_info(textbox_holder[0], image_data)  # Access the textbox from the list
        textbox_holder[0] = image_info_textbox  # Update the textbox in the list
        return images, textbox_holder[0]  # Return the new Textbox
    return generate_images_callback


def load_model_callback(models_id):
    load_model(models_id)

def update_image_info(image_info_textbox, image_data):
    # Convert image_data to string
    image_data_str = str(image_data)
    new_textbox = gr.Textbox(value=image_data_str)  # Create a new Textbox
    return new_textbox


with gr.Blocks(css=style.css) as demo:
  gr.Markdown("Stable DeepGen v0.1", elem_classes="version")
  image_output_block = gr.Gallery(elem_classes="image_out")
  load_model_button = gr.Button("Load Model", elem_classes="load-button")
  load_model_button.click(load_model_callback, inputs=[gr.Dropdown(container=False, choices=modelnames, elem_classes="model", value="stablediffusionapi/realisian111")])
  gr.Markdown("Select Model", elem_classes="model-mark")
  schedulers = gr.Dropdown(container=False, choices=scheduler_choices, elem_classes="scheduler")
  gr.Markdown("Scheduler", elem_classes="scheduler-mark")
  styles = gr.CheckboxGroup(container=False, label="Styles", choices=["Anime", "3d", "Realistic"], elem_classes="style")
  gr.Markdown("Styles", elem_classes="style-mark")
  image_info_textbox = gr.Textbox(placeholder="Output Details", container=False, lines=9, elem_classes="details")
  with gr.Tab("Txt2Img", elem_classes="tab"):
    with gr.Row():
      with gr.Column():
        prompt_str = gr.Textbox(placeholder="Prompt", container=False, lines=3, elem_classes="prompt")
        neg_prompt_str = gr.Textbox(placeholder="Negative Prompt", container=False, lines=3, elem_classes="prompt")
      with gr.Column():
        gr.Markdown("Height", elem_classes="height-mark")
        height_int = gr.Slider(minimum=408, maximum=1600, step=8, value=408, container=False, elem_classes="height")
        gr.Markdown("Width", elem_classes="width-mark")
        width_int = gr.Slider(minimum=408, maximum=1600, step=8, value=408, container=False, elem_classes="width")
      with gr.Column():
        num_steps_int = gr.Slider(minimum=1, maximum=100, value=10, step=1, container=False, elem_classes="num_steps")
        gr.Markdown("Sampling Steps", elem_classes="num-steps-mark")
        guid_scale_float = gr.Slider(minimum=1, maximum=10, value=5, step=0.1, container=False, elem_classes="guidance")
        gr.Markdown("CFG Scale", elem_classes="guidance-mark")
      with gr.Column():
        num_images_int = gr.Slider(minimum=1, maximum=10, value=1, step=1, container=False, elem_classes="num_images")
        gr.Markdown("Batch Count", elem_classes="num-images-mark")
      with gr.Row():
        seed_inp = gr.Textbox(placeholder="Seed", container=False, elem_classes="seed-inp")
        generate = gr.Button(elem_classes="generate")
  with gr.Tab("Img2Img", elem_classes=""):
    with gr.Row():
      text = gr.Textbox()
  with gr.Tab("Txt2Vid", elem_classes=""):
    with gr.Row():
      text = gr.Textbox()
  with gr.Tab("Upscale", elem_classes=""):
    with gr.Row():
      text = gr.Textbox()
  with gr.Tab("Settings", elem_classes="Settings"):
    with gr.Row():
      text = gr.Textbox()
  callback = create_callback()  # Create the callback
  generate.click(fn=callback, inputs=[prompt_str, neg_prompt_str, height_int, width_int, num_steps_int, guid_scale_float, num_images_int, seed_inp, schedulers], outputs=[image_output_block, image_info_textbox])

demo.launch(share=True, debug=True)
