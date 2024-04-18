# /content/modules/pipeline.py
from diffusers import DiffusionPipeline

pipeline = None
def load_pipeline_global(model_id):
  global pipeline
  pipeline = DiffusionPipeline.from_pretrained(model_id).to('cuda')
  pipeline.safety_checker = None
  return pipeline
