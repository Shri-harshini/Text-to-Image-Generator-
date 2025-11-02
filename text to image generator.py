# 🖼️ Text-to-Image Generator using Gradio + Stable Diffusion
# Works directly in Google Colab

!pip install -q gradio diffusers transformers accelerate torch

import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion Model (may take a few seconds)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate image from text
def text_to_image(prompt):
    if not prompt:
        return "Please type something like 'a rat on a table'."
    image = pipe(prompt).images[0]
    return image

# Gradio Interface
with gr.Blocks(title="Text to Image Generator") as demo:
    gr.Markdown("# 🎨 Text-to-Image Generator using AI")
    gr.Markdown("Type any text below (e.g., **rat**, **cat wearing sunglasses**, **mountain sunrise**) to generate an image.")
    prompt = gr.Textbox(label="Enter text here", placeholder="Type something like 'rat'")
    btn = gr.Button("Generate Image")
    output = gr.Image(label="Generated Image")
    btn.click(fn=text_to_image, inputs=prompt, outputs=output)

# Launch the app
demo.launch(share=True)