import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# --- 1. Load the Model ---

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Use torch.float16 to save memory (requires a GPU)
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# Move the pipeline to the GPU (cuda)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    print("Warning: CUDA (GPU) not available. Running on CPU, which will be very slow.")
    pipe = pipe.to("cpu")

# --- 2. Define AI Influencer Prompts ---
# A good prompt is highly detailed.

prompt = (
    """photorealistic medium shot, 28-year-old male fashion influencer,
      walking on SoHo cobblestone street.
      Black leather jacket, white t-shirt, dark jeans.
      Natural morning light, soft shadows.
Shot on 85mm f/1.8 lens, shallow depth of field, natural bokeh, 8k, cinematic."""
)

# Negative prompts are just as important. They tell the model what to AVOID.
negative_prompt = (
    """
    deformed, disfigured, blurry, low quality, pixelated
ugly, bad anatomy, bad hands, mutated hands
extra fingers, extra limbs, malformed
grainy, noisy, cartoon, 3d, painting
drawing, illustration, watermark, text, signature
    """
)

# --- 3. Generate the Image ---
print("Generating image... This may take a moment.")

# We use a generator for reproducible results. Change the seed for a different image.
generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,       # Number of steps (30-40 is good)
    guidance_scale=7.5,           # How much to follow the prompt (7-8 is standard)
    generator=generator
).images[0]

# --- 4. Save the Image ---
output_filename = "ai_influencer.png"
image.save(output_filename)

print(f"Image saved as {output_filename}")