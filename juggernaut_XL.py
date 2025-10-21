import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# --- 1. Configuration ---

model_id = "RunDiffusion/Juggernaut-XL-v9"

# Check for GPU (cuda) and set the data type for optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")

# --- 2. Load the Juggernaut XL Pipeline ---
# The 'variant="fp16"' and 'torch_dtype' are used for memory optimization on GPUs.
# Juggernaut's VAE is already baked in, so no separate VAE is needed.
print("Loading Juggernaut XL model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    variant="fp16",
    use_safetensors=True
)
pipe = pipe.to(device)
print("Model loaded successfully.")

# --- 3. Define Prompts ---
# Juggernaut responds well to detailed, cinematic prompts.
# You can also start with no negative prompt, as recommended by the creator.

prompt = (
    "cinematic portrait of Batman, brooding on a rainy Gotham rooftop. "
    "Tactical batsuit, intense expression. "
    "Dramatic shadows, 8k, hyper-detailed, dark, gritty."
)

# A standard negative prompt to avoid common issues
negative_prompt = (
    "deformed, disfigured, blurry, low quality, pixelated, ugly, "
    "bad anatomy, bad hands, mutated hands, extra fingers, "
    "cartoon, 3d, painting, drawing, illustration, watermark, text"
)

# --- 4. Generate the Image ---
print("Generating image...")

# Use a generator for reproducible results. Change the seed for a new image.
generator = torch.Generator(device=device).manual_seed(1234)

# Juggernaut works well with 30-40 steps and a lower CFG scale (3-7)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=35,
    guidance_scale=5.5,
    generator=generator,
    width=832,  # Recommended resolution for Juggernaut
    height=1216 # Recommended resolution for Juggernaut
).images[0]

# --- 5. Save the Image ---
output_filename = "juggernaut_output.png"
image.save(output_filename)

print(f"Image saved successfully as {output_filename}")