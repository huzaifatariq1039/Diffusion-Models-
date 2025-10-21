import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# --- 1. Configuration ---

model_id = "SG161222/RealVisXL_V4.0"

# Check for GPU (cuda) and set the data type
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")

# --- 2. Load the RealVisXL Pipeline ---
# RealVisXL is based on SDXL, so we use StableDiffusionXLPipeline.
print(f"Loading RealVisXL model: {model_id}...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    use_safetensors=True
    # variant="fp16" # Usually not needed if torch_dtype is float16
)
pipe = pipe.to(device)
print("Model loaded successfully.")

# --- 3. Define Prompts ---
# RealVisXL excels at realistic human portraits and scenes.

prompt = (
    "photorealistic portrait of a young woman with freckles, "
    "natural light, soft smile, looking slightly away from camera. "
    "Shallow depth of field, detailed skin texture, Canon EOS R5 photo, 50mm lens, f/1.8."
)

# A standard negative prompt
negative_prompt = (
    "ugly, deformed, disfigured, blurry, low quality, pixelated, noisy, "
    "bad anatomy, bad hands, mutated hands, extra fingers, extra limbs, "
    "cartoon, 3d, painting, drawing, illustration, sketch, watermark, text, signature"
)

# --- 4. Generate the Image ---
print("Generating image...")

# Use a generator for reproducible results
generator = torch.Generator(device=device).manual_seed(5678)

# Standard SDXL parameters work well
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.0,
    generator=generator,
    width=1024, # SDXL native resolution
    height=1024 # SDXL native resolution
).images[0]

# --- 5. Save the Image ---
output_filename = "realvisxl_output.png"
image.save(output_filename)

print(f"Image saved successfully as {output_filename}")