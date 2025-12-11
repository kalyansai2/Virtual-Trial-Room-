import argparse
import os
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image


def load_binary_mask(mask_path):
    """Convert VITON-HD cloth-mask to a binary inpainting mask."""
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)

    # VITON-HD cloth-mask convention: shirt = pixel value 1
    binary = (arr == 1).astype(np.uint8) * 255  # white = edit, black = keep

    return Image.fromarray(binary, mode="L")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--mask_image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=12.0)
    parser.add_argument("--strength", type=float, default=0.9)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SDXL inpainting model (safetensors only)...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    print("Model loaded.")

    # Load inputs
    image = load_image(args.input_image).convert("RGB")
    mask = load_binary_mask(args.mask_image)

    print("Mask loaded. Unique mask values:", np.unique(np.array(mask)))

    # Stronger recoloring prompt
    prompt = (
        args.prompt
        + ", preserve original shirt texture and shape, no artifacts, no extra pockets, no beard, keep face unchanged"
    )

    print("Running Stable Diffusion...")
    output = pipe(
        prompt=args.prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
    ).images[0]
    
    import re

    # Extract numbers from the input filename (e.g., "00013_00.jpg" -> ["00013", "00"])
    input_name = os.path.basename(args.input_image)
    numbers = re.findall(r"\d+", input_name)

    #  Choose the first number block (e.g., "00013") and convert to int → 13
    if numbers:
        file_id = int(numbers[0])
    else:
        file_id = "unknown"

    # Build output filename automatically
    out_filename = f"recolored_{file_id}.png"
    out_path = os.path.join(args.output_dir, out_filename)

    # Save
    output.save(out_path)
    print(f"Saved recolored result to: {out_path}")



if __name__ == "__main__":
    main()

