#!/usr/bin/env python3
"""
Run Tree-Ring Watermarking with Stable Diffusion (latent diffusion).
Uses Hugging Face diffusers. Install: pip install diffusers transformers accelerate torch
"""

import argparse
import sys
from pathlib import Path

# Add parent so we can import tree_ring
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_watermarking.tree_ring import (
    inject_watermark_noise_latent,
    detect_tree_ring,
)


def main():
    parser = argparse.ArgumentParser(description="Tree-Ring watermark: generate and detect")
    parser.add_argument("--mode", choices=["generate", "detect", "both"], default="both")
    parser.add_argument("--key", choices=["zeros", "rand", "rings"], default="rings")
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="A photo of an astronaut riding a horse on Mars")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="outputs_tree_ring")
    args = parser.parse_args()

    try:
        import torch
        from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
        from diffusers.models.autoencoding_kl import AutoencoderKLOutput
    except ImportError as e:
        print("Install: pip install diffusers transformers accelerate torch")
        raise e

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Latent shape for SD: (1, 4, 64, 64)
    latent_shape = (1, 4, 64, 64)
    h, w = 64, 64

    if args.mode in ("generate", "both"):
        # 1) Create watermarked initial noise in latent space
        latents_watermarked = inject_watermark_noise_latent(
            (4, h, w),
            key_type=args.key,
            radius=args.radius,
            seed=args.seed,
        )
        latents_watermarked = torch.from_numpy(latents_watermarked).unsqueeze(0).to(device).to(pipe.unet.dtype)
        latents_watermarked = latents_watermarked * pipe.scheduler.init_noise_sigma

        # 2) Generate image with DDIM
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        out = pipe(
            prompt=args.prompt,
            latents=latents_watermarked,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            generator=generator,
        )
        image = out.images[0]
        path_img = out_dir / "watermarked.png"
        image.save(path_img)
        print("Saved", path_img)

        # Also save non-watermarked for comparison
        latents_clean = torch.randn(latent_shape, device=device, dtype=pipe.unet.dtype, generator=generator)
        latents_clean = latents_clean * pipe.scheduler.init_noise_sigma
        out_clean = pipe(
            prompt=args.prompt,
            latents=latents_clean,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            generator=torch.Generator(device=device).manual_seed(args.seed + 2),
        )
        out_clean.images[0].save(out_dir / "clean.png")
        print("Saved", out_dir / "clean.png")

    if args.mode in ("detect", "both"):
        # Load the generated image and run DDIM inversion to get latent noise
        from PIL import Image
        from diffusers import DDIMInverseScheduler

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

        image_path = out_dir / "watermarked.png"
        if not image_path.exists():
            print("Run with --mode generate first to create watermarked.png")
            return
        pil_image = Image.open(image_path).convert("RGB")

        # Encode image to latent
        with torch.no_grad():
            latent = pipe.vae.encode(
                pipe.image_processor.preprocess(pil_image).to(device).to(pipe.vae.dtype).unsqueeze(0) / 0.18215
            ).latent_dist.sample()
        # DDIM inversion: latent x0 -> xT
        inv_scheduler = pipe.inverse_scheduler
        inv_scheduler.set_timesteps(args.steps)
        timesteps = inv_scheduler.timesteps
        latent_inv = latent.clone()
        for t in timesteps:
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            noise_pred = pipe.unet(latent_inv, t_batch, encoder_hidden_states=pipe.encode_prompt("")).sample
            latent_inv = inv_scheduler.step(noise_pred, t, latent_inv).prev_sample

        inverted_noise = latent_inv.cpu().float().numpy()[0]  # (4, 64, 64)
        result = detect_tree_ring(
            inverted_noise,
            key_type=args.key,
            radius=args.radius,
            seed=args.seed,
            return_p_value=True,
        )
        print("Detection result:", result)
        if result["is_watermarked"]:
            print("-> Image is WATERMARKED (p = {:.2e})".format(result["p_value"]))
        else:
            print("-> Image is not watermarked or attack altered the signal (p = {:.2e})".format(result["p_value"]))

    print("Done.")


if __name__ == "__main__":
    main()
