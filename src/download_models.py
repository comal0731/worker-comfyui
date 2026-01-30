import os
import subprocess
import sys

def download_file(url, output_path, token=None):
    """
    Downloads a file using wget.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = ["wget", "-q", "-O", output_path]
    if token:
        cmd.append(f"--header=Authorization: Bearer {token}")
    
    cmd.append(url)
    
    print(f"Downloading {url} to {output_path}...")
    try:
        subprocess.run(cmd, check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}: {e}")
        sys.exit(1)

def main():
    model_type = os.environ.get("MODEL_TYPE", "flux1-dev-fp8")
    token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

    print(f"Starting model download for type: {model_type}")

    # Define model configurations
    # Format: "model_type": [("output_path", "url", needs_token)]
    models = {
        "sdxl": [
            ("models/checkpoints/sd_xl_base_1.0.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", False),
            ("models/vae/sdxl_vae.safetensors", "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors", False),
            ("models/vae/sdxl-vae-fp16-fix.safetensors", "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors", False),
        ],
        "sd3": [
            ("models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors", "https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors", True),
        ],
        "flux1-schnell": [
            ("models/unet/flux1-schnell.safetensors", "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors", True),
            ("models/clip/clip_l.safetensors", "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors", False),
            ("models/clip/t5xxl_fp8_e4m3fn.safetensors", "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors", False),
            ("models/vae/ae.safetensors", "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors", True),
        ],
        "flux1-dev": [
            ("models/unet/flux1-dev.safetensors", "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors", True),
            ("models/clip/clip_l.safetensors", "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors", False),
            ("models/clip/t5xxl_fp8_e4m3fn.safetensors", "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors", False),
            ("models/vae/ae.safetensors", "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors", True),
        ],
        "flux1-dev-fp8": [
            ("models/checkpoints/flux1-dev-fp8.safetensors", "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors", False),
        ],
        "z-image-turbo": [
            ("models/text_encoders/qwen_3_4b.safetensors", "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors", False),
            ("models/diffusion_models/z_image_turbo_bf16.safetensors", "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors", False),
            ("models/vae/ae.safetensors", "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors", False),
            ("models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors", "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors", False),
        ]
    }

    if model_type not in models:
        print(f"Unknown MODEL_TYPE: {model_type}")
        print(f"Available types: {list(models.keys())}")
        # Proceed without downloading anything, or exit? 
        # Original Dockerfile didn't error, just did nothing for unknown types.
        # We'll mimic that behavior but print a warning.
        return

    for output_path, url, needs_token in models[model_type]:
        current_token = token if needs_token else None
        download_file(url, output_path, current_token)

if __name__ == "__main__":
    main()
