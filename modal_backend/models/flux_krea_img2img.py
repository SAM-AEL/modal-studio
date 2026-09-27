import modal
import os

app = modal.App("flux-krea-img2img-api")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "huggingface_hub"
    )
    .env({"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
)

@app.cls(
    gpu="A100",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=90,
)
class FluxImg2ImgModel:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import FluxImg2ImgPipeline
        print("Loading Flux Krea Image-to-Image model...")
        token = os.environ.get("HF_TOKEN")
        
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Krea-dev",
            torch_dtype=torch.bfloat16,
            token=token
        ).to("cuda")
        
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        print("Model loaded.")

    @modal.method()
    def generate(self, image_base64: str, prompt: str, strength: float = 0.6, seed: int = -1, guidance_scale: float = 3.5, num_inference_steps: int = 28, width: int = 1024, height: int = 1024, batch_size: int = 1):
        import torch
        import base64
        from io import BytesIO
        from PIL import Image
        
        print(f"Requesting {batch_size} image-to-image streams. Strength: {strength}")
        
        # 1. Decode base64 image
        try:
            image_bytes = base64.b64decode(image_base64)
            init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            # Resize for compatibility if needed, though Flux might handle it
            init_image = init_image.resize((width, height), Image.LANCZOS)
        except Exception as e:
             raise ValueError(f"Failed to decode base64 image: {e}")

        # 2. Seeding
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        results = []
        SAFE_BATCH_SIZE = 2 
        
        for i in range(0, batch_size, SAFE_BATCH_SIZE):
            current_chunk_size = min(SAFE_BATCH_SIZE, batch_size - i)
            print(f"Processing chunk: {i//SAFE_BATCH_SIZE + 1} ({current_chunk_size} images)")
            
            generator = torch.Generator("cuda").manual_seed(seed + i)
            
            chunk_images = self.pipe(
                prompt=[prompt] * current_chunk_size,
                image=[init_image] * current_chunk_size,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images

            for img in chunk_images:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                results.append(buffer.getvalue())
            
            torch.cuda.empty_cache()
            
        return results
