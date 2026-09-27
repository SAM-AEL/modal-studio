import modal

app = modal.App("flux-schnell-api")

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
    gpu="A100", # Schnell is fast but big, A100 keeps it instant
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
)
class FluxSchnell:

    @modal.enter()
    def load(self):
        import torch
        import os
        from diffusers import FluxPipeline
        token = os.environ["HF_TOKEN"]
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            token=token
        ).to("cuda")
        
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

    @modal.method()
    def generate(self, prompt: str, seed: int = -1, guidance_scale: float = 0.0, num_inference_steps: int = 4, width: int = 1024, height: int = 1024, batch_size: int = 1):
        import torch
        from io import BytesIO
        
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        results = []
        SAFE_BATCH_SIZE = 2 
        
        for i in range(0, batch_size, SAFE_BATCH_SIZE):
            current_chunk_size = min(SAFE_BATCH_SIZE, batch_size - i)
            generator = torch.Generator("cuda").manual_seed(seed + i)
            
            chunk_images = self.pipe(
                [prompt] * current_chunk_size,
                guidance_scale=guidance_scale, # Schnell usually uses 0.0
                num_inference_steps=num_inference_steps, # Schnell is designed for 4 steps
                width=width,
                height=height,
                generator=generator
            ).images

            for img in chunk_images:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                results.append(buffer.getvalue())
            
            torch.cuda.empty_cache()
            
        return results
