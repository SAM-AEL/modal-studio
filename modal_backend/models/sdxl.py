import modal

app = modal.App("sdxl-api")

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
)

@app.cls(
    gpu="A10G", # SDXL fits easily in 24GB
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
)
class SDXLModel:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import DiffusionPipeline
        
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")
        
        self.pipe.enable_model_cpu_offload()

    @modal.method()
    def generate(self, prompt: str, seed: int = -1, guidance_scale: float = 7.5, num_inference_steps: int = 30, width: int = 1024, height: int = 1024, batch_size: int = 1):
        import torch
        from io import BytesIO
        
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        results = []
        SAFE_BATCH_SIZE = 4 # SDXL is lighter, can handle bigger safe chunks
        
        for i in range(0, batch_size, SAFE_BATCH_SIZE):
            current_chunk_size = min(SAFE_BATCH_SIZE, batch_size - i)
            generator = torch.Generator("cuda").manual_seed(seed + i)
            
            chunk_images = self.pipe(
                [prompt] * current_chunk_size,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
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
