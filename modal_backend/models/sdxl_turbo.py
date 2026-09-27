import modal

app = modal.App("sdxl-turbo-api")

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
    gpu="A10G",
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
)
class SDXLTurboModel:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import AutoPipelineForText2Image
        
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")

    @modal.method()
    def generate(self, prompt: str, seed: int = -1, guidance_scale: float = 0.0, num_inference_steps: int = 1, width: int = 512, height: int = 512, batch_size: int = 1):
        import torch
        from io import BytesIO
        
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        results = []
        SAFE_BATCH_SIZE = 4
        
        for i in range(0, batch_size, SAFE_BATCH_SIZE):
            current_chunk_size = min(SAFE_BATCH_SIZE, batch_size - i)
            generator = torch.Generator("cuda").manual_seed(seed + i)
            
            # SDXL Turbo is fixed at 512x512 usually, but we allow pass-through
            chunk_images = self.pipe(
                prompt=[prompt] * current_chunk_size,
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
