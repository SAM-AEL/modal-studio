import modal
import os

app = modal.App("flux-lora-api")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("lora-outputs", create_if_missing=True)
model_volume = modal.Volume.from_name("flux-model", create_if_missing=False)
krea_model_volume = modal.Volume.from_name("flux-krea-model", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "Pillow",
        "huggingface_hub",
        "peft" # Required for LoRA management
    )
    .env({"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
)

@app.cls(
    gpu="a100",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/outputs": output_volume,
        "/flux-model": model_volume,
        "/flux-krea-model": krea_model_volume
    },
    min_containers=0,
    max_containers=1,
    scaledown_window=90,
)
class FluxLoraModel:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import FluxPipeline
        print("Loading Flux base model from volume...")
        
        # Look for Krea in volume first
        model_path = "/flux-krea-model"
        
        if not os.path.exists(os.path.join(model_path, "model_index.json")):
             print(f"Warning: /flux-krea-model empty or incomplete. Trying original flux-model volume as fallback...")
             model_path = "/flux-model/FLUX.1-dev"

        if not os.path.exists(model_path):
             print(f"Warning: neither volume has model, falling back to HF download...")
             model_path = "black-forest-labs/FLUX.1-Krea-dev"

        token = os.environ.get("HF_TOKEN")
        self.pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            token=token
        ).to("cuda")
        
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        print("Model loaded.")

    @modal.method()
    def generate(self, prompt: str, seed: int = -1, guidance_scale: float = 4.5, num_inference_steps: int = 20, width: int = 1024, height: int = 1024, batch_size: int = 1, loras: list = []):
        import torch
        from io import BytesIO
        
        # Reload volume to see latest training outputs
        output_volume.reload()

        print(f"Requesting {batch_size} image(s). Loras: {loras}")
        
        # 1. Manage LoRAs
        # Unload existing to avoid carry-over
        try:
            self.pipe.unload_lora_weights()
            print("Unloaded previous LoRAs.")
        except Exception as e:
            print(f"Note: Unload LoRAs skipped ({e})")

        adapter_names = []
        adapter_weights = []

        for i, lora in enumerate(loras):
            name = lora.get("name")
            weight = lora.get("weight", 1.0)
            if not name:
                continue
                
            path = f"/outputs/{name}/{name}_lora.safetensors"
            print(f"Checking LoRA at {path}...")
            
            if os.path.exists(path):
                adapter_name = f"lora_{name}_{i}"
                print(f"Loading LoRA: {name} with weight {weight}")
                try:
                    self.pipe.load_lora_weights(path, adapter_name=adapter_name)
                    adapter_names.append(adapter_name)
                    adapter_weights.append(weight)
                except Exception as e:
                    print(f"Error loading LoRA {name}: {e}")
            else:
                print(f"LoRA path not found: {path}")

        if adapter_names:
            print(f"Setting adapters: {adapter_names} with weights: {adapter_weights}")
            self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        else:
            print("No LoRAs applied.")

        # 2. Seeding
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        results = []
        # Batching logic similar to flux_krea.py
        SAFE_BATCH_SIZE = 2 
        
        for i in range(0, batch_size, SAFE_BATCH_SIZE):
            current_chunk_size = min(SAFE_BATCH_SIZE, batch_size - i)
            print(f"Processing chunk: {i//SAFE_BATCH_SIZE + 1} ({current_chunk_size} images)")
            
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


@app.cls(volumes={"/outputs": output_volume})
class LoraManager:
    @modal.method()
    def list_loras(self) -> list:
        print("Listing LoRAs from /outputs...")
        output_volume.reload()
        if not os.path.exists("/outputs"):
            return []
        
        loras = []
        for d in os.listdir("/outputs"):
            if os.path.isdir(f"/outputs/{d}"):
                safe_path = f"/outputs/{d}/{d}_lora.safetensors"
                if os.path.exists(safe_path):
                    loras.append(d)
        print(f"Found LoRAs: {loras}")
        return loras


@app.function(
    image=image,
    volumes={"/flux-krea-model": krea_model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600
)
def download_krea_model():
    from huggingface_hub import snapshot_download
    local_dir = "/flux-krea-model"
    sentinel = os.path.join(local_dir, "model_index.json")
    if os.path.exists(sentinel):
        print("Krea model already downloaded to volume.")
        return
    print("Downloading Krea model to /flux-krea-model volume...")
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-Krea-dev",
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        token=os.environ.get("HF_TOKEN"),
    )
    krea_model_volume.commit()
    print("Done downloading Krea model to volume.")


@app.local_entrypoint()
def main(download: bool = False):
    if download:
        print("Starting download krea model task...")
        download_krea_model.remote()
        return
        
    print("--- Testing list_loras() ---")
    try:
        manager = LoraManager()
        loras = manager.list_loras.remote()
        print(f"Result: {loras}")
    except Exception as e:
        print(f"Error: {e}")
