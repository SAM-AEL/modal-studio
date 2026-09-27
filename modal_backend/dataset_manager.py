import modal
import os
import io
import zipfile

app = modal.App("lora-dataset-manager")
dataset_volume = modal.Volume.from_name("lora-datasets", create_if_missing=True)
model_volume   = modal.Volume.from_name("flux-model", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub")
)

@app.cls(
    image=image,
    volumes={
        "/datasets": dataset_volume,
        "/flux-model": model_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=1,
    timeout=7200,
)
class DatasetManager:
    @modal.method()
    def save_image(self, dataset_name: str, filename: str, image_bytes: bytes, caption: str):
        dataset_path = f"/datasets/{dataset_name}"
        os.makedirs(dataset_path, exist_ok=True)
        
        # Save image
        with open(f"{dataset_path}/{filename}.png", "wb") as f:
            f.write(image_bytes)
            
        # Save caption
        with open(f"{dataset_path}/{filename}.txt", "w") as f:
            f.write(caption)
            
        dataset_volume.commit()
        return f"Saved to {dataset_name}/{filename}"

    @modal.method()
    def save_images_batch(self, dataset_name: str, files: list):
        dataset_path = f"/datasets/{dataset_name}"
        os.makedirs(dataset_path, exist_ok=True)
        
        for f in files:
            name = f["name"]
            image_bytes = f["image_bytes"]
            caption = f["caption"]
            
            with open(f"{dataset_path}/{name}.png", "wb") as img_f:
                img_f.write(image_bytes)
            with open(f"{dataset_path}/{name}.txt", "w") as cap_f:
                cap_f.write(caption)
                
        dataset_volume.commit()
        return f"Saved {len(files)} files to {dataset_name}"

    @modal.method()
    def list_datasets(self):
        if not os.path.exists("/datasets"):
            return []
        return [d for d in os.listdir("/datasets") if os.path.isdir(os.path.join("/datasets", d))]

    @modal.method()
    def get_dataset_info(self, dataset_name: str):
        path = f"/datasets/{dataset_name}"
        if not os.path.exists(path):
            return []
        
        files = []
        for f in os.listdir(path):
            if f.endswith(".png"):
                base = f.rsplit(".", 1)[0]
                caption_path = f"{path}/{base}.txt"
                caption = ""
                if os.path.exists(caption_path):
                    with open(caption_path, "r") as cf:
                        caption = cf.read()
                files.append({"name": base, "caption": caption})
        return files

    @modal.method()
    def rename_all_to_sequential(self, dataset_name: str):
        path = f"/datasets/{dataset_name}"
        if not os.path.exists(path):
            raise Exception("Dataset not found")
            
        files = [f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        files.sort()
        
        for i, f in enumerate(files):
            ext = f.rsplit(".", 1)[1]
            new_name = f"{i+1:03d}.{ext}"
            old_path = os.path.join(path, f)
            new_path = os.path.join(path, new_name)
            
            # If caption exists, rename it too
            old_cap = os.path.join(path, f.rsplit(".", 1)[0] + ".txt")
            new_cap = os.path.join(path, f"{i+1:03d}.txt")
            
            os.rename(old_path, new_path)
            if os.path.exists(old_cap):
                os.rename(old_cap, new_cap)
                
        dataset_volume.commit()
        return f"Renamed {len(files)} files sequentially"

    @modal.method()
    def create_batch_captions(self, dataset_name: str, caption: str):
        path = f"/datasets/{dataset_name}"
        if not os.path.exists(path):
            raise Exception("Dataset not found")
            
        files = [f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for f in files:
            base = f.rsplit(".", 1)[0]
            with open(os.path.join(path, f"{base}.txt"), "w") as cf:
                cf.write(caption)
                
        dataset_volume.commit()
        return f"Created captions for {len(files)} images"

    @modal.method()
    def export_zip(self, dataset_name: str):
        path = f"/datasets/{dataset_name}"
        if not os.path.exists(path):
            raise Exception("Dataset not found")
            
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zip_file:
            for f in os.listdir(path):
                zip_file.write(os.path.join(path, f), f)
        
        return buffer.getvalue()

    @modal.method()
    def download_flux(self):
        """Download FLUX.1-dev to the flux-model volume."""
        from huggingface_hub import snapshot_download
        HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"
        local = "/flux-model/FLUX.1-dev"
        sentinel = os.path.join(local, "model_index.json")
        
        if os.path.exists(sentinel):
            return f"✓ Model already cached at {local}"
            
        print(f"Environment keys list: {[k for k in os.environ if 'HF' in k.upper() or 'HUGGING_FACE' in k.upper() or 'HUGGINGFACE' in k.upper()]}")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_CO_API_KEY") or os.environ.get("HUGGING_FACE_TOKEN")
        if not token:
            raise ValueError(f"No HF token found in environment. Checked: HF_TOKEN, HUGGINGFACE_TOKEN, HUGGINGFACE_CO_API_KEY. Found keys: {[k for k in os.environ if 'HF' in k.upper() or 'HUGGINGFACE' in k.upper()]}")
            
        print(f"Downloading {HF_MODEL_ID} from HuggingFace Hub...")
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=local,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
            token=token,
        )
        model_volume.commit()
        return f"✓ Model downloaded and cached: {local}"
