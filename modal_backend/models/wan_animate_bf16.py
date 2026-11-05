import modal
import os

# Wan 2.2 Animate 14B - BF16 full-quality variant.
# Weights live on the wan-animate-model Modal Volume (ideogram-like).
# Deploy: modal deploy modal_backend/models/wan_animate_bf16.py

app = modal.App("wan-animate-bf16-api")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
wan_volume = modal.Volume.from_name("wan-animate-model", create_if_missing=True)

BF16_DIR = "/models/wan-animate/bf16"
HF_REPO = "Wan-AI/Wan2.2-Animate-14B-Diffusers"

image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "diffusers>=0.35.0",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
        "ftfy",
        "regex",
        "sentencepiece",
    )
    .env({"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
)


@app.function(
    image=image,
    volumes={"/models/wan-animate": wan_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def download_bf16_model():
    """One-time seeding: store BF16 weights on the volume."""
    from huggingface_hub import snapshot_download

    sentinel = os.path.join(BF16_DIR, "model_index.json")
    if os.path.exists(sentinel):
        return f"BF16 model already cached at {BF16_DIR}"
    os.makedirs(BF16_DIR, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=BF16_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        token=os.environ.get("HF_TOKEN"),
    )
    wan_volume.commit()
    return f"BF16 model cached at {BF16_DIR}"


@app.cls(
    gpu="H100",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/models/wan-animate": wan_volume,
    },
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=3600,
    startup_timeout=1200,
)
class WanAnimateBF16:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import WanAnimatePipeline

        wan_volume.reload()
        local_index = os.path.join(BF16_DIR, "model_index.json")
        src = BF16_DIR if os.path.exists(local_index) else HF_REPO
        print(f"Loading Wan Animate BF16 from: {src}")
        token = os.environ.get("HF_TOKEN")
        self.pipe = WanAnimatePipeline.from_pretrained(
            src, torch_dtype=torch.bfloat16, token=token,
        )
        try:
            self.pipe.vae.to(torch.float32)
        except Exception as e:
            print(f"VAE upcast notice: {e}")
        self.pipe.to("cuda")
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as e:
            print(f"Backend flag notice: {e}")
        try:
            if not os.path.exists(local_index):
                print("Snapshotting first download to volume...")
                from huggingface_hub import snapshot_download
                os.makedirs(BF16_DIR, exist_ok=True)
                snapshot_download(
                    repo_id=HF_REPO, local_dir=BF16_DIR,
                    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
                    token=token,
                )
            wan_volume.commit()
            print("Volume commit ok.")
        except Exception as e:
            print(f"Volume commit notice: {e}")
        print("Wan Animate BF16 ready.")

    @modal.method()
    def generate(self, ref_b64: str, drive_b64: str, prompt: str = "",
                 neg: str = "", height: int = 720, width: int = 1280,
                 steps: int = 20, cfg: float = 1.0, seg: int = 77,
                 fps: int = 24, seed: int = -1) -> bytes:
        import base64, random, tempfile, torch
        from io import BytesIO
        from PIL import Image
        import imageio.v2 as imageio
        from diffusers.utils import export_to_video

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ref = Image.open(BytesIO(base64.b64decode(ref_b64))).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(base64.b64decode(drive_b64))
            dpath = f.name
        reader = imageio.get_reader(dpath, "ffmpeg")
        frames = [Image.fromarray(fr) for fr in reader]
        reader.close()
        if not frames:
            raise ValueError("No frames decoded from driving video (need .mp4).")
        # Fallback: raw driving frames stand in for pose+face videos.
        # For best quality pre-process with DWPose + face crops upstream.
        print(f"Animating {len(frames)} frames {height}x{width} "
              f"steps={steps} cfg={cfg} seed={seed}")
        with torch.inference_mode():
            out = self.pipe(
                image=ref, pose_video=frames, face_video=frames,
                prompt=prompt or "character animation, natural motion, high quality",
                negative_prompt=neg or None,
                height=int(height), width=int(width),
                segment_frame_length=int(seg),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg), mode="animate", generator=gen,
            )
            vframes = out.frames[0]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as o:
            opath = o.name
        export_to_video(vframes, opath, fps=int(fps))
        with open(opath, "rb") as f:
            return f.read()
