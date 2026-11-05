import modal
import os

# Wan 2.2 Animate 14B - FP8 quantized variant (Hopper-only e4m3fn).
# Transformer loaded in FP8 from okabeee/Wan2.2_Animate_14B_FP8 weights,
# rest (VAE/text/image encoders) stay BF16 from the official repo.
# Weights live on the wan-animate-model Modal Volume (ideogram-like).
# Deploy: modal deploy modal_backend/models/wan_animate_fp8.py

app = modal.App("wan-animate-fp8-api")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
wan_volume = modal.Volume.from_name("wan-animate-model", create_if_missing=True)

FP8_DIR = "/models/wan-animate/fp8"
BASE_REPO = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
FP8_FILE = "Wan2.2_Animate_14B_FP8.safetensors"
FP8_REPO = "okabeee/Wan2.2_Animate_14B_FP8"

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
def download_fp8_model():
    """One-time seeding: base repo + FP8 transformer on the volume."""
    from huggingface_hub import snapshot_download, hf_hub_download

    os.makedirs(FP8_DIR, exist_ok=True)
    base_dir = os.path.join(FP8_DIR, "base")
    if not os.path.exists(os.path.join(base_dir, "model_index.json")):
        snapshot_download(
            repo_id=BASE_REPO, local_dir=base_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*",
                             "transformer/*"],
            token=os.environ.get("HF_TOKEN"),
        )
    fp8_path = os.path.join(FP8_DIR, FP8_FILE)
    if not os.path.exists(fp8_path):
        import shutil
        tmp = hf_hub_download(
            repo_id=FP8_REPO, filename=FP8_FILE,
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copy(tmp, fp8_path)
    wan_volume.commit()
    return f"FP8 files cached at {FP8_DIR}"


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
class WanAnimateFP8:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import WanAnimatePipeline, WanAnimateTransformer3DModel

        # FP8 e4m3fn needs Hopper+ (H100). Fail fast elsewhere.
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            print("WARNING: FP8 variant expects H100/Hopper; continuing anyway.")

        wan_volume.reload()
        base_dir = os.path.join(FP8_DIR, "base")
        fp8_path = os.path.join(FP8_DIR, FP8_FILE)
        if os.path.exists(os.path.join(base_dir, "model_index.json")):
            base_src = base_dir
        else:
            base_src = BASE_REPO
        print(f"Loading Wan Animate FP8: base={base_src} fp8={fp8_path}")
        token = os.environ.get("HF_TOKEN")

        # Load transformer in FP8: from single-file FP8 dump when cached,
        # else from the FP8 HF repo directly.
        if os.path.exists(fp8_path):
            transformer = WanAnimateTransformer3DModel.from_single_file(
                fp8_path, torch_dtype=torch.float8_e4m3fn,
            )
        else:
            transformer = WanAnimateTransformer3DModel.from_pretrained(
                FP8_REPO, torch_dtype=torch.float8_e4m3fn, token=token,
            )
        self.pipe = WanAnimatePipeline.from_pretrained(
            base_src, transformer=transformer,
            torch_dtype=torch.bfloat16, token=token,
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
            wan_volume.commit()
        except Exception as e:
            print(f"Volume commit notice: {e}")
        print("Wan Animate FP8 ready.")

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
        print(f"FP8 animating {len(frames)} frames {height}x{width} "
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
