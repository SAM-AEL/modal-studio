import modal
import os

# Wan 2.2 Animate 14B BF16 -> FFmpeg -> BiRefNet -> aligned RGBA -> spritesheet.
# Weights live on volumes (ideogram-like): wan model on wan-animate-model,
# BiRefNet rides the hf-cache. Deploy:
#   modal deploy modal_backend/models/wan_animate_sprite.py

app = modal.App("wan-animate-sprite-api")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
wan_volume = modal.Volume.from_name("wan-animate-model", create_if_missing=True)

BF16_DIR = "/models/wan-animate/bf16"
HF_REPO = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
BIREFNET_REPO = "ZhengPeng7/BiRefNet"

image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers>=0.35.0",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "timm",
        "Pillow",
        "numpy",
        "scipy",
        "opencv-python-headless",
        "imageio",
        "imageio-ffmpeg",
        "ftfy",
        "regex",
        "sentencepiece",
    )
    .env({"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
)


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
class WanAnimateSprite:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import WanAnimatePipeline
        from transformers import AutoModelForImageSegmentation

        wan_volume.reload()
        src = BF16_DIR if os.path.exists(
            os.path.join(BF16_DIR, "model_index.json")) else HF_REPO
        print(f"[wan] loading {src}")
        token = os.environ.get("HF_TOKEN")
        self.pipe = WanAnimatePipeline.from_pretrained(
            src, torch_dtype=torch.bfloat16, token=token)
        try:
            self.pipe.vae.to(torch.float32)
        except Exception as e:
            print(f"[wan] vae note: {e}")
        self.pipe.to("cuda")
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        try:
            if src != BF16_DIR:
                from huggingface_hub import snapshot_download
                os.makedirs(BF16_DIR, exist_ok=True)
                snapshot_download(
                    repo_id=HF_REPO, local_dir=BF16_DIR,
                    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
                    token=token)
            wan_volume.commit()
        except Exception as e:
            print(f"[wan] volume note: {e}")

        print(f"[matt] loading {BIREFNET_REPO}")
        self.matt = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_REPO, trust_remote_code=True,
            token=token).to("cuda").eval()
        try:
            wan_volume.commit()
            hf_cache.commit()
        except Exception as e:
            print(f"[vol] commit note: {e}")
        print("[ready] wan+ birefnet loaded")
    def _matt_one(self, rgb):
        import torch
        import numpy as np
        from torchvision.transforms.functional import normalize
        inp = rgb.resize((1024, 1024))
        arr = np.asarray(inp).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(
            "cuda", dtype=torch.float32)
        t = normalize(t, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        with torch.inference_mode():
            out = self.matt(t)
            pred = out[-1] if isinstance(out, (list, tuple)) else out
            pred = pred.sigmoid().squeeze().float().cpu().numpy()
        m = (pred * 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(m).resize(rgb.size)
    @modal.method()
    def animate_sprite(self, ref_b64: str, drive_b64: str,
                       prompt: str = "", neg: str = "",
                       size: int = 720, steps: int = 20,
                       cfg: float = 1.0, fps: int = 12,
                       seed: int = -1, out_size: int = 512,
                       cols: int = 4, green_bg: bool = True,
                       fmt: str = "png") -> dict:
        import base64, random, tempfile, torch, math
        import numpy as np
        import cv2
        from io import BytesIO
        from PIL import Image
        import imageio.v2 as imageio
        from diffusers.utils import export_to_video
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        raw = Image.open(BytesIO(base64.b64decode(ref_b64))).convert("RGBA")
        bg = (0, 255, 0) if green_bg else (255, 255, 255)
        flat = Image.new("RGB", raw.size, bg)
        flat.paste(raw, mask=raw.split()[3])
        ref = flat.resize((size, size))
        with tempfile.NamedTemporaryFile(suffix=".mp4",
                                         delete=False) as f:
            f.write(base64.b64decode(drive_b64))
            dpath = f.name
        reader = imageio.get_reader(dpath, "ffmpeg")
        dframes = [Image.fromarray(fr) for fr in reader]
        reader.close()
        if not dframes:
            raise ValueError("Need a decodable driving .mp4")
        bp = (prompt.strip() if prompt
              else "game symbol character, lively idle motion")
        if green_bg:
            bp += ", on solid pure green background, centered, no shadow"
        print(f"[wan] {len(dframes)} drv size={size} steps={steps} seed={seed}")
        with torch.inference_mode():
            out = self.pipe(
                image=ref, pose_video=dframes, face_video=dframes,
                prompt=bp, negative_prompt=neg or None,
                height=size, width=size, segment_frame_length=77,
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                mode="animate", generator=gen)
            vframes = [np.asarray(im).copy() for im in out.frames[0]]
        print(f"[matt] {len(vframes)} frames")
        masks = [np.asarray(self._matt_one(Image.fromarray(fr)))
                 for fr in vframes]
        sm = []
        for i, m in enumerate(masks):
            a = masks[max(0, i - 1)]
            b = masks[min(len(masks) - 1, i + 1)]
            sm.append(np.median(np.stack([a, m, b]), axis=0).astype(np.uint8))
        rgba = []
        for fr, m in zip(vframes, sm):
            blur = cv2.GaussianBlur(m, (3, 3), 0)
            _, th = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
            er = cv2.erode(th, np.ones((3, 3), np.uint8), iterations=1)
            soft = cv2.GaussianBlur(er, (3, 3), 0).astype(np.float32) / 255.0
            rgb = np.clip(fr.astype(np.float32), 0, 255)
            a = (soft * 255).astype(np.uint8)
            rgba.append(Image.fromarray(
                np.dstack([rgb.astype(np.uint8), a])))
        boxes = []
        for im in rgba:
            a = np.asarray(im)[:, :, 3]
            ys, xs = np.where(a > 16)
            boxes.append(None if len(xs) == 0 else
                         (xs.min(), ys.min(), xs.max(), ys.max()))
        good = [b for b in boxes if b]
        if not good:
            raise ValueError("BiRefNet found no foreground")
        gx = sum((b[0] + b[2]) / 2 for b in good) / len(good)
        gy = sum((b[1] + b[3]) / 2 for b in good) / len(good)
        gw = max(b[2] - b[0] for b in good)
        gh = max(b[3] - b[1] for b in good)
        side = int(max(gw, gh) * 1.15) or size
        aligned = []
        for im in rgba:
            w, h = im.size
            x0 = int(max(0, min(w - side, gx - side / 2)))
            y0 = int(max(0, min(h - side, gy - side / 2)))
            crop = im.crop((x0, y0, min(w, x0 + side), min(h, y0 + side)))
            sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            sq.paste(crop, (0, 0))
            aligned.append(sq.resize((out_size, out_size), Image.LANCZOS))
        n = len(aligned)
        cols = max(1, min(n, int(cols)))
        rows = math.ceil(n / cols)
        sheet = Image.new("RGBA", (cols * out_size, rows * out_size),
                          (0, 0, 0, 0))
        for i, im in enumerate(aligned):
            sheet.paste(im, ((i % cols) * out_size,
                             (i // cols) * out_size), im)
        sbuf = BytesIO()
        if fmt == "webp":
            sheet.save(sbuf, format="WEBP", lossless=True)
        else:
            sheet.save(sbuf, format="PNG")
        import base64 as b64
        sheet_b64 = b64.b64encode(sbuf.getvalue()).decode()
        with tempfile.NamedTemporaryFile(suffix=".mp4",
                                         delete=False) as o:
            opath = o.name
        export_to_video([np.asarray(im.convert("RGB")) for im in
                         vframes], opath, fps=int(fps))
        with open(opath, "rb") as f:
            prev_b64 = b64.b64encode(f.read()).decode()
        meta = {"frames": n, "cols": cols, "rows": rows,
                "cell": out_size, "fps": int(fps),
                "format": fmt, "seed": seed,
                "sheet_w": sheet.width, "sheet_h": sheet.height}
        print(f"[done] {n}f {cols}x{rows} cell={out_size}")
        return {"sheet": sheet_b64, "preview_mp4": prev_b64,
                "meta": meta}
