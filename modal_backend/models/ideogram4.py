import modal

app = modal.App("ideogram4-api")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "bitsandbytes",
        "Pillow",
        "huggingface_hub",
        "sentencepiece"
    )
    .env({"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
)

@app.cls(
    gpu="H100",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=120,  # keep warm 2 min to avoid cold starts between generations
    timeout=900,  # headroom for batched / 2K renders on a warm container
    startup_timeout=600,
)
class Ideogram4Model:

    @modal.enter()
    def load(self):
        import torch
        import os
        from diffusers import Ideogram4Pipeline

        token = os.environ.get("HF_TOKEN")
        model_id = "ideogram-ai/ideogram-4-nf4-diffusers"

        print(f"Loading {model_id} onto GPU...")
        self.pipe = Ideogram4Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=token
        )
        self.pipe.to("cuda")

        # H100 optimizations: TF32 for ~10-20% matmul speedup, no visible
        # quality loss on bf16/NF4.
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        except Exception as be:
            print(f"Backend flag notice: {be}")

        # Silence per-step progress logs (less stream spam on warm container).
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        # H100 has 80GB: keep VAE tiling for 1K-2K renders, but skip
        # enable_slicing (saves VRAM at the cost of speed — unnecessary here).
        if hasattr(self.pipe, "vae") and self.pipe.vae:
            try:
                if hasattr(self.pipe.vae, "enable_tiling"):
                    self.pipe.vae.enable_tiling()
            except Exception as vae_err:
                print(f"VAE tiling notice: {vae_err}")

        # Persist downloaded weights to the Modal Volume so future containers never redownload
        try:
            hf_cache.commit()
            print("Successfully committed model weights to hf-cache volume.")
        except Exception as e:
            print(f"Volume commit notice: {e}")

    def _to_json_caption(self, prompt: str, transparent: bool, width: int, height: int) -> str:
        """Wrap plain-text prompts in Ideogram 4's native JSON caption format.

        Plain text "will not work and will likely trigger a safety warning"
        (grey "Image blocked by safety filter" screen). JSON captions match
        training distribution and drastically cut false positives.
        Full schema needs high_level_description + style_description
        (exactly one of photo/art_style, strict key order) +
        compositional_deconstruction(background, elements).
        Serialize with separators=(",", ":") and ensure_ascii=False.
        """
        import json

        raw = (prompt or "").strip()
        if not raw:
            raw = "a high quality image"

        # Pass through prompts that are already valid JSON captions.
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "compositional_deconstruction" in parsed:
                    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
            except Exception:
                pass

        low = raw.lower()
        is_asset = transparent or any(
            k in low
            for k in (
                "slot", "symbol", "icon", "doubloon", "coin", "emerald",
                "wild", "scatter", "badge", "game asset", "cutout",
                "transparent", "sticker", "logo",
            )
        )

        if is_asset:
            desc = raw
            if "transparent" not in low:
                desc = (
                    f"{raw}, isolated game asset, centered, crisp sharp edges, "
                    f"transparent background, alpha PNG cutout, 8k resolution"
                )
            # Non-photo variant: strict order aesthetics, lighting, medium, art_style
            caption = {
                "high_level_description": desc,
                "style_description": {
                    "aesthetics": "clean, sharp, vibrant, centered",
                    "lighting": "even studio lighting, soft subtle shadows",
                    "medium": "3d_render",
                    "art_style": "stylized 3D game asset render, crisp edges, high detail",
                },
                "compositional_deconstruction": {
                    "background": "Transparent background, isolated game asset cutout, no scene, no shadow, no environment",
                    "elements": [{"type": "obj", "desc": desc}],
                },
            }
        else:
            # Photo variant: strict order aesthetics, lighting, photo, medium
            caption = {
                "high_level_description": raw,
                "style_description": {
                    "aesthetics": "natural, detailed, balanced",
                    "lighting": "soft natural daylight, gentle shadows",
                    "photo": "sharp focus, eye-level, natural colors",
                    "medium": "photograph",
                },
                "compositional_deconstruction": {
                    "background": f"Natural environment complementing the scene: {raw}",
                    "elements": [{"type": "obj", "desc": raw}],
                },
            }
        return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)

    @modal.method()
    def generate(
        self,
        prompt: str,
        seed: int = -1,
        guidance_scale: float = 4.5,
        num_inference_steps: int = 30,
        width: int = 1024,
        height: int = 1024,
        batch_size: int = 1,
        transparent: bool = False
    ):
        import torch
        import random
        import json
        from io import BytesIO

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # --- Ideogram 4 safety-filter fix ---
        # The model was trained EXCLUSIVELY on structured JSON captions.
        # Per ideogram-oss/ideogram4 docs/prompting.md:
        #   "Passing in plain-text prompts directly to the model will not work
        #    and will likely trigger a safety warning."
        # Instead of an image it returns a grey screen with the text
        # "Image blocked by safety filter". False-positive rate is high for
        # non-JSON prompts. This is NOT caused by guidance_scale/CFG.
        # So: if the caller passes plain text, wrap it in a minimal valid
        # JSON caption with strict key order, serialized with
        # separators=(",", ":") and ensure_ascii=False.
        final_prompt = self._to_json_caption(prompt, transparent, width, height)

        actual_guidance = max(1.0, min(12.0, float(guidance_scale) if guidance_scale is not None else 4.5))
        actual_steps = max(1, min(50, int(num_inference_steps) if num_inference_steps is not None else 25))
        batch_size = max(1, min(4, int(batch_size) if batch_size else 1))

        results = []
        with torch.inference_mode():
            for i in range(batch_size):
                current_seed = seed + i
                generator = torch.Generator(device="cuda").manual_seed(current_seed)

                output = self.pipe(
                    prompt=final_prompt,
                    guidance_scale=actual_guidance,
                    guidance_schedule=None,
                    num_inference_steps=actual_steps,
                    width=width,
                    height=height,
                    generator=generator
                )

                img = output.images[0]

                # Ensure RGBA image format for transparent sprites
                if transparent and img.mode != "RGBA":
                    img = img.convert("RGBA")

                buf = BytesIO()
                img.save(buf, format="PNG")
                results.append(buf.getvalue())

        return results
