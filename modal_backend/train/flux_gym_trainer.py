import modal
import os
import shutil
import subprocess
import sys
import time
import re
import threading
import base64
from pathlib import Path

# label: "Flux Gym Trainer (A100)"

CUDA_VERSION = "12.8.0"
BASE_IMAGE = f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04"
_CACHE_BUST = "2026-03-16-v28"

# Base64-encode the patch script to avoid all shell/Dockerfile quoting issues
_PATCH_SCRIPT = base64.b64encode("""
with open('/train_dreambooth_lora_flux.py') as f:
    c = f.read()
old = 'instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]'
new = 'instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir()) if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")]'
assert old in c, "Patch target not found - diffusers script may have changed"
with open('/train_dreambooth_lora_flux.py', 'w') as f:
    f.write(c.replace(old, new))
print("Patch applied OK")
""".encode()).decode()

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install(
        "git", "wget", "libgl1", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev",
        "build-essential",
    )
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.46.0",
        "accelerate>=1.1.0",
        "bitsandbytes>=0.44.0",
        "peft>=0.13.0",
        "safetensors>=0.4.5",
        "huggingface_hub>=0.26.0",
        "sentencepiece>=0.2.0",
    )
    .pip_install(
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
        "einops>=0.8.0",
        "prodigyopt>=1.0",
        "datasets>=3.0.0",
    )
    .run_commands(
        f"echo '{_CACHE_BUST}'",
        "wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux.py -O /train_dreambooth_lora_flux.py",
        f"echo '{_PATCH_SCRIPT}' | base64 -d > /patch_train.py",
        "python3 /patch_train.py",
        "python -c \"import torch; print('torch', torch.__version__); print('CUDA:', torch.version.cuda)\"",
    )
    .env({
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "CUDA_MODULE_LOADING": "LAZY",
    })
)

app = modal.App("flux-gym-trainer", image=image)

dataset_volume = modal.Volume.from_name("lora-datasets",  create_if_missing=True)
output_volume  = modal.Volume.from_name("lora-outputs",   create_if_missing=True)
model_volume   = modal.Volume.from_name("flux-model",     create_if_missing=False)

LOCAL_DS   = "/local/dataset"
LOCAL_OUT  = "/local/outputs"
LOCAL_LOGS = "/tmp/logs"

# --------------------------------------------------------------------------
# Fixed training constants
# --------------------------------------------------------------------------
REPEATS            = 20
RANK               = 32
RESOLUTION         = 1024
BATCH_SIZE         = 1
GRAD_ACCUM         = 1
LEARNING_RATE      = "2e-4"
LR_SCHEDULER       = "cosine_with_restarts"
LR_WARMUP_STEPS    = 50
MIXED_PRECISION    = "bf16"
HF_MODEL_ID        = "black-forest-labs/FLUX.1-dev"
HF_MODEL_CACHE     = "/flux-model"

# Save a full diffusers checkpoint every N steps so we can resume if killed.
# The checkpoint at step 1000 is saved to the volume mid-run.
CHECKPOINT_STEPS   = 300

_TAIL_LINES = 40

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def ensure_hf_model() -> str:
    from huggingface_hub import snapshot_download
    local = Path(HF_MODEL_CACHE) / "FLUX.1-dev"
    sentinel = local / "model_index.json"
    if sentinel.exists():
        print(f"  ✓ Model already at {local}")
        return str(local)
    print(f"  model_index.json not found — downloading {HF_MODEL_ID}...")
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=str(local),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        token=os.environ.get("HF_TOKEN"),
    )
    model_volume.commit()
    print(f"  ✓ Model cached: {local}")
    return str(local)


def stage_dataset(src: str, dataset_name: str) -> str:
    local_path = os.path.join(LOCAL_DS, dataset_name)
    os.makedirs(local_path, exist_ok=True)
    existing = [f for f in os.listdir(local_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    if existing:
        print(f"  ✓ {len(existing)} images already on local NVMe")
        return local_path
    t = time.time()
    n = 0
    for fname in os.listdir(src):
        if not fname.startswith("."):
            shutil.copy2(os.path.join(src, fname), os.path.join(local_path, fname))
            n += 1
    print(f"  ✓ Copied {n} files in {time.time()-t:.1f}s")
    return local_path


def find_resume_checkpoint(vol_out_dir: str, local_out: str) -> str | None:
    """
    Check if a diffusers checkpoint folder exists in the output volume.
    If found, copy it to local NVMe and return 'latest' so diffusers resumes.
    """
    vol_ckpt_dir = Path(vol_out_dir) / "checkpoints"
    if not vol_ckpt_dir.exists():
        return None

    # Find the highest-numbered checkpoint
    ckpts = sorted(vol_ckpt_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        return None

    # Filter for checkpoints that are complete (have optimizer.bin)
    valid_ckpts = [p for p in ckpts if (p / "optimizer.bin").exists()]
    if not valid_ckpts:
        print("  ⚠️ Found checkpoints but none are complete (missing optimizer.bin) — starting fresh")
        return None

    latest = valid_ckpts[-1]
    step_num = int(latest.name.split("-")[1])
    local_ckpt_parent = Path(local_out)
    local_ckpt_path = local_ckpt_parent / latest.name

    print(f"  ✓ Found checkpoint at step {step_num} in volume — copying to NVMe...")
    if local_ckpt_path.exists():
        shutil.rmtree(str(local_ckpt_path))
    shutil.copytree(str(latest), str(local_ckpt_path))
    print(f"  ✓ Checkpoint ready at {local_ckpt_path}")
    return "latest"


def save_checkpoint_to_volume(local_out: str, vol_out_dir: str, log_fn):
    """
    Copy any checkpoint-NNNN folders from local NVMe to the output volume.
    Called mid-training after we detect a checkpoint was saved.
    """
    local_out_path = Path(local_out)
    ckpts = sorted(local_out_path.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        return

    vol_ckpt_dir = Path(vol_out_dir) / "checkpoints"
    vol_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ckpt in ckpts:
        dst = vol_ckpt_dir / ckpt.name
        if dst.exists():
            continue  # already saved
        log_fn(f"  💾 Saving checkpoint {ckpt.name} to volume...")
        shutil.copytree(str(ckpt), str(dst))

    output_volume.commit()
    log_fn(f"  ✓ Checkpoint(s) committed to volume")


def build_cmd(
    model_path: str,
    dataset_dir: str,
    output_dir: str,
    output_name: str,
    trigger_word: str,
    steps: int,
    rank: int = 32,
    resolution: int = 1024,
    batch_size: int = 1,
    lr: float = 2e-4,
    resume_from_checkpoint: str | None = None,
) -> list:
    train_cmd = [
        sys.executable, "/train_dreambooth_lora_flux.py",

        f"--pretrained_model_name_or_path={model_path}",

        f"--instance_data_dir={dataset_dir}",
        f"--instance_prompt={trigger_word}",

        f"--output_dir={output_dir}",

        f"--rank={rank}",

        f"--resolution={resolution}",
        "--random_flip",

        f"--train_batch_size={batch_size}",
        f"--gradient_accumulation_steps={GRAD_ACCUM}",
        "--gradient_checkpointing",

        f"--learning_rate={lr}",
        f"--lr_scheduler={LR_SCHEDULER}",
        f"--lr_warmup_steps={LR_WARMUP_STEPS}",

        f"--max_train_steps={steps}",
        "--seed=42",
        f"--mixed_precision={MIXED_PRECISION}",
        "--use_8bit_adam",

        # Save a full diffusers checkpoint every CHECKPOINT_STEPS steps
        # so we can resume if the container is killed
        f"--checkpointing_steps={CHECKPOINT_STEPS}",

        "--guidance_scale=1",

        f"--logging_dir={LOCAL_LOGS}",
        "--report_to=tensorboard",
    ]

    if resume_from_checkpoint:
        train_cmd.append(f"--resume_from_checkpoint={resume_from_checkpoint}")

    # stdbuf forces line-buffered stdout/stderr so tqdm lines flush immediately
    return ["stdbuf", "-oL", "-eL"] + train_cmd


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

@app.cls(
    gpu="a100-80gb",
    volumes={
        "/datasets":   dataset_volume,
        "/outputs":    output_volume,
        "/flux-model": model_volume,
    },
    timeout=7200,
    scaledown_window=180,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=1,
    ephemeral_disk=512 * 1024,
)
class FluxGymTrainer:

    @modal.method()
    def train(
        self,
        dataset_name: str,
        trigger_word: str = "TOK",
        steps: int = 2000,
        rank: int = 32,
        resolution: int = 1024,
        batch_size: int = 1,
        lr: float = 2e-4,
    ):
        import torch
        t0 = time.time()

        def _log(msg, kind="info"):
            elapsed = f"[{time.time()-t0:.0f}s]"
            print(f"{elapsed} {msg}", flush=True)
            return {"type": kind, "msg": f"{elapsed} {msg}"}

        yield _log(f"GPU  : {torch.cuda.get_device_name(0)}")
        yield _log(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        yield _log(f"Torch: {torch.__version__}  CUDA: {torch.version.cuda}")
        yield _log(
            f"Config: {steps} steps | rank {rank} | {resolution}px | "
            f"batch {batch_size} (accum {GRAD_ACCUM}) | lr {lr}"
        )

        # ------------------------------------------------------------------
        yield _log("Phase 1/4 — Ensuring model is cached...")
        try:
            model_path = ensure_hf_model()
        except Exception as e:
            yield _log(f"❌ {e}", "error"); raise

        # ------------------------------------------------------------------
        yield _log("Phase 2/4 — Staging dataset...")
        dataset_volume.reload()
        output_volume.reload()

        src = f"/datasets/{dataset_name}"
        if not os.path.exists(src):
            yield _log(f"❌ Dataset not found: {src}", "error")
            raise RuntimeError(f"Dataset not found: {src}")

        local_ds = stage_dataset(src, dataset_name)
        images = [f for f in os.listdir(local_ds)
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        yield _log(f"  {len(images)} images")
        if not images:
            raise RuntimeError("No images in dataset")

        has_captions = any(
            (Path(local_ds) / (Path(img).stem + ".txt")).exists()
            for img in images
        )
        if not has_captions:
            yield _log(f"  No captions found — using trigger word '{trigger_word}' for all images")
        else:
            yield _log(f"  Per-image caption files detected — using those")

        total_samples   = len(images) * REPEATS
        steps_per_epoch = max(1, total_samples // (batch_size * GRAD_ACCUM))
        num_epochs      = steps / steps_per_epoch
        yield _log(
            f"  {len(images)} imgs x {REPEATS} repeats -> "
            f"{steps_per_epoch} steps/epoch | {num_epochs:.1f} epochs"
        )

        # ------------------------------------------------------------------
        yield _log("Phase 3/4 — Training...")
        local_out = os.path.join(LOCAL_OUT, dataset_name)
        os.makedirs(local_out, exist_ok=True)
        os.makedirs(LOCAL_LOGS, exist_ok=True)

        vol_out_dir = f"/outputs/{dataset_name}"
        os.makedirs(vol_out_dir, exist_ok=True)

        output_name = f"{dataset_name}_lora"

        # Check if a checkpoint exists in the volume to resume from
        resume_from = find_resume_checkpoint(vol_out_dir, local_out)
        if resume_from:
            yield _log(f"  ↩️  Resuming from checkpoint in volume")
        else:
            yield _log(f"  Starting fresh — checkpoints every {CHECKPOINT_STEPS} steps")

        cmd = build_cmd(
            model_path, local_ds, local_out, output_name, trigger_word,
            steps, rank, resolution, batch_size, lr,
            resume_from_checkpoint=resume_from,
        )

        print("CMD:\n  " + " \\\n  ".join(cmd), flush=True)
        yield _log("  gradient_checkpointing + 8bit_adam — expect ~50-60 GB VRAM")
        yield _log(f"  {steps} steps at ~1-2s/it")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                # TERM=dumb stops tqdm using \r carriage returns so every
                # progress line becomes a newline that flushes through the pipe
                "TERM": "dumb",
                "TQDM_MININTERVAL": "5",
            },
        )

        import queue
        from collections import deque

        output_tail: deque[str] = deque(maxlen=_TAIL_LINES)
        line_queue: queue.Queue = queue.Queue()
        last_log_ts = [time.time()]
        # Track which checkpoints we have already committed to the volume
        committed_checkpoints: set = set()

        def _reader():
            for raw in proc.stdout:
                line_queue.put(raw.rstrip())
            line_queue.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        def heartbeat():
            while proc.poll() is None:
                time.sleep(60)
                idle = time.time() - last_log_ts[0]
                if idle > 120:
                    print(f"⏳ No output for {idle:.0f}s — still running...", flush=True)

        threading.Thread(target=heartbeat, daemon=True).start()

        last_step_seen = -1

        while True:
            try:
                line = line_queue.get(timeout=1)
            except queue.Empty:
                continue

            if line is None:
                break
            if not line:
                continue

            last_log_ts[0] = time.time()
            output_tail.append(line)
            print(f"  [sd] {line}", flush=True)

            # Diffusers accelerate progress: "Steps: 42%|████ | 840/2000 [...]"
            m = re.search(r"Steps:.*?(\d+)/(\d+)", line)
            if not m:
                m = re.search(r"(\d+)/(\d+).*?(?:loss|avr_loss)=([\d.]+)", line)

            if m:
                cur   = int(m.group(1))
                total = int(m.group(2))
                if cur == last_step_seen:
                    continue
                last_step_seen = cur

                loss_m = re.search(r"loss[=:\s]+([\d.]+)", line)
                loss   = float(loss_m.group(1)) if loss_m else 0.0

                elapsed_min = (time.time() - t0) / 60
                yield {
                    "type":        "progress",
                    "step":        cur,
                    "total":       total,
                    "pct":         round(cur / total * 100, 1),
                    "loss":        loss,
                    "elapsed_min": round(elapsed_min, 1),
                    "msg":         f"Step {cur}/{total} ({cur/total*100:.1f}%)"
                                   + (f" loss={loss:.4f}" if loss else "")
                                   + f" [{elapsed_min:.1f}min]",
                }
                continue

            lower = line.lower()

            # Detect when diffusers saves a checkpoint and immediately
            # commit it to the volume so we don't lose it if killed
            # Detect when diffusers is FINISHED saving
            if "checkpoint saved to" in lower or "model weights saved" in lower:
                yield {"type": "log", "msg": f"💾 {line}"}
                
                # Wait a few seconds for the OS to flush the file writes
                time.sleep(5) 
                
                new_ckpts = [
                    p for p in Path(local_out).glob("checkpoint-*")
                    if p.name not in committed_checkpoints
                ]
                
                for ckpt in new_ckpts:
                    # CRITICAL: Only copy if the optimizer file actually exists locally
                    if not (ckpt / "optimizer.bin").exists():
                        continue 
                        
                    try:
                        vol_ckpt_dir = Path(vol_out_dir) / "checkpoints"
                        vol_ckpt_dir.mkdir(parents=True, exist_ok=True)
                        dst = vol_ckpt_dir / ckpt.name
                        
                        # Copy to a temporary name first, then rename (atomic swap)
                        temp_dst = vol_ckpt_dir / f"tmp_{ckpt.name}"
                        if temp_dst.exists(): shutil.rmtree(str(temp_dst))
                        
                        shutil.copytree(str(ckpt), str(temp_dst))
                        if dst.exists(): shutil.rmtree(str(dst))
                        temp_dst.rename(dst) 
                        
                        output_volume.commit()
                        committed_checkpoints.add(ckpt.name)
                        yield _log(f"  ✓ {ckpt.name} successfully committed to volume")
                    except Exception as e:
                        yield _log(f"  ⚠️ Failed to commit {ckpt.name}: {e}", "warning")

            elif "saving" in lower and ("lora" in lower or "weights" in lower):
                yield {"type": "log", "msg": f"💾 {line}"}
            elif "epoch" in lower:
                yield {"type": "log", "msg": f"📚 {line}"}
            elif "warning" in lower:
                yield {"type": "log", "msg": f"⚠️ {line}"}
            elif "error" in lower:
                yield {"type": "log", "msg": f"❌ {line}"}

        proc.wait()
        total_min = (time.time() - t0) / 60

        if proc.returncode != 0:
            if output_tail:
                yield _log(f"❌ Last {len(output_tail)} lines:", "error")
                for ln in output_tail:
                    yield {"type": "error", "msg": f"  [tail] {ln}"}
            yield _log(f"❌ Training failed (exit {proc.returncode})", "error")
            raise RuntimeError(f"Training exited {proc.returncode}")

        # ------------------------------------------------------------------
        yield _log("Phase 4/4 — Saving LoRA to volume...")

        # diffusers saves the final LoRA as pytorch_lora_weights.safetensors
        candidates = sorted(Path(local_out).rglob("pytorch_lora_weights.safetensors"))
        # fallback: any safetensors not inside a checkpoint- folder
        if not candidates:
            candidates = [
                p for p in Path(local_out).rglob("*.safetensors")
                if "checkpoint-" not in str(p)
            ]

        out_file = candidates[-1] if candidates else None

        size_mb = 0
        dst = None
        if out_file and out_file.exists():
            dst = Path(vol_out_dir) / f"{output_name}.safetensors"
            shutil.copy2(str(out_file), str(dst))
            output_volume.commit()
            size_mb = round(out_file.stat().st_size / 1e6, 1)
            yield _log(f"  LoRA -> {dst}  ({size_mb} MB)")
        else:
            yield _log("⚠️  No .safetensors found in output dir", "warning")

        yield _log(f"✅ Done in {total_min:.1f} min")
        yield {
            "type":       "done",
            "msg":        f"LoRA saved ({size_mb} MB) in {total_min:.1f} min",
            "output_dir": vol_out_dir,
            "lora_path":  str(dst) if dst else "",
            "total_min":  round(total_min, 1),
        }

    @modal.method()
    def list_outputs(self):
        output_volume.reload()
        if not os.path.exists("/outputs"):
            return []
        return [d for d in os.listdir("/outputs") if os.path.isdir(f"/outputs/{d}")]

    @modal.method()
    def get_lora_bytes(self, dataset_name: str) -> bytes:
        output_volume.reload()
        for root, _, files in os.walk(f"/outputs/{dataset_name}"):
            for f in files:
                if f.endswith(".safetensors") and "checkpoint-" not in root:
                    return Path(os.path.join(root, f)).read_bytes()
        raise FileNotFoundError(f"No LoRA found for '{dataset_name}'")


@app.local_entrypoint()
def main(
    dataset_name: str,
    trigger_word: str = "TOK",
    steps: int = 2000,
):
    print(f"Starting: dataset={dataset_name}  trigger='{trigger_word}'  steps={steps}")
    trainer = FluxGymTrainer()

    for update in trainer.train.remote_gen(
        dataset_name=dataset_name,
        trigger_word=trigger_word,
        steps=steps,
    ):
        if not isinstance(update, dict):
            print(update)
            continue

        kind = update.get("type", "log")
        msg  = update.get("msg", "")

        if kind == "progress":
            print(f"\r  {msg}", end="", flush=True)

        elif kind == "done":
            print()
            print(f"\n✅ {msg}")
            print(f"   output_dir : {update.get('output_dir', '')}")
            print(f"   lora_path  : {update.get('lora_path', '')}")
            print(f"   total_min  : {update.get('total_min', '')} min")
            break

        elif kind == "error":
            print()
            print(f"❌ {msg}", flush=True)

        else:
            if msg:
                print(f"  {msg}")