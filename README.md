# Modal Studio — Serverless Image Generation & Flux LoRA Training

An open-source creative studio for **serverless text-to-image generation** and **Flux LoRA training** on [Modal](https://modal.com) GPUs.

> **Note:** This is a personal tool I built for my own use — not production software. Expect rough edges, hardcoded assumptions, and minimal error handling. Shared as-is in case it's useful; contributions welcome but support is best-effort.

- **Studio** (`/studio`) — generate images across 14 models (Ideogram 4, Flux.1, SDXL, SD 1.5/3.5, Kolors, HunyuanDiT, PixArt-Sigma, AuraFlow, Kandinsky 3) with live GPU cost estimates.
- **Image-to-Image** (`/img2img`) — upload a base image and describe variations (Flux Krea).
- **Datasets** (`/datasets`) — curate caption datasets, rename/caption in batch, sync to a Modal Volume, export ZIPs.
- **Train** (`/train`) — launch Flux LoRA training on A100s, stream logs, stop jobs, download `.safetensors` weights.
- **Settings** (`/settings`) — one-click seeding of shared model-weight volumes.

> **Stack:** Next.js 16 (App Router) · React 19 · Tailwind CSS 4 · Modal JS SDK (`modal`) + Modal Python SDK · diffusers/Flux training on CUDA 12.8.

---

## Table of Contents

- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Deploying the Modal backend](#deploying-the-modal-backend)
- [Using the app](#using-the-app)
- [Training LoRAs](#training-loras-full-workflow)
- [API reference](#api-reference)
- [Adding your own model](#adding-your-own-model)
- [Project structure](#project-structure)
- [GPU costs](#gpu-costs)
- [Security notes](#security-notes)
- [Contributing](#contributing)
- [License](#license)

---

## How it works

```
Browser (Next.js UI)
   │  fetch /api/* (NDJSON / JSON / SSE streams)
   ▼
Next.js API routes (app/api/*) — thin Modal JS SDK clients
   │  ModalClient: cls.fromName(app, class).instance().method(...).remote/spawn
   ▼
Modal Cloud — Python apps (modal_backend/*)
   │  GPU inference classes  ·  DatasetManager  ·  FluxGymTrainer  ·  LoraManager
   ▼
Modal Volumes: lora-datasets · lora-outputs · flux-model · hf-cache
```

Each file in `modal_backend/models/` is a self-contained Modal app
(`modal.App("<name>-api")`) exposing `generate(...) → list[bytes]` (PNG bytes).
The frontend maps friendly model IDs to `(appName, className)` pairs — see
[`app/api/generate/route.ts`](app/api/generate/route.ts).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Node.js 20+ | `node --version` |
| npm | Repo standardizes on npm (`package-lock.json`). |
| Python 3.11+ | Only needed to edit/deploy the Modal backend directly. |
| [Modal](https://modal.com) account | With GPU credits. CLI: `pip install modal && modal setup` |
| [Hugging Face](https://huggingface.co) token | Read access to gated weights (e.g. `black-forest-labs/FLUX.1-*`, `ideogram-ai/ideogram-4-*`). Get one at <https://huggingface.co/settings/tokens> |

---

## Quickstart

```bash
# 1. Clone
git clone <your-fork-url> modal-studio
cd modal-studio

# 2. Install frontend deps
npm install

# 3. Configure credentials (never commit this file — it's gitignored)
cp .env.example .env.local
# then edit .env.local:
#   MODAL_TOKEN_ID=...
#   MODAL_TOKEN_SECRET=...
#   HF_TOKEN=...
#   LORA_LOCAL_DIR=          # optional, see Configuration

# 4. Store the HF token as a Modal Secret (one time)
modal secret create huggingface-secret HF_TOKEN=$HF_TOKEN

# 5. Deploy the Modal backend (see table below)
modal deploy modal_backend/dataset_manager.py
modal deploy modal_backend/train/flux_gym_trainer.py
modal deploy modal_backend/models/flux_krea.py
# ... repeat per model, or deploy all:
for f in modal_backend/models/*.py; do modal deploy "$f"; done

# 6. Run the studio (port 3002)
npm run dev
# open http://localhost:3002
```

---

## Configuration

All secrets come from the environment — **no credentials live in the repo**.

| Variable | Where | Purpose |
|---|---|---|
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | `.env.local` (server) | Authenticates Next.js API routes via the Modal JS SDK. |
| `HF_TOKEN` | `.env.local` + Modal Secret `huggingface-secret` | Downloads gated Hugging Face weights inside Modal containers. |
| `LORA_LOCAL_DIR` | `.env.local` (optional) | Local folder of `*_lora.safetensors` files that `/api/loras` syncs into the `lora-outputs` volume. Unset = cloud-only listing. |

Checklist before first run:

- [ ] `.env.local` exists and is **not** committed (`git status` should not show it).
- [ ] `modal secret list` shows `huggingface-secret`.
- [ ] Required Modal apps are deployed (next section).
- [ ] Volumes exist — auto-created on first deploy/run (`create_if_missing=True`), except `flux-model` which is seeded via **Settings → Download Flux.1 weights**.


## Deploying the Modal backend

Each file is an independent Modal app. Deploy only what you need:

| File | App name | Class | GPU | Purpose |
|---|---|---|---|---|
| `modal_backend/dataset_manager.py` | `lora-dataset-manager` | `DatasetManager` | CPU | Dataset CRUD on `lora-datasets`; seeds `flux-model` |
| `modal_backend/train/flux_gym_trainer.py` | `flux-gym-trainer` | `FluxGymTrainer` | A100 | Flux LoRA training; `list_outputs`, `get_lora_bytes` |
| `modal_backend/models/flux_krea.py` | `flux-krea-api` | `FluxModel` | A100 | Flux.1-Krea-dev text-to-image |
| `modal_backend/models/flux_with_loras.py` | `flux-lora-api` | `FluxLoraModel` / `LoraManager` | A100 | Flux + LoRA inference; LoRA volume listing |
| `modal_backend/models/flux_schnell.py` | `flux-schnell-api` | `FluxSchnell` | A100 | Flux.1-Schnell (4-step) |
| `modal_backend/models/flux_krea_img2img.py` | `flux-krea-img2img-api` | `FluxImg2ImgModel` | A100 | Image-to-image editing |
| `modal_backend/models/ideogram4.py` | `ideogram4-api` | `Ideogram4Model` | H100 | Ideogram 4 (transparent sprites, JSON-caption safety fix) |
| `modal_backend/models/sdxl.py` | `sdxl-api` | `SDXLModel` | A10G | SDXL 1.0 |
| `modal_backend/models/sdxl_turbo.py` | `sdxl-turbo-api` | `SDXLTurboModel` | A10G | SDXL Turbo (1-step) |
| `modal_backend/models/sd15.py` | `sd15-api` | `SD15Model` | A10G | Stable Diffusion 1.5 |
| `modal_backend/models/sd35_medium.py` | `sd35-medium-api` | `SD35MediumModel` | A10G | SD 3.5 Medium |
| `modal_backend/models/kolors.py` | `kolors-api` | `KolorsModel` | A10G | Kolors (Alibaba) |
| `modal_backend/models/hunyuandit.py` | `hunyuandit-api` | `HunyuanDiTModel` | A100 | HunyuanDiT (Tencent) |
| `modal_backend/models/pixart_sigma.py` | `pixart-sigma-api` | `PixArtSigmaModel` | A10G | PixArt-Sigma |
| `modal_backend/models/auraflow.py` | `auraflow-api` | `AuraFlowModel` | A100 | AuraFlow v0.1 |
| `modal_backend/models/kandinsky3.py` | `kandinsky3-api` | `Kandinsky3Model` | A10G | Kandinsky 3.0 |

```bash
# Deploy everything (bash):
for f in modal_backend/dataset_manager.py modal_backend/train/flux_gym_trainer.py modal_backend/models/*.py; do
  modal deploy "$f"
done

# Verify:
modal app list
```

**Volumes** (created automatically):

| Volume | Stores |
|---|---|
| `lora-datasets` | Training images + `.txt` captions |
| `lora-outputs` | Trained `*.safetensors` LoRAs |
| `flux-model` | Cached FLUX.1-dev weights (seed via Settings page) |
| `flux-krea-model` | Cached FLUX.1-Krea-dev weights |
| `hf-cache` | Hugging Face hub cache shared by inference apps |

---

## Using the app

| Page | Route | What to do |
|---|---|---|
| Studio | `/studio` | Pick a model → write a prompt → tune seed/CFG/steps/size → Generate. Toggle **Transparent** for Ideogram sprite cutouts. |
| Image-to-Image | `/img2img` | Upload a base image, set Strength (0–1), describe the change, Generate. |
| Datasets | `/datasets` | Curate a dataset → **Rename Set** → **Batch Caption** → **Sync with Cloud** → **Download ZIP** to verify. |
| Train | `/train` | Select dataset + trigger word + steps/rank/LR → **Start Training** → watch console → **Download LoRA** when done. **Stop Training** kills the Modal call (stops GPU billing). |
| Settings | `/settings` | **Trigger Download** caches FLUX.1-dev weights into the `flux-model` volume. |

### Training defaults

`datasetName=my-concept`, `triggerWord=my_trigger_word`, `steps=1000`, `rank=32`, `lr=1e-4`, `resolution=1024`. Full walkthrough in [`HOW_TO_TRAIN.md`](HOW_TO_TRAIN.md).

---

## Training LoRAs (full workflow)

Condensed version — details in [`HOW_TO_TRAIN.md`](HOW_TO_TRAIN.md):

1. **Prepare images** — 10–20 high-quality PNG/JPGs of one subject/concept.
2. **Datasets page** — upload, **Rename Files (001…)**, **Batch Caption All** (include your trigger word in every caption, e.g. `"a photo of my_trigger_word in a forest"`).
3. **Sync with Cloud** — pushes images into the `lora-datasets` Modal Volume.
4. **Train page** — select dataset, trigger word, steps (~1000 for 15–20 images), rank 16–32, LR `1e-4`. Click **Start Training**.
5. **Monitor** — logs stream live from the A100 worker. Abort anytime with **Stop Training**.
6. **Download** — when status flips to **Training Complete**, use **Download LoRA** (or find it under **Ready LoRAs**).

> Training runs on A100 GPUs — ensure your Modal account has credits. Checkpoints are saved every 300 steps so interrupted runs can resume.

---

## API reference

All routes are Next.js Route Handlers; training calls stream **NDJSON** (`application/x-ndjson`, one JSON object per line).

| Route | Method | Body / Query | Returns |
|---|---|---|---|
| `/api/generate` | POST | `{ prompt, model, seed, guidanceScale, steps, width, height, batchSize, loras?, transparent?, image?, strength? }` | NDJSON status events → `{ status: "success", images: base64[] }` |
| `/api/train` | POST | `{ action: "start", datasetName, triggerWord, trainerApp?, className?, steps, rank, lr, resolution, batchSize }` | NDJSON `{ type: "info"/"log"/"progress"/"done"/"error" }` + `callId` |
| `/api/train` | POST | `{ action: "download", datasetName }` | `application/octet-stream` (`<dataset>_lora.safetensors`) |
| `/api/train` | POST | `{ action: "stop", callId }` | `{ success: true }` (cancels the Modal function call) |
| `/api/train?action=list` | GET | `?trainerApp=&className=` | `{ outputs: string[] }` |
| `/api/train?action=pipelines` | GET | — | `{ pipelines: [...] }` (parsed from `modal_backend/train/*.py` `# label:` comments) |
| `/api/pipelines` | GET | — | `{ pipelines: [...] }` (same source, alternate route) |
| `/api/dataset?action=list` | GET | `&cloud=true` to include Modal listing | `{ datasets: string[] }` |
| `/api/dataset?action=view` | GET | `?name=&cloud=true` | `{ files: [{ name, caption, data?, source }] }` |
| `/api/dataset` | POST | `{ action: "save"/"sync"/"rename"/"batch_caption"/"export", ... }` | JSON or ZIP (`export`) |
| `/api/loras` | GET | — | `{ loras: string[] }` (cloud volume + optional `LORA_LOCAL_DIR` sync) |
| `/api/settings` | POST | `{ action: "download_flux" }` | SSE stream of the weight-download job |
| `/api/pricing` | GET | — | `{ source, updatedAt, pricing }` — live scrape of modal.com/pricing with cached fallback |

---

## Adding your own model

Three steps. Takes about 5 minutes once you've done it once.

### 1. Create the Modal backend file

Copy the simplest existing model and adapt it:

```bash
cp modal_backend/models/sdxl.py modal_backend/models/my_model.py
```

Every model file follows the same shape — a Modal app with a class that
loads a diffusers pipeline once, then serves a `generate` method returning
PNG bytes. `sdxl.py` is the shortest full example, worth reading first.

Things worth knowing:

- The argument order of `generate` matters. `/api/generate` calls it
  positionally as `(prompt, seed, guidanceScale, steps, width, height,
  batchSize)`. Keep that signature unless you also update the route.
- If your model needs extra trailing args (like the Ideogram `transparent`
  flag or the Flux-LoRA `loras` list), append them after `batch_size` and
  handle them in `app/api/generate/route.ts` (see step 3).
- For gated repos, request access on Hugging Face first, then add
  `secrets=[modal.Secret.from_name("huggingface-secret")]` to the `@app.cls`
  decorator and pass `token=os.environ["HF_TOKEN"]` to `from_pretrained`.
  (See `flux_schnell.py` for a working example.)
- GPU cheat sheet: A10G (24GB) is fine for SD1.5/SDXL-size models, A100 for
  Flux-size, H100 for the big ones like Ideogram 4. Bigger than needed just
  costs more per second.

### 2. Deploy it to Modal

```bash
# test it first (runs once, prints any errors):
modal run modal_backend/models/my_model.py

# deploy it as a serverless endpoint:
modal deploy modal_backend/models/my_model.py

# confirm it's live:
modal app list
```

### 3. Wire it into the frontend

Register the app in `app/api/generate/route.ts`:

```ts
const MODEL_MAPPING = {
  // ...
  "my-model": { appName: "my-model-api", className: "MyModel" },
};
```

Then add it to the picker in `app/studio/page.tsx`:

```tsx
{ id: "my-model", name: "My Model", cfg: 7.5, steps: 30, gpu: "A10G", requiredVram: 12 },
```

`cfg`/`steps` are just the defaults pre-filled on selection. `gpu` and
`requiredVram` drive the cost estimate and the "not enough VRAM" warning,
so set them honestly. Restart `npm run dev` and it shows up in the Studio
dropdown.

---

## Project structure

```
modal-studio/
├── app/
│   ├── api/
│   │   ├── dataset/route.ts     # dataset CRUD via DatasetManager
│   │   ├── generate/route.ts    # MODEL_MAPPING → Modal inference apps
│   │   ├── loras/route.ts       # LoRA listing (+ optional local-folder sync)
│   │   ├── pipelines/route.ts   # training-pipeline discovery
│   │   ├── pricing/route.ts     # GPU pricing (live + DEFAULT_GPU_PRICING fallback)
│   │   ├── settings/route.ts    # weight-seeding jobs
│   │   └── train/route.ts       # training start/stop/download/list
│   ├── components/Navbar.tsx
│   ├── datasets/page.tsx
│   ├── img2img/page.tsx
│   ├── settings/page.tsx
│   ├── studio/page.tsx
│   ├── train/page.tsx
│   ├── layout.tsx
│   ├── page.tsx                 # → redirects to /studio
│   └── globals.css
├── modal_backend/
│   ├── dataset_manager.py       # lora-dataset-manager app
│   ├── models/                  # one Modal app per inference model
│   │   ├── flux_krea.py  flux_with_loras.py  flux_schnell.py
│   │   ├── flux_krea_img2img.py  ideogram4.py  sdxl.py  sdxl_turbo.py
│   │   ├── sd15.py  sd35_medium.py  kolors.py  hunyuandit.py
│   │   └── pixart_sigma.py  auraflow.py  kandinsky3.py
│   └── train/
│       └── flux_gym_trainer.py  # flux-gym-trainer app (diffusers DreamBooth-LoRA)
├── public/                      # static assets (add your own logo/og image)
├── .env.example                 # template — copy to .env.local (gitignored)
├── HOW_TO_TRAIN.md              # detailed LoRA training guide
├── LICENSE                      # MIT
└── package.json                 # npm-based (Next 16, React 19, modal JS SDK)
```

---

## GPU costs

The Studio shows per-generation cost estimates. Pricing is fetched live from
<https://modal.com/pricing> by `/api/pricing` (cached 1h) with a hardcoded
fallback (`DEFAULT_GPU_PRICING` in `app/api/pricing/route.ts`). Spot-check the
fallback table against modal.com periodically — GPU prices change.

---

## Security notes

- `.env.local` holds your real tokens and is gitignored — don't commit it.
- Keep your Modal and Hugging Face tokens private. If one leaks, revoke it
  and make a new one, then update `.env.local` and the `huggingface-secret`
  Modal Secret (see Quickstart step 4).
- The backend expects a Modal Secret called `huggingface-secret` containing
  `HF_TOKEN`.

---

## Contributing

1. Fork → feature branch → PR.
2. `npm run lint` and `npm run build` must pass.
3. Do not commit `.env.local`, weights (`*.safetensors`), `datasets/`, `lora/`, `__pycache__`, or lockfiles other than `package-lock.json`.
4. For new models, follow [Adding your own model](#adding-your-own-model).

---

## License

MIT — see [LICENSE](LICENSE).

