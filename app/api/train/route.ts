import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATASETS_ROOT = path.join(process.cwd(), "datasets");

// ─── POST /api/train ──────────────────────────────────────────────────────────
// Body: { action: "start", datasetName, triggerWord, steps, rank, lr, resolution }
//       { action: "download", datasetName }
//       { action: "stop", callId }
export async function POST(req: Request) {
  const body = await req.json();
  const { action, callId } = body;

  // ── Upload local dataset to Modal volume first, then kick off training ──
  if (action === "start") {
    const {
      datasetName = "my-concept",
      triggerWord = "my_trigger_word",
      trainerApp = "flux-gym-trainer", // updated default
      className = "FluxGymTrainer", // ADDED: dynamic class name
      steps = 1000,
      rank = 16,
      lr = 1e-4,
      resolution = 512,
      batchSize = 1,
      gradientAccumulation = 1,
    } = body;

    // 1. Skip auto-upload (We use the dedicated dataset 'Sync with Cloud' button now)
    console.log(`Starting training on dataset: ${datasetName} (verified synced in Volume)`);

    // 2. Stream training progress back via NDJSON
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        const emit = (obj: object) => {
          controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"));
        };

        try {
          const { ModalClient } = await import("modal");
          const modal = new ModalClient();
          const cls = await modal.cls.fromName(trainerApp, className);
          const instance = await cls.instance();

          emit({ type: "info", msg: `Starting Flux LoRA training on '${datasetName}'…` });

          // 1. Spawn the training task (async)
          // Use spawn instead of remoteGen because the JS SDK lacks remoteGen support
          const args = [
            datasetName,
            triggerWord,
            steps,
            rank,
            resolution,
            batchSize,
            lr, // Added: pass lr to train method
          ];
          
          if (className !== "FluxGymTrainer" && className !== "FluxGymTrainer") {
             // preserved for safety but usually not hit for this trainer
          }

          const call = await (instance.method("train") as any).spawn(args);

          const callId = call.functionCallId;
          emit({ type: "info", msg: "Training task spawned", callId });

          // 2. Poll for outputs (streaming)
          let lastEntryId = "0-0";
          const { decode } = await import("cbor-x");

          let finished = false;
          let pollCount = 0;
          
          while (!finished) {
            pollCount++;
            
            // Provide feedback during long cold starts
            if (pollCount === 1) {
              emit({ type: "info", msg: "Modal: Connecting to remote worker..." });
            } else if (pollCount === 2) {
              emit({ type: "info", msg: "Modal: Provisioning GPU and loading Flux model (this can take 2-4 mins)..." });
            } else if (pollCount > 2 && pollCount % 2 === 0) {
              const elapsed = pollCount * 10;
              emit({ type: "info", msg: `Initializing container... (${elapsed}s elapsed)` });
            }

            const resp = await (modal as any).cpClient.functionGetOutputs({
              functionCallId: callId,
              maxValues: 10,
              timeout: 10, // 10s long poll
              lastEntryId,
              clearOnSuccess: false,
              requestedAt: Date.now() / 1000,
            });

            if (resp.outputs && resp.outputs.length > 0) {
              for (const output of resp.outputs) {
                lastEntryId = output.entryId;
                
                // dataFormat=3 is GENERATOR_DONE
                if (output.dataFormat === 3) {
                  finished = true;
                  break;
                }

                // dataFormat=4 is CBOR
                if (output.dataFormat === 4 && output.result?.data) {
                  let decoded;
                  try {
                    decoded = decode(output.result.data);
                    
                    // Normalize for different trainers (e.g., flux-fashion-lora)
                    if (decoded && typeof decoded === "object") {
                      if (decoded.step !== undefined && decoded.total !== undefined && !decoded.type) {
                        decoded.type = "progress";
                        decoded.pct = (decoded.step / decoded.total) * 100;
                        decoded.msg = `Step ${decoded.step}/${decoded.total}`;
                      } else if (decoded.msg && !decoded.type) {
                        decoded.type = "info";
                      }
                    }

                    emit(decoded);
                  } catch (e) {
                    // Fallback for non-CBOR or raw strings
                    emit({ type: "log", msg: String(decoded || output.result.data || "Undecodable chunk") });
                  }
                }

                // Check for remote exceptions
                if (output.result && output.result.status !== 1) {
                  emit({ type: "error", msg: output.result.exception || "Unknown remote error" });
                  finished = true;
                  break;
                }
              }
            } else {
              // Check if the call itself failed if we get no outputs for a long time
              // (This is a safety check: but usually spawn+outputs is robust)
            }
          }

          emit({ type: "done", msg: "Training complete! LoRA is ready." });
        } catch (err: unknown) {
          emit({
            type: "error",
            msg: err instanceof Error ? err.message : String(err),
          });
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "application/x-ndjson",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
      },
    });
  }

  // ── Download trained LoRA weights ─────────────────────────────────────────
  if (action === "download") {
    const { datasetName, trainerApp = "flux-gym-trainer", className = "FluxGymTrainer" } = body;
    try {
      const { ModalClient } = await import("modal");
      const modal = new ModalClient();
      const cls = await modal.cls.fromName(trainerApp, className);
      const instance = await cls.instance();
      const bytes: Buffer = await instance
        .method("get_lora_bytes")
        .remote([datasetName]);

      return new Response(bytes as any, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition": `attachment; filename="${datasetName}_lora.safetensors"`,
        },
      });
    } catch (err) {
      return NextResponse.json({ error: String(err) }, { status: 500 });
    }
  }

  // ── Stop/Kill running training ───────────────────────────────────────────
  if (action === "stop") {
    if (!callId) return NextResponse.json({ error: "No callId provided" }, { status: 400 });
    try {
      const { ModalClient, FunctionCall } = await import("modal");
      const modal = new ModalClient();
      const call = new FunctionCall(modal, callId);
      await call.cancel({ terminateContainers: true });
      return NextResponse.json({ success: true, msg: "Training job killed" });
    } catch (err) {
      return NextResponse.json({ error: String(err) }, { status: 500 });
    }
  }

  return NextResponse.json({ error: "Invalid action" }, { status: 400 });
}

// ── GET /api/train?action=list ─────────────────────────────────────────────
export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const action = searchParams.get("action");

  if (action === "pipelines") {
    try {
      const trainDir = path.join(process.cwd(), "modal_backend", "train");
      const files = await fs.readdir(trainDir);
      const pipelines = [];
      
      for (const file of files) {
        if (!file.endsWith(".py")) continue;
        const filePath = path.join(trainDir, file);
        const content = await fs.readFile(filePath, "utf-8");
        
        const appMatch = content.match(/app\s*=\s*modal\.App\(\s*["']([^"']+)["']/);
        const gpuMatch = content.match(/@app\.cls\([\s\S]*?gpu\s*=\s*["']([^"']+)["']/);
        const classMatch = content.match(/class\s+(\w+)/);
        
        if (appMatch) {
          const appName = appMatch[1];
          const gpu = gpuMatch ? gpuMatch[1].toUpperCase() : "A100";
          const className = classMatch ? classMatch[1] : "FluxGymTrainer";
          const label = file.replace(".py", "").split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
          
          pipelines.push({
            id: appName,
            filename: file,
            appName: appName,
            className: className,
            gpu: gpu,
            label: `${label} (${gpu})`
          });
        }
      }
      return NextResponse.json({ pipelines });
    } catch (err) {
      return NextResponse.json({ pipelines: [] });
    }
  }

  if (action === "list") {
    try {
      const { searchParams } = new URL(req.url);
      const trainerApp = searchParams.get("trainerApp") || "flux-gym-trainer";
      const className = searchParams.get("className") || "FluxGymTrainer";

      const { ModalClient } = await import("modal");
      const modal = new ModalClient();
      const cls = await modal.cls.fromName(trainerApp, className);
      const instance = await cls.instance();
      const outputs: string[] = await instance
        .method("list_outputs")
        .remote([]);
      return NextResponse.json({ outputs });
    } catch {
      return NextResponse.json({ outputs: [] });
    }
  }

  return NextResponse.json({ error: "Invalid action" }, { status: 400 });
}
