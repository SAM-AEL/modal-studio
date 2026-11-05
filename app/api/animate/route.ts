import { ModalClient } from "modal";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

// Ideogram-like: both variants keep weights on the shared
// `wan-animate-model` Modal Volume and load from it on every run.
const MODEL_MAPPING: Record<string, { appName: string; className: string }> = {
  "wan-animate-bf16": { appName: "wan-animate-bf16-api", className: "WanAnimateBF16" },
  "wan-animate-fp8": { appName: "wan-animate-fp8-api", className: "WanAnimateFP8" },
};

export async function POST(req: Request) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const sendUpdate = (data: any) => {
        controller.enqueue(encoder.encode(JSON.stringify(data) + "\n"));
      };

      try {
        const body = await req.json();
        const {
          refImage, drivingVideo, prompt = "", negativePrompt = "",
          model = "wan-animate-fp8", height = 720, width = 1280,
          steps = 20, guidanceScale = 1.0, segmentFrames = 77,
          fps = 24, seed = -1,
        } = body;

        if (!refImage) {
          sendUpdate({ status: "error", error: "No reference image provided" });
          controller.close();
          return;
        }
        if (!drivingVideo) {
          sendUpdate({ status: "error", error: "No driving video provided" });
          controller.close();
          return;
        }

        const modelConfig = MODEL_MAPPING[model] || MODEL_MAPPING["wan-animate-fp8"];

        sendUpdate({ status: "initializing", details: `Waking up ${model}...` });
        const modal = new ModalClient();

        sendUpdate({ status: "connecting", details: `Connecting to ${modelConfig.className}...` });
        const cls = await modal.cls.fromName(modelConfig.appName, modelConfig.className);
        const instance = await cls.instance();
        const fn = instance.method("generate");

        sendUpdate({ status: "generating", details: `Animating via ${model} (this takes minutes)...` });

        const args = [
          refImage, drivingVideo,
          prompt, negativePrompt,
          height, width, steps, guidanceScale, segmentFrames, fps, seed,
        ];

        const result = await fn.remote(args);

        if (result) {
          sendUpdate({ status: "processing", details: "Converting video to display format..." });
          const video = Buffer.from(result).toString("base64");
          sendUpdate({ status: "success", video });
        } else {
          throw new Error("Modal did not return a video.");
        }

        controller.close();
      } catch (error) {
        console.error("Animate Route Error:", error);
        sendUpdate({ status: "error", error: error instanceof Error ? error.message : "Internal Error" });
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "application/x-ndjson" },
  });
}
