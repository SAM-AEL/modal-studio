import { ModalClient } from "modal";

export const dynamic = "force-dynamic";

const MODEL_MAPPING: Record<string, { appName: string, className: string }> = {
  "flux-dev": { appName: "flux-krea-api", className: "FluxModel" },
  "flux-lora": { appName: "flux-lora-api", className: "FluxLoraModel" },
  "flux-schnell": { appName: "flux-schnell-api", className: "FluxSchnell" },
  "flux-krea-img2img": { appName: "flux-krea-img2img-api", className: "FluxImg2ImgModel" },
  "sdxl": { appName: "sdxl-api", className: "SDXLModel" },
  "sdxl-turbo": { appName: "sdxl-turbo-api", className: "SDXLTurboModel" },
  "sd15": { appName: "sd15-api", className: "SD15Model" },
  "sd35-medium": { appName: "sd35-medium-api", className: "SD35MediumModel" },
  "kolors": { appName: "kolors-api", className: "KolorsModel" },
  "hunyuandit": { appName: "hunyuandit-api", className: "HunyuanDiTModel" },
  "pixart-sigma": { appName: "pixart-sigma-api", className: "PixArtSigmaModel" },
  "auraflow": { appName: "auraflow-api", className: "AuraFlowModel" },
  "kandinsky3": { appName: "kandinsky3-api", className: "Kandinsky3Model" },
  "ideogram-4": { appName: "ideogram4-api", className: "Ideogram4Model" },
  "ideogram-4-fp8": { appName: "ideogram4-api", className: "Ideogram4Model" },
  "ideogram-4-bf16": { appName: "ideogram4-api", className: "Ideogram4Model" },
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
        const { prompt, seed, guidanceScale, steps, width, height, batchSize, model = "flux-dev" } = body;

        if (!prompt) {
          sendUpdate({ status: "error", error: "No prompt provided" });
          controller.close();
          return;
        }

        const modelConfig = MODEL_MAPPING[model] || MODEL_MAPPING["flux-dev"];

        sendUpdate({ status: "initializing", details: `Waking up ${model}...` });
        const modal = new ModalClient();

        sendUpdate({ status: "connecting", details: `Connecting to ${modelConfig.className}...` });
        const cls = await modal.cls.fromName(modelConfig.appName, modelConfig.className);
        const instance = await cls.instance();
        const fn = instance.method("generate");

        sendUpdate({ status: "generating", details: `Generating ${batchSize || 1} image(s) via ${model}...` });
        
        let args = [];
        
        if (model === "flux-krea-img2img") {
          args = [
            body.image, // image_base64
            prompt,
            body.strength ?? 0.6,
            seed ?? -1,
            guidanceScale ?? 3.5,
            steps ?? 28,
            width ?? 1024,
            height ?? 1024,
            batchSize ?? 1
          ];
        } else {
          args = [
            prompt, 
            seed ?? -1, 
            guidanceScale, 
            steps ?? 20, 
            width ?? 1024, 
            height ?? 1024, 
            batchSize ?? 1
          ];
          
          if (model === "flux-lora" && body.loras) {
            args.push(body.loras);
          } else if (model.startsWith("ideogram-4")) {
            args.push(Boolean(body.transparent));
          }
        }

        const results = await fn.remote(args);

        if (Array.isArray(results)) {
           sendUpdate({ status: "processing", details: "Converting images to display format..." });
           const images = results.map(res => Buffer.from(res).toString('base64'));
           sendUpdate({ status: "success", images });
        } else {
           throw new Error("Modal did not return an array of images.");
        }

        controller.close();
      } catch (error) {
        console.error("Route Error:", error);
        sendUpdate({ status: "error", error: error instanceof Error ? error.message : "Internal Error" });
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "application/x-ndjson" },
  });
}