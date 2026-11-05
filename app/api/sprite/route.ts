import { ModalClient } from "modal";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

export async function POST(req: Request) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: any) => {
        controller.enqueue(encoder.encode(JSON.stringify(data) + "\n"));
      };
      try {
        const b = await req.json();
        const {
          refImage, drivingVideo, prompt = "", negativePrompt = "",
          size = 720, steps = 20, guidanceScale = 1.0,
          fps = 12, seed = -1, outSize = 512, cols = 4,
          greenBg = true, format = "png",
        } = b;
        if (!refImage || !drivingVideo) {
          send({ status: "error", error: "Need refImage + drivingVideo" });
          controller.close();
          return;
        }
        send({ status: "initializing", details: "Waking up wan-animate-sprite..." });
        const modal = new ModalClient();
        const cls = await modal.cls.fromName("wan-animate-sprite-api", "WanAnimateSprite");
        const inst = await cls.instance();
        const fn = inst.method("animate_sprite");
        send({ status: "generating", details: "Wan animating + BiRefNet matting (minutes)..." });
        const args = [
          refImage, drivingVideo, prompt, negativePrompt,
          size, steps, guidanceScale, fps, seed, outSize, cols, greenBg, format,
        ];
        const r = await fn.remote(args);
        const raw = (r as any)?.value ?? r;
        const sheet = raw.sheet ?? raw[0];
        const preview = raw.preview_mp4 ?? raw[1];
        const meta = raw.meta ?? raw[2] ?? {};
        send({ status: "processing", details: "Packing spritesheet..." });
        send({ status: "success", sheet, preview, meta });
        controller.close();
      } catch (e) {
        console.error("Sprite route error:", e);
        send({ status: "error", error: e instanceof Error ? e.message : "Internal Error" });
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: { "Content-Type": "application/x-ndjson" },
  });
}
