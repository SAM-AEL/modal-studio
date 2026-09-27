import { ModalClient } from "modal";
import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { action } = body;

    const modal = new ModalClient();
    
    if (action === "download_flux") {
      const cls = await modal.cls.fromName("lora-dataset-manager", "DatasetManager");
      const instance = await cls.instance();

      // Spawn download async
      const call = await (instance.method("download_flux") as any).spawn([]);
      const callId = call.functionCallId;

      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async start(controller) {
          const emit = (obj: any) => {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
          };

          emit({ type: "info", msg: "Download task spawned", callId });

          let lastEntryId = "0-0";
          let finished = false;
          
          while (!finished) {
            const resp = await (modal as any).cpClient.functionGetOutputs({
              functionCallId: callId,
              maxValues: 10,
              timeout: 10,
              lastEntryId,
              clearOnSuccess: false,
              requestedAt: Date.now() / 1000,
            });

            if (resp.outputs && resp.outputs.length > 0) {
              for (const output of resp.outputs) {
                lastEntryId = output.entryId;
                if (output.type === "stdout" || output.type === "stderr") {
                    emit({ type: "log", msg: output.value });
                }
                if (output.type === "return") {
                    emit({ type: "done", msg: "Download complete!" });
                    finished = true;
                }
                if (output.type === "exception") {
                    emit({ type: "error", msg: `Download failed: ${output.value}` });
                    finished = true;
                }
              }
            }
            await new Promise(resolve => setTimeout(resolve, 2000));
          }
          controller.close();
        }
      });

      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        },
      });
    }

    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
