import { ModalClient } from "modal";
import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

const DATASETS_ROOT = path.join(process.cwd(), "datasets");

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const action = searchParams.get("action");
    const datasetName = searchParams.get("name") || "default";

    if (action === "list") {
      // 1. List Local (missing folder = no local datasets, not an error)
      let localNames: string[] = [];
      try {
        const localDirs = await fs.readdir(DATASETS_ROOT, { withFileTypes: true });
        localNames = localDirs.filter((d) => d.isDirectory()).map((d) => d.name);
      } catch (err: any) {
        if (err?.code !== "ENOENT") throw err;
      }

      // 2. List Cloud (Only if explicitly requested via ?cloud=true)
      let cloudNames: string[] = [];
      if (searchParams.get("cloud") === "true") {
        const { ModalClient } = await import("modal");
        const modal = new ModalClient();
        const cls = await modal.cls.fromName("lora-dataset-manager", "DatasetManager");
        const instance = await cls.instance();
        cloudNames = await instance.method("list_datasets").remote([]);
      }

      // Merge unique
      const allNames = Array.from(new Set([...localNames, ...cloudNames]));
      return NextResponse.json({ datasets: allNames });
    }

    if (action === "view") {
      // 1. Get Cloud Info (Only if requested)
      let cloudFiles: any[] = [];
      if (searchParams.get("cloud") === "true") {
        const { ModalClient } = await import("modal");
        const modal = new ModalClient();
        const cls = await modal.cls.fromName("lora-dataset-manager", "DatasetManager");
        const instance = await cls.instance();
        cloudFiles = await instance.method("get_dataset_info").remote([datasetName]);
      }
      
      // 2. Get Local Info
      const localPath = path.join(DATASETS_ROOT, datasetName);
      let localFiles: any[] = [];
      try {
        const files = await fs.readdir(localPath);
        for (const file of files) {
          if (file.endsWith(".png")) {
             const base = file.replace(".png", "");
             let caption = "";
             try {
               caption = await fs.readFile(path.join(localPath, `${base}.txt`), "utf-8");
             } catch {}
             
             // We serve local images via a public path or base64. 
             // For simplicity in this demo studio, we can convert to base64 or assuming /datasets is served.
             // Let's use base64 for unified handling with cloud images in the UI.
             const imgBuffer = await fs.readFile(path.join(localPath, file));
             localFiles.push({ name: base, caption, data: imgBuffer.toString("base64"), source: "local" });
          }
        }
      } catch {}

      return NextResponse.json({ files: [...localFiles, ...cloudFiles] });
    }

    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error: any) {
    console.error("Dataset API Error:", error);
    return NextResponse.json({ error: error.message, stack: error.stack }, { status: 500 });
  }
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { action, datasetName, filename, image, caption } = body;

    const modal = new ModalClient();
    const cls = await modal.cls.fromName("lora-dataset-manager", "DatasetManager");
    const instance = await cls.instance();

    if (action === "sync") {
      const { files = [] } = body; // expects [{ name, image, caption }]
      const batchFiles = files.map((f: any) => ({
        name: f.name,
        // files[x].image is base64 string
        image_bytes: Buffer.from(f.image, "base64"),
        caption: f.caption || ""
      }));
      const res = await instance.method("save_images_batch").remote([datasetName, batchFiles]);
      return NextResponse.json({ success: true, msg: res });
    }

    if (action === "save") {
      // We save to BOTH for safety and ease of use as requested
      
      // 1. Save Local
      const localPath = path.join(DATASETS_ROOT, datasetName);
      await fs.mkdir(localPath, { recursive: true });
      const base64Data = image.split(",")[1] || image;
      await fs.writeFile(path.join(localPath, `${filename}.png`), Buffer.from(base64Data, "base64"));
      await fs.writeFile(path.join(localPath, `${filename}.txt`), caption);

      // 2. Save Cloud
      await instance.method("save_image").remote([
        datasetName,
        filename,
        Buffer.from(base64Data, "base64"),
        caption
      ]);

      return NextResponse.json({ success: true });
    }

    if (action === "export") {
        const zipBytes = await instance.method("export_zip").remote([datasetName]);
        return new Response(zipBytes, {
            headers: {
                "Content-Type": "application/zip",
                "Content-Disposition": `attachment; filename=${datasetName}.zip`
            }
        });
    }

    if (action === "rename") {
      const result = await (instance.method("rename_all_to_sequential") as any).remote([datasetName]);
      return NextResponse.json({ success: true, message: result });
    }

    if (action === "batch_caption") {
      const result = await (instance.method("create_batch_captions") as any).remote([datasetName, caption]);
      return NextResponse.json({ success: true, message: result });
    }

    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error) {
    console.error("Dataset POST Error:", error);
    return NextResponse.json({ error: "Internal Error" }, { status: 500 });
  }
}
