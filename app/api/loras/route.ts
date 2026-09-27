import { ModalClient } from "modal";
import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { exec } from "child_process";

export const dynamic = "force-dynamic";

function log(msg: string) {
  try {
    fs.appendFileSync("/tmp/next_api_logs.txt", `[${new Date().toISOString()}] ${msg}\n`);
  } catch (e) {}
}

export async function GET() {
  log("--- GET /api/loras triggered ---");
  try {
    const modal = new ModalClient();
    const cls = await modal.cls.fromName("flux-lora-api", "LoraManager");
    const instance = await cls.instance();
    
    // 1. Get list from Modal cloud volume
    log("Fetching cloud loras...");
    const cloudLoras = await instance.method("list_loras").remote([]);
    log(`Cloud Loras: ${JSON.stringify(cloudLoras)}`);
    
    // 2. Scan local lora/ folder (optional local cache of .safetensors files).
    // Set LORA_LOCAL_DIR to sync a local folder into the Modal volume,
    // or leave unset to use cloud-only listing.
    const localLoraDir = process.env.LORA_LOCAL_DIR || "";
    log(`Scanning local dir: ${localLoraDir || "(disabled — set LORA_LOCAL_DIR to enable)"}`);
    let localLoras: string[] = [];

    if (localLoraDir && fs.existsSync(localLoraDir)) {
      const files = fs.readdirSync(localLoraDir);
      log(`Files in dir: ${JSON.stringify(files)}`);
      for (const file of files) {
        if (file.endsWith(".safetensors")) {
          // normalize name
          const name = file.replace("_lora.safetensors", "").replace(".safetensors", "");
          localLoras.push(name);
          
          // 3. Upload if not already in cloud list
          if (!cloudLoras.includes(name)) {
             log(`[Sync] Triggering upload for ${name}`);
             const localPath = path.join(localLoraDir, file);
             const volRemotePath = `/${name}/${name}_lora.safetensors`;
             
             // Run background upload via Modal CLI
             exec(`modal volume put lora-outputs ${localPath} ${volRemotePath}`, (err, stdout, stderr) => {
               if (err) log(`[Sync Error] Failed ${name}: ${err.message}`);
               else log(`[Sync Success] Sync complete for ${name}`);
             });
          }
        }
      }
    } else {
      if (localLoraDir) log("Local lora dir NOT found — skipping local sync.");
    }

    // Merge lists for instant UI listing (even while background upload is going on)
    const allLoras = Array.from(new Set([...cloudLoras, ...localLoras]));
    log(`Returning all Loras: ${JSON.stringify(allLoras)}`);
    return NextResponse.json({ loras: allLoras });

  } catch (error: any) {
    log(`--- Error: ${error.message}`);
    console.error("Loras API Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
