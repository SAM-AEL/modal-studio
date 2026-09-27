import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"

export async function GET() {
  try {
    const dir = path.join(process.cwd(), "modal_backend", "train")
    if (!fs.existsSync(dir)) {
      return NextResponse.json({ pipelines: [] })
    }

    const files = fs.readdirSync(dir)
    const pipelines = []

    for (const file of files) {
      if (!file.endsWith(".py")) continue

      const filePath = path.join(dir, file)
      const content = fs.readFileSync(filePath, "utf-8")

      // Match label comment: # label: "..."
      const labelMatch = content.match(/#\s*label:\s*"([^"]+)"/)
      const label = labelMatch ? labelMatch[1] : null

      // Match gpu name: gpu="..."
      const gpuMatch = content.match(/gpu\s*=\s*"([^"]+)"/)
      const gpu = gpuMatch ? gpuMatch[1].toUpperCase() : "Unknown"

      // Match class name: class [Name]:
      const classMatch = content.match(/class\s+([A-Za-z0-9_]+)/)
      const className = classMatch ? classMatch[1] : null

      if (className) {
        // Match app name: app = modal.App("...")
        const appMatch = content.match(/app\s*=\s*(?:modal\.)?App\(\s*["']([^"']+)["']/)
        const appName = appMatch ? appMatch[1] : file.replace(".py", "").replace(/_/g, "-")
        pipelines.push({
          id: file.replace(".py", ""),
          label: label || file,
          appName: appName,
          className: className,
          gpu: gpu
        })
      }
    }

    // Sort or filter if needed, e.g., only return if has @app.cls or is a Trainer
    return NextResponse.json({ pipelines })
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 })
  }
}
