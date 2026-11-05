"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import {
  Sparkles, Download, Loader2, Cpu, Terminal, Upload,
  Image as ImageIcon, Clapperboard,
} from "lucide-react";

interface LogEntry {
  status: string;
  details?: string;
  timestamp: string;
}

const VARIANTS = [
  {
    id: "wan-animate-fp8",
    name: "Wan 2.2 Animate 14B — FP8",
    desc: "Quantized transformer (~17GB). Fits H100 comfortably, fastest.",
    steps: 20, cfg: 1.0, gpu: "H100", badge: "⚡ Recommended",
  },
  {
    id: "wan-animate-bf16",
    name: "Wan 2.2 Animate 14B — BF16",
    desc: "Full-precision (~28GB+). Max quality, slower, H100 only.",
    steps: 20, cfg: 1.0, gpu: "H100", badge: "Full Precision",
  },
];

export default function AnimatePage() {
  const [prompt, setPrompt] = useState("");
  const [neg, setNeg] = useState("");
  const [variant, setVariant] = useState("wan-animate-fp8");
  const [refImg, setRefImg] = useState<string | null>(null);
  const [driveVid, setDriveVid] = useState<string | null>(null);
  const [driveName, setDriveName] = useState("");
  const [steps, setSteps] = useState(20);
  const [cfg, setCfg] = useState(1.0);
  const [width, setWidth] = useState(1280);
  const [height, setHeight] = useState(720);
  const [seg, setSeg] = useState(77);
  const [fps, setFps] = useState(24);
  const [seed, setSeed] = useState(-1);
  const [mode, setMode] = useState<"video" | "sprite">("sprite");
  const [video, setVideo] = useState<string | null>(null);
  const [sheet, setSheet] = useState<string | null>(null);
  const [meta, setMeta] = useState<any>(null);
  const [outSize, setOutSize] = useState(512);
  const [cols, setCols] = useState(4);
  const [sheetFmt, setSheetFmt] = useState("png");
  const [tab, setTab] = useState<"sheet" | "preview">("sheet");
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [status, setStatus] = useState("");
  const logEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => { logEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [logs]);
  const readFile = (f: File): Promise<string> => new Promise((res, rej) => {
    const r = new FileReader();
    r.onloadend = () => res(r.result as string);
    r.onerror = rej;
    r.readAsDataURL(f);
  });
  const generate = async () => {
    if (!refImg || !driveVid || loading) return;
    setLoading(true); setVideo(null); setSheet(null); setMeta(null);
    setLogs([]); setStatus("Initializing...");
    try {
      const url = mode === "sprite" ? "/api/sprite" : "/api/animate";
      const res = await fetch(url, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mode === "sprite" ? {
          refImage: refImg.split(",")[1] || refImg,
          drivingVideo: driveVid.split(",")[1] || driveVid,
          prompt, negativePrompt: neg,
          size: Math.max(height, width), steps, guidanceScale: cfg,
          fps, seed, outSize, cols, greenBg: true, format: sheetFmt,
        } : {
          refImage: refImg.split(",")[1] || refImg,
          drivingVideo: driveVid.split(",")[1] || driveVid,
          prompt, negativePrompt: neg, model: variant,
          height, width, steps, guidanceScale: cfg,
          segmentFrames: seg, fps, seed,
        }),
      });
      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const d = JSON.parse(line);
            if (d.status) {
              setLogs((p) => [...p, { status: d.status, details: d.details, timestamp: new Date().toLocaleTimeString() }]);
              setStatus(d.details || d.status);
            }
            if (d.video) setVideo(`data:video/mp4;base64,${d.video}`);
            if (d.preview) setVideo(`data:video/mp4;base64,${d.preview}`);
            if (d.sheet) setSheet(`data:image/${d.meta?.format === "webp" ? "webp" : "png"};base64,${d.sheet}`);
            if (d.meta && d.sheet) setMeta(d.meta);
            if (d.status === "error") throw new Error(d.error || "failed");
          } catch (e) { if (e instanceof Error && e.message !== "failed" && (e as SyntaxError).name !== "SyntaxError") throw e; }
        }
      }
    } catch (e) {
      setLogs((p) => [...p, { status: "error", details: e instanceof Error ? e.message : String(e), timestamp: new Date().toLocaleTimeString() }]);
      setStatus("Failed");
    } finally { setLoading(false); }
  };
  return (
    <div className="flex h-screen overflow-hidden bg-white font-sans">
      <aside className="w-[380px] border-r border-black/10 flex flex-col bg-[#fafafa] shrink-0">
        <div className="p-8 border-b border-black/10">
          <div className="flex items-center gap-2">
            <Clapperboard className="w-4 h-4" />
            <h1 className="text-xl font-black tracking-tight">Wan Animate</h1>
          </div>
          <p className="text-[11px] text-black/40 font-medium mt-1">ref image + driving video to slot sprite</p>
          <div className="flex gap-2 mt-4">
            <button onClick={() => setMode("sprite")}
              className={`flex-1 text-xs font-bold py-2 rounded-xl transition-all ${mode === "sprite" ? "bg-black text-white" : "bg-black/5 hover:bg-black/10"}`}>
              Slot Sprite
            </button>
            <button onClick={() => setMode("video")}
              className={`flex-1 text-xs font-bold py-2 rounded-xl transition-all ${mode === "video" ? "bg-black text-white" : "bg-black/5 hover:bg-black/10"}`}>
              Raw MP4
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-8 space-y-6 custom-scrollbar">
          <div className="space-y-2">
            {VARIANTS.map((v) => (
              <button key={v.id} onClick={() => { setVariant(v.id); setSteps(v.steps); setCfg(v.cfg); }}
                className={`w-full text-left p-4 rounded-2xl border transition-all ${variant === v.id ? "border-black bg-black text-white shadow-lg" : "border-black/10 bg-white hover:border-black/30"}`}>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-bold">{v.name}</span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${variant === v.id ? "bg-white/20" : "bg-black/5"}`}>{v.badge}</span>
                </div>
                <p className={`text-[11px] mt-1 ${variant === v.id ? "text-white/60" : "text-black/40"}`}>{v.desc}</p>
              </button>
            ))}
          </div>
          <div>
            <label className="text-[11px] font-bold uppercase tracking-wider text-black/40">Reference character</label>
            <label className="mt-2 flex items-center gap-3 p-4 rounded-2xl border border-dashed border-black/20 bg-white cursor-pointer hover:border-black/50 transition-all">
              <ImageIcon className="w-5 h-5 text-black/40" />
              <span className="text-xs font-bold">{refImg ? "Change image" : "Upload PNG/JPG"}</span>
              <input type="file" accept="image/*" className="hidden" onChange={async (e) => { const f = e.target.files?.[0]; if (f) setRefImg(await readFile(f)); }} />
            </label>
            {refImg && <img src={refImg} alt="ref" className="mt-2 rounded-xl border border-black/10 w-full object-cover" />}
          </div>
          <div>
            <label className="text-[11px] font-bold uppercase tracking-wider text-black/40">Driving video (.mp4)</label>
            <label className="mt-2 flex items-center gap-3 p-4 rounded-2xl border border-dashed border-black/20 bg-white cursor-pointer hover:border-black/50 transition-all">
              <Upload className="w-5 h-5 text-black/40" />
              <span className="text-xs font-bold">{driveName || "Upload motion video"}</span>
              <input type="file" accept="video/mp4,video/*" className="hidden" onChange={async (e) => { const f = e.target.files?.[0]; if (f) { setDriveName(f.name); setDriveVid(await readFile(f)); } }} />
            </label>
            {driveVid && <video src={driveVid} className="mt-2 rounded-xl border border-black/10 w-full" controls muted />}
          </div>
          <div>
            <label className="text-[11px] font-bold uppercase tracking-wider text-black/40">Motion prompt</label>
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="e.g. character dancing, natural motion, high quality" className="mt-2 w-full bg-white border border-black/10 rounded-2xl p-4 text-sm outline-none focus:border-black/40 min-h-[80px]" />
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <label>Steps<input type="number" value={steps} onChange={(e) => setSteps(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
            <label>CFG<input type="number" step="0.1" value={cfg} onChange={(e) => setCfg(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
            {mode === "sprite" ? (
              <>
                <label>Cell px<input type="number" value={outSize} onChange={(e) => setOutSize(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
                <label>Columns<input type="number" value={cols} onChange={(e) => setCols(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
                <label>FPS<input type="number" value={fps} onChange={(e) => setFps(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
                <label>Format<select value={sheetFmt} onChange={(e) => setSheetFmt(e.target.value)} className="w-full border rounded-lg p-2 mt-1"><option value="png">PNG</option><option value="webp">WebP</option></select></label>
              </>
            ) : (
              <>
                <label>Width<input type="number" value={width} onChange={(e) => setWidth(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
                <label>Height<input type="number" value={height} onChange={(e) => setHeight(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
                <label>Seg frames<input type="number" value={seg} onChange={(e) => setSeg(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
                <label>FPS<input type="number" value={fps} onChange={(e) => setFps(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
              </>
            )}
            <label className="col-span-2">Seed (-1 random)<input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
            <label className="col-span-2">Negative<input value={neg} onChange={(e) => setNeg(e.target.value)} className="w-full border rounded-lg p-2 mt-1" /></label>
          </div>
          <button onClick={generate} disabled={!refImg || !driveVid || loading} className="w-full bg-black text-white font-bold py-4 rounded-2xl hover:bg-black/80 disabled:opacity-30 transition-all flex items-center justify-center gap-2 text-sm">
            {loading ? <><Loader2 className="w-4 h-4 animate-spin" /> Animating...</> : <><Sparkles className="w-4 h-4" /> {mode === "sprite" ? "Export Slot Sprite" : "Animate Character"}</>}
          </button>
          <p className="text-[10px] text-black/30 leading-relaxed">Sprite: green-screen Wan + BiRefNet + temporal smooth + aligned cells. Driving video is used for pose+face.</p>
        </div>
      </aside>
      <div className="flex-1 flex flex-col bg-white overflow-hidden">
        <header className="h-16 border-b border-black/10 flex items-center justify-between px-10 shrink-0">
          <span className="flex items-center gap-2 text-sm font-medium">
            <div className={`w-2 h-2 rounded-full ${loading ? "bg-yellow-400 animate-pulse" : "bg-green-500"}`} />
            {loading ? status : "Ready"}
          </span>
          <div className="flex items-center gap-3">
            {sheet && <a href={sheet} download={`slot-sprite.${meta?.format || "png"}`} className="flex items-center gap-1.5 text-xs bg-black text-white px-3 py-1.5 rounded-full font-bold hover:bg-black/80 transition-all"><Download className="w-3 h-3" /> Sprite {meta ? `${meta.cols}x${meta.rows} · ${meta.cell}px` : ""}</a>}
            {video && !sheet && <a href={video} download="wan-animate.mp4" className="flex items-center gap-1.5 text-xs bg-black/5 px-3 py-1.5 rounded-full font-bold hover:bg-black hover:text-white transition-all"><Download className="w-3 h-3" /> Download MP4</a>}
            {sheet && video && <a href={video} download="wan-preview.mp4" className="flex items-center gap-1.5 text-xs bg-black/5 px-3 py-1.5 rounded-full font-bold hover:bg-black hover:text-white transition-all"><Download className="w-3 h-3" /> MP4</a>}
            <div className="flex items-center gap-1.5 text-xs text-black/50 bg-black/5 px-3 py-1.5 rounded-full font-bold"><Cpu className="w-3 h-3" /> H100 GPU</div>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-10 bg-[#f4f4f4] flex items-center justify-center">
          {sheet ? (
            <div className="flex flex-col items-center gap-4 max-w-full">
              <div className="flex gap-2">
                <button onClick={() => setTab("sheet")} className={`text-xs font-bold px-4 py-2 rounded-full ${tab === "sheet" ? "bg-black text-white" : "bg-white border border-black/10"}`}>Spritesheet{meta ? ` ${meta.cols}x${meta.rows}` : ""}</button>
                <button onClick={() => setTab("preview")} className={`text-xs font-bold px-4 py-2 rounded-full ${tab === "preview" ? "bg-black text-white" : "bg-white border border-black/10"}`}>Preview MP4</button>
              </div>
              {tab === "sheet" ? (
                <motion.img key={sheet.slice(0, 64)} src={sheet} alt="spritesheet" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="max-h-[70vh] max-w-full rounded-3xl shadow-2xl border border-black/5"
                  style={{ backgroundImage: "repeating-conic-gradient(#ddd 0% 25%, #fff 0% 50%)", backgroundSize: "24px 24px" }} />
              ) : (
                video && <motion.video key={video.slice(0, 64)} src={video} controls autoPlay loop muted initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-h-[70vh] max-w-full rounded-3xl shadow-2xl border border-black/5" />
              )}
              {meta && <p className="text-[11px] font-mono text-black/50">{meta.frames}f · {meta.cols}x{meta.rows} · {meta.cell}px cells · {meta.sheet_w}x{meta.sheet_h} · {meta.fps}fps · seed {meta.seed}</p>}
            </div>
          ) : video ? (
            <motion.video key={video.slice(0, 64)} src={video} controls autoPlay loop muted initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-h-full max-w-full rounded-3xl shadow-2xl border border-black/5" />
          ) : loading ? (
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="w-12 h-12 border-4 border-black/10 border-t-black rounded-full animate-spin" />
              <p className="text-sm font-black">Animating... (minutes on first run)</p>
              <p className="text-[10px] font-bold uppercase tracking-wider text-black/40 animate-pulse">{status}</p>
            </div>
          ) : (
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="p-14 rounded-[3.5rem] border border-black/5 shadow-inner bg-white"><Clapperboard className="w-16 h-16 text-black/20" /></div>
              <h3 className="text-2xl font-bold tracking-tight">Animate Canvas</h3>
              <p className="text-black/50 text-xs font-medium max-w-md">Upload a character image + driving mp4, pick FP8 or BF16, and generate.</p>
            </div>
          )}
        </main>
        <div className="h-36 border-t border-black/10 flex flex-col shrink-0 bg-white">
          <div className="px-6 py-2 border-b border-black/5 flex items-center justify-between">
            <div className="flex items-center gap-2"><Terminal className="w-3 h-3 text-black/40" /><span className="text-[10px] uppercase font-bold text-black/40">Engine Stream</span></div>
            <span className="text-[10px] font-mono text-black/40">Modal Serverless H100</span>
          </div>
          <div className="flex-1 p-5 font-mono text-[11px] overflow-y-auto custom-scrollbar">
            {logs.map((l, i) => (
              <div key={i} className="flex gap-4 mb-1"><span className="text-black/20 shrink-0">[{l.timestamp}]</span><span className="font-bold w-20 uppercase tracking-tighter shrink-0">{l.status}</span><span className="text-black/60 truncate">{l.details}</span></div>
            ))}
            {!logs.length && <div className="text-black/20 italic text-[10px]">Awaiting cycle start...</div>}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
