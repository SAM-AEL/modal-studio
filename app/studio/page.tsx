"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Sparkles, Download, Loader2, SlidersHorizontal, ChevronUp, ChevronDown, 
  Cpu, Terminal, DollarSign, AlertTriangle, CheckCircle2, Layers
} from "lucide-react";
import { DEFAULT_GPU_PRICING, GpuPricingSpec } from "../api/pricing/route";

interface LogEntry {
  status: string;
  details?: string;
  timestamp: string;
}

interface ModelDef {
  id: string;
  name: string;
  cfg: number;
  steps: number;
  gpu: string;
  requiredVram: number;
  supportsTransparency?: boolean;
  badge?: string;
}

const MODELS: ModelDef[] = [
  { 
    id: "ideogram-4-fp8", 
    name: "Ideogram 4.0 (Diffusers 4-Bit/FP8)", 
    cfg: 4.5, 
    steps: 25, 
    gpu: "H100", 
    requiredVram: 24, 
    supportsTransparency: true,
    badge: "⚡ 4.4x Faster on H100"
  },
  { 
    id: "ideogram-4-bf16", 
    name: "Ideogram 4.0 (Full Quality)", 
    cfg: 4.5, 
    steps: 35, 
    gpu: "H100", 
    requiredVram: 48, 
    supportsTransparency: true,
    badge: "Full Precision"
  },
  { id: "flux-dev", name: "Flux.1 Dev", cfg: 3.5, steps: 28, gpu: "A100-40GB", requiredVram: 28 },
  { id: "flux-lora", name: "Flux.1 with LoRA", cfg: 3.5, steps: 28, gpu: "A100-40GB", requiredVram: 32 },
  { id: "flux-schnell", name: "Flux.1 Schnell", cfg: 0.0, steps: 4, gpu: "A100-40GB", requiredVram: 24 },
  { id: "sdxl", name: "SDXL 1.0", cfg: 7.5, steps: 30, gpu: "A10G", requiredVram: 12 },
  { id: "sdxl-turbo", name: "SDXL Turbo", cfg: 0.0, steps: 1, gpu: "A10G", requiredVram: 12 },
  { id: "sd15", name: "Stable Diffusion 1.5", cfg: 7.5, steps: 30, gpu: "A10G", requiredVram: 6 },
  { id: "sd35-medium", name: "SD 3.5 Medium", cfg: 4.5, steps: 40, gpu: "A10G", requiredVram: 16 },
  { id: "kolors", name: "Kolors (Alibaba)", cfg: 5.0, steps: 25, gpu: "A10G", requiredVram: 20 },
  { id: "hunyuandit", name: "HunyuanDiT (Tencent)", cfg: 6.0, steps: 50, gpu: "A100-40GB", requiredVram: 30 },
  { id: "pixart-sigma", name: "PixArt-Sigma", cfg: 4.5, steps: 20, gpu: "A10G", requiredVram: 14 },
  { id: "auraflow", name: "AuraFlow v0.1", cfg: 3.5, steps: 50, gpu: "A100-40GB", requiredVram: 28 },
  { id: "kandinsky3", name: "Kandinsky 3.0", cfg: 3.0, steps: 25, gpu: "A10G", requiredVram: 16 },
];

const GPU_OPTIONS = [
  "T4",
  "L4",
  "A10G",
  "L40S",
  "A100-40GB",
  "A100-80GB",
  "H100"
];

const SLOT_SPRITE_PRESETS = [
  "Lucky 7 fiery wild symbol, golden beveled borders, sparkling gemstone accents, 3d slot art, transparent background",
  "Scatter golden pyramid bonus symbol with glowing hieroglyphs, vibrant jewelry, transparent background",
  "Stacks of shiny gold pirate doubloons and emeralds, casino slot symbol, sharp edges, transparent background",
  "Crown wild symbol with polished rubies, high multiplier badge, vector slot game style, transparent background"
];

export default function StudioPage() {
  const [prompt, setPrompt] = useState("");
  const [settings, setSettings] = useState({
    model: "ideogram-4-fp8",
    seed: -1,
    guidanceScale: 4.5,
    steps: 25,
    width: 1024,
    height: 1024,
    batchSize: 1
  });

  const [selectedGpu, setSelectedGpu] = useState("H100");
  const [transparentMode, setTransparentMode] = useState(true);
  const [pricing, setPricing] = useState<Record<string, GpuPricingSpec>>(DEFAULT_GPU_PRICING);

  const [showSettings, setShowSettings] = useState(false);
  const [availableLoras, setAvailableLoras] = useState<string[]>([]);
  const [selectedLoras, setSelectedLoras] = useState<{name: string, weight: number}[]>([]);
  
  const [images, setImages] = useState<{url: string, base64: string}[]>([]);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [currentStatus, setCurrentStatus] = useState<string>("");

  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    fetchLoras();
    fetchPricing();
  }, []);

  const fetchPricing = async () => {
    try {
      const res = await fetch("/api/pricing");
      const data = await res.json();
      if (data.pricing) {
        setPricing(data.pricing);
      }
    } catch (e) {
      console.error("Failed to load live Modal pricing, using cached fallback", e);
    }
  };
  
  const fetchLoras = async () => {
    try {
      const res = await fetch("/api/loras");
      const data = await res.json();
      if (data.loras) setAvailableLoras(data.loras);
    } catch (e) { console.error("Failed to fetch loras", e); }
  };

  const handleModelChange = (modelId: string) => {
    const model = MODELS.find(m => m.id === modelId);
    if (model) {
      setSettings({ 
        ...settings, 
        model: modelId, 
        guidanceScale: model.cfg, 
        steps: model.steps 
      });
      setSelectedGpu(model.gpu);
      if (model.supportsTransparency) {
        setTransparentMode(true);
      }
    }
  };

  const addLog = (status: string, details?: string) => {
    setLogs(prev => [...prev.slice(-9), {
      status,
      details,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    }]);
  };

  async function generate() {
    if (!prompt) return;
    setLoading(true); setImages([]); setLogs([]); setCurrentStatus("Waking up worker...");
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          prompt, 
          ...settings,
          transparent: transparentMode,
          ...(settings.model === "flux-lora" ? { loras: selectedLoras } : {})
        })
      });
      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const textDecoder = new TextDecoder();
      let resultBuffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        resultBuffer += textDecoder.decode(value, { stream: true });
        const lines = resultBuffer.split("\n");
        resultBuffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.status === "success") {
              const formattedImages = data.images.map((img: string) => ({ url: `data:image/png;base64,${img}`, base64: img }));
              setImages(formattedImages);
              addLog("Success", `Generated ${formattedImages.length} image(s)`);
              setCurrentStatus("Complete!");
            } else if (data.status === "error") {
              throw new Error(data.error);
            } else {
              addLog(data.status, data.details);
              setCurrentStatus(data.details || data.status);
            }
          } catch (e) { console.error("Error parsing NDJSON", e); }
        }
      }
    } catch (error) {
      addLog("Error", error instanceof Error ? error.message : "Generation failed");
      setCurrentStatus("Failed");
    } finally { setLoading(false); }
  }

  const downloadImage = (img: string, index: number) => {
    const a = document.createElement("a"); 
    a.href = img; 
    a.download = `slot-sprite-${Date.now()}-${index}.png`; 
    a.click();
  };

  const downloadAllImages = () => {
    images.forEach((img, i) => {
      setTimeout(() => {
        downloadImage(img.url, i);
      }, i * 300);
    });
  };

  const currentModel = MODELS.find(m => m.id === settings.model) || MODELS[0];
  const activeGpuSpec = pricing[selectedGpu] || pricing["H100"] || DEFAULT_GPU_PRICING["H100"];
  const isMemorySufficient = activeGpuSpec.vramGb >= currentModel.requiredVram;
  const memoryDelta = activeGpuSpec.vramGb - currentModel.requiredVram;

  return (
    <div className="flex h-screen overflow-hidden bg-white text-black font-sans">
      
      {/* Sidebar Controls */}
      <aside className="w-[410px] border-r border-black/10 flex flex-col bg-[#fafafa] shrink-0">
        <div className="p-7 flex-1 overflow-y-auto no-scrollbar space-y-5">
          <div>
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold tracking-tight">Creative Studio</h2>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-black/5 text-black/60">
                Modal Cloud
              </span>
            </div>
            <p className="text-xs text-black/40 font-medium tracking-tight mt-0.5">
              Slot Game Sprites & Multi-Model Inference
            </p>
          </div>

          {/* Model selection */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-[11px] uppercase tracking-wider font-bold text-black/40">Base Model</label>
              {currentModel.badge && (
                <span className="text-[10px] font-bold text-indigo-600 bg-indigo-50 border border-indigo-100 px-2 py-0.5 rounded-full">
                  {currentModel.badge}
                </span>
              )}
            </div>
            <select 
              value={settings.model} 
              onChange={e => handleModelChange(e.target.value)} 
              className="w-full bg-white border border-black/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-black appearance-none shadow-sm font-bold"
            >
              {MODELS.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          {/* Live GPU Pricing & VRAM Memory Monitor */}
          <div className="p-3.5 bg-white border border-black/10 rounded-2xl shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-bold text-black/70">
                <Cpu className="w-3.5 h-3.5 text-black/50" />
                <span>Compute Hardware</span>
              </div>
              <select 
                value={selectedGpu} 
                onChange={e => setSelectedGpu(e.target.value)}
                className="text-xs font-bold bg-[#f5f5f5] hover:bg-black/10 border border-black/10 rounded-lg px-2 py-1 focus:outline-none cursor-pointer"
              >
                {GPU_OPTIONS.map(gpu => (
                  <option key={gpu} value={gpu}>{gpu}</option>
                ))}
              </select>
            </div>

            {/* Pricing metrics */}
            <div className="grid grid-cols-2 gap-2 pt-1 border-t border-black/5">
              <div className="bg-[#f9f9f9] p-2.5 rounded-xl border border-black/5">
                <div className="text-[10px] font-medium text-black/40 uppercase tracking-wider flex items-center gap-1">
                  <DollarSign className="w-3 h-3 text-emerald-600" /> Per Hour
                </div>
                <div className="text-base font-extrabold text-black mt-0.5 font-mono">
                  ${activeGpuSpec.pricePerHour.toFixed(2)}
                  <span className="text-[11px] font-normal text-black/40">/hr</span>
                </div>
              </div>
              <div className="bg-[#f9f9f9] p-2.5 rounded-xl border border-black/5">
                <div className="text-[10px] font-medium text-black/40 uppercase tracking-wider flex items-center gap-1">
                  <DollarSign className="w-3 h-3 text-blue-600" /> Per Second
                </div>
                <div className="text-sm font-extrabold text-black mt-0.5 font-mono">
                  ${activeGpuSpec.pricePerSec.toFixed(6)}
                  <span className="text-[10px] font-normal text-black/40">/s</span>
                </div>
              </div>
            </div>

            {/* Memory Adequacy Gauge */}
            <div className={`p-2.5 rounded-xl border flex items-start gap-2 ${
              isMemorySufficient 
                ? "bg-emerald-50/70 border-emerald-200 text-emerald-950" 
                : "bg-amber-50/80 border-amber-300 text-amber-950"
            }`}>
              {isMemorySufficient ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              )}
              <div className="text-xs">
                <div className="font-bold flex items-center gap-1.5">
                  <span>{isMemorySufficient ? "Memory Sufficient" : "Insufficient VRAM (OOM Risk)"}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-black/5">
                    {activeGpuSpec.vramGb}GB / ~{currentModel.requiredVram}GB req
                  </span>
                </div>
                <p className="text-[11px] opacity-75 mt-0.5">
                  {isMemorySufficient 
                    ? `Comfortable headroom (+${memoryDelta}GB) for 2K resolution & tiling.`
                    : `Needs at least ${currentModel.requiredVram}GB VRAM. Switch GPU to A100/H100.`}
                </p>
              </div>
            </div>
          </div>

          {/* Slot Game Sprite Transparency Mode */}
          <div className="bg-white border border-black/10 rounded-2xl p-3.5 shadow-sm space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-black flex items-center gap-1.5 cursor-pointer">
                <Layers className="w-3.5 h-3.5 text-indigo-600" />
                <span>Transparent Sprite Mode (RGBA)</span>
              </label>
              <input 
                type="checkbox" 
                checked={transparentMode} 
                onChange={e => setTransparentMode(e.target.checked)} 
                className="w-4 h-4 accent-black rounded cursor-pointer"
              />
            </div>
            <p className="text-[11px] text-black/50 leading-relaxed">
              Generates slot symbols on a native alpha channel without background halos or edge fringing.
            </p>

            {/* Quick Presets for Slot Assets */}
            <div className="pt-2 border-t border-black/5">
              <span className="text-[10px] uppercase font-bold text-black/40 block mb-1.5">
                Quick Slot Asset Prompts
              </span>
              <div className="flex flex-wrap gap-1.5">
                {["Lucky 7 Wild", "Scatter Pyramid", "Gold Doubloons", "Gem Crown"].map((label, idx) => (
                  <button
                    key={label}
                    type="button"
                    onClick={() => setPrompt(SLOT_SPRITE_PRESETS[idx])}
                    className="text-[10px] font-semibold bg-[#f0f0f0] hover:bg-black hover:text-white px-2 py-1 rounded-lg transition-colors"
                  >
                    + {label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Prompt Area */}
          <div className="space-y-2">
            <label className="block text-[11px] uppercase tracking-wider font-bold text-black/40">Prompt Context</label>
            <textarea 
              value={prompt} 
              onChange={e => setPrompt(e.target.value)} 
              className="w-full bg-white border border-black/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-black shadow-sm resize-none h-28" 
              placeholder="Describe your slot symbol or graphic..." 
            />
          </div>

          {/* LoRA Selection - only for flux-lora */}
          {settings.model === "flux-lora" && (
            <div className="space-y-3">
              <label className="block text-[11px] uppercase tracking-wider font-bold text-black/40">Apply LoRAs</label>
              <select 
                disabled={availableLoras.filter(l => !selectedLoras.some(sl => sl.name === l)).length === 0}
                onChange={e => {
                  const name = e.target.value;
                  if (name && !selectedLoras.some(l => l.name === name)) {
                    setSelectedLoras([...selectedLoras, { name, weight: 1.0 }]);
                  }
                  e.target.value = "";
                }} 
                className="w-full bg-white border border-black/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-black appearance-none shadow-sm font-bold disabled:opacity-50"
              >
                <option value="">
                  {availableLoras.filter(l => !selectedLoras.some(sl => sl.name === l)).length === 0 
                    ? "No other LoRAs found" 
                    : "+ Add LoRA..."}
                </option>
                {availableLoras
                  .filter(l => !selectedLoras.some(sl => sl.name === l))
                  .map(l => <option key={l} value={l}>{l}</option>)
                }
              </select>
              <div className="space-y-2">
                {selectedLoras.map((lora, i) => (
                  <div key={i} className="bg-white border border-black/10 rounded-xl p-3 space-y-2 shadow-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold truncate">{lora.name}</span>
                      <button onClick={() => setSelectedLoras(selectedLoras.filter((_, idx) => idx !== i))} className="text-red-500 hover:text-red-600 text-[10px] font-bold">Remove</button>
                    </div>
                    <div className="flex items-center gap-2">
                      <input type="range" min="0" max="2" step="0.1" value={lora.weight} onChange={e => {
                        const newLoras = [...selectedLoras];
                        newLoras[i].weight = parseFloat(e.target.value);
                        setSelectedLoras(newLoras);
                      }} className="flex-1 accent-black h-1 bg-black/10 rounded-lg appearance-none cursor-pointer" />
                      <span className="text-xs font-mono w-8 text-right">x{lora.weight.toFixed(1)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Advanced config button */}
          <div className="space-y-4">
            <button onClick={() => setShowSettings(!showSettings)} className="flex items-center justify-between w-full px-4 py-3 rounded-xl bg-white border border-black/10 text-black text-sm font-bold transition-all hover:bg-black hover:text-white">
              <div className="flex items-center gap-2"><SlidersHorizontal className="w-4 h-4" /> Config & Steps</div>
              {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            <AnimatePresence>
              {showSettings && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden space-y-4 pt-2">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase font-bold text-black/40">Inference Steps</label>
                      <input type="number" min="1" max="100" value={settings.steps} onChange={e => setSettings({ ...settings, steps: parseInt(e.target.value) || 30 })} className="w-full bg-white border border-black/10 rounded-lg px-3 py-2 text-xs font-mono" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase font-bold text-black/40">Guidance Scale</label>
                      <input type="number" step="0.5" min="0" max="20" value={settings.guidanceScale} onChange={e => setSettings({ ...settings, guidanceScale: parseFloat(e.target.value) || 4.5 })} className="w-full bg-white border border-black/10 rounded-lg px-3 py-2 text-xs font-mono" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase font-bold text-black/40">Seed</label>
                      <input type="number" value={settings.seed} onChange={e => setSettings({ ...settings, seed: parseInt(e.target.value) })} className="w-full bg-white border border-black/10 rounded-lg px-3 py-2 text-xs font-mono" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase font-bold text-black/40">Batch Size</label>
                      <input type="number" min="1" max="10" value={settings.batchSize} onChange={e => setSettings({ ...settings, batchSize: Math.min(10, Math.max(1, parseInt(e.target.value) || 1)) })} className="w-full bg-white border border-black/10 rounded-lg px-3 py-2 text-xs font-mono" />
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div className="p-7 pt-0 border-t border-black/5 bg-[#fafafa]">
          <button onClick={generate} disabled={loading || !prompt} className="w-full bg-black text-white font-bold py-4 rounded-2xl active:scale-[0.98] transition-all disabled:opacity-30 shadow-xl shadow-black/10 flex items-center justify-center gap-2">
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5" />}
            <span>{loading ? "Generating Asset…" : "Generate Sprite"}</span>
          </button>
        </div>
      </aside>

      {/* Main View Area */}
      <div className="flex-1 flex flex-col bg-white overflow-hidden">
        <header className="h-16 border-b border-black/10 flex items-center justify-between px-8 shrink-0">
          <div className="flex items-center gap-4 text-sm font-medium">
            <span className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${loading ? "bg-yellow-400 animate-pulse" : "bg-green-500"}`} />
              {loading ? currentStatus : "Ready for generation"}
            </span>
            <div className="w-px h-4 bg-black/10 mx-1" />
            <span className="font-semibold text-black/70">{currentModel.name}</span>
          </div>
          <div className="flex items-center gap-3">
            {images.length > 1 && (
              <button onClick={downloadAllImages} className="flex items-center gap-1.5 text-xs text-black bg-black/5 px-3 py-1.5 rounded-full font-bold hover:bg-black hover:text-white transition-all shadow-sm">
                <Download className="w-3 h-3" /> Download All ({images.length})
              </button>
            )}
            <div className="flex items-center gap-2 text-xs bg-black/5 px-3 py-1.5 rounded-full font-bold">
              <Cpu className="w-3.5 h-3.5 text-black/60" />
              <span>{activeGpuSpec.name}</span>
              <span className="text-black/40">|</span>
              <span className="text-emerald-700 font-mono">${activeGpuSpec.pricePerHour.toFixed(2)}/hr</span>
            </div>
          </div>
        </header>

        {/* Main Canvas */}
        <div className="flex-1 overflow-y-auto p-10 custom-scrollbar bg-[#f2f2f2] flex flex-col h-full">
          <div className="flex-1 flex flex-col items-center justify-center">
            {images.length > 0 ? (
              <div className={`grid gap-8 ${images.length === 1 ? "max-w-2xl mx-auto" : "grid-cols-2"}`}>
                {images.map((img, i) => (
                  <motion.div 
                    key={i} 
                    initial={{ opacity: 0, scale: 0.95 }} 
                    animate={{ opacity: 1, scale: 1 }} 
                    className="group relative rounded-[2rem] overflow-hidden shadow-2xl border border-black/10 transition-all"
                    style={{
                      backgroundImage: transparentMode 
                        ? "repeating-conic-gradient(#e5e7eb 0% 25%, #f9fafb 0% 50%)" 
                        : "none",
                      backgroundSize: "20px 20px"
                    }}
                  >
                    <img src={img.url} className="w-full h-auto object-contain block" alt="Generated game asset" />
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center gap-4 backdrop-blur-sm pointer-events-none group-hover:pointer-events-auto">
                      <button onClick={() => downloadImage(img.url, i)} className="bg-white text-black px-6 py-3 rounded-xl font-bold hover:bg-black hover:text-white transition-all flex items-center gap-2 shadow-lg">
                        <Download className="w-4 h-4" /> Download PNG (RGBA)
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center space-y-6">
                {loading ? (
                  <div className="space-y-6">
                    <div className="w-24 h-24 border-t-4 border-black rounded-full mx-auto" style={{ animation: "spin 1.2s linear infinite" }} />
                    <h3 className="text-2xl font-bold tracking-tight">Rendering High-Fidelity Pixels…</h3>
                    <p className="text-xs text-black/50 font-mono">Running on {activeGpuSpec.name} ({activeGpuSpec.vramGb}GB VRAM)</p>
                  </div>
                ) : (
                  <>
                    <div 
                      className="p-14 rounded-[3.5rem] border border-black/5 shadow-inner"
                      style={{
                        backgroundImage: "repeating-conic-gradient(#ebebeb 0% 25%, #f7f7f7 0% 50%)",
                        backgroundSize: "24px 24px"
                      }}
                    >
                      <Sparkles className="w-16 h-16 text-black/20" />
                    </div>
                    <div className="space-y-2 max-w-md">
                      <h3 className="text-2xl font-bold tracking-tight">Slot Asset Canvas</h3>
                      <p className="text-black/50 text-xs font-medium leading-relaxed">
                        Prompt for wild symbols, scatter gems, gold coins, or UI badges. Transparent sprites render on native alpha channels.
                      </p>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Engine logs list pinned at bottom */}
          <div className="h-36 border-t border-black/10 flex flex-col shrink-0 bg-white -m-10 mt-10">
            <div className="px-6 py-2 border-b border-black/5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Terminal className="w-3 h-3 text-black/40" />
                <span className="text-[10px] uppercase font-bold text-black/40">Engine Stream</span>
              </div>
              <span className="text-[10px] font-mono text-black/40">
                Modal Serverless · {activeGpuSpec.name}
              </span>
            </div>
            <div className="flex-1 p-5 font-mono text-[11px] overflow-y-auto custom-scrollbar">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-4 mb-1">
                  <span className="text-black/20 shrink-0">[{log.timestamp}]</span>
                  <span className="font-bold w-20 uppercase tracking-tighter shrink-0">{log.status}</span>
                  <span className="text-black/60 truncate">{log.details}</span>
                </div>
              ))}
              {!logs.length && <div className="text-black/20 italic text-[10px]">Awaiting cycle start…</div>}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
