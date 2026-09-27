"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Sparkles, Download, Loader2, SlidersHorizontal, ChevronUp, ChevronDown, Cpu, Terminal, Upload, Image as ImageIcon
} from "lucide-react";

interface LogEntry {
  status: string;
  details?: string;
  timestamp: string;
}

export default function Img2ImgPage() {
  const [prompt, setPrompt] = useState("");
  const [baseImage, setBaseImage] = useState<string | null>(null);
  const [strength, setStrength] = useState<number>(0.6);
  
  const [settings, setSettings] = useState({
    seed: -1,
    guidanceScale: 3.5,
    steps: 28,
    width: 1024,
    height: 1024,
    batchSize: 1
  });

  const [showSettings, setShowSettings] = useState(false);
  
  const [images, setImages] = useState<{url: string, base64: string}[]>([]);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [currentStatus, setCurrentStatus] = useState<string>("");

  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      setBaseImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const generate = async () => {
    if (!prompt || !baseImage) return;

    setLoading(true);
    setImages([]);
    setLogs([]);
    setCurrentStatus("Initializing...");

    try {
      // Stripping data:image/png;base64, header for Modal API if needed
      const base64Clean = baseImage.split(",")[1] || baseImage;

      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          model: "flux-krea-img2img",
          image: base64Clean,
          strength: strength,
          seed: settings.seed,
          guidanceScale: settings.guidanceScale,
          steps: settings.steps,
          width: settings.width,
          height: settings.height,
          batchSize: settings.batchSize
        }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              
              if (data.status) {
                setLogs(prev => [...prev, { status: data.status, details: data.details, timestamp: new Date().toLocaleTimeString() }]);
                setCurrentStatus(data.details || data.status);
              }

              if (data.images) {
                setImages(data.images.map((base64: string) => ({
                  url: `data:image/png;base64,${base64}`,
                  base64
                })));
              }

              if (data.status === "error") {
                console.error("Generation error:", data.error);
                setLoading(false);
                return;
              }
            } catch (e) {
              console.error("JSON parse error:", e, "Line:", line);
            }
          }
        }
      }
    } catch (error) {
      console.error("Fetch error:", error);
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = (img: string, index: number) => {
    const a = document.createElement("a");
    a.href = img;
    a.download = `flux-img2img-${Date.now()}-${index}.png`;
    a.click();
  };

  const downloadAllImages = () => {
    images.forEach((img, i) => {
      setTimeout(() => downloadImage(img.url, i), i * 300);
    });
  };

  return (
    <div className="flex h-screen overflow-hidden bg-white text-black font-sans">
      
      {/* Sidebar Controls */}
      <aside className="w-[380px] border-r border-black/10 flex flex-col bg-[#fafafa] shrink-0">
        <div className="p-8 flex-1 overflow-y-auto no-scrollbar space-y-6">
          <div>
            <h1 className="text-xl font-black tracking-tight">Image Edge</h1>
            <p className="text-xs text-black/40 font-bold uppercase tracking-wider mt-0.5">Flux.1 Krea Img2Img</p>
          </div>

          {/* Image Upload Area */}
          <div className="space-y-2">
            <label className="block text-[11px] uppercase tracking-wider font-bold text-black/40">Base Image</label>
            {!baseImage ? (
              <label className="flex flex-col items-center justify-center w-full h-40 border border-dashed border-black/10 rounded-2xl cursor-pointer hover:bg-black/5 transition-all bg-white shadow-sm">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-5 h-5 text-black/40 mb-2" />
                  <p className="text-xs font-bold text-black/60">Upload reference image</p>
                  <p className="text-[10px] text-black/30 mt-1">Drag and drop or click</p>
                </div>
                <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
              </label>
            ) : (
              <div className="relative group">
                <img src={baseImage} alt="Base" className="w-full h-40 object-cover rounded-2xl border border-black/5 shadow-sm" />
                <button 
                  onClick={() => setBaseImage(null)} 
                  className="absolute top-2 right-2 bg-black/80 text-white p-1.5 rounded-full backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-all hover:bg-black text-[10px] font-bold"
                >
                  Change
                </button>
              </div>
            )}
          </div>

          {/* Prompt Context */}
          <div className="space-y-2">
            <label className="block text-[11px] uppercase tracking-wider font-bold text-black/40">Prompt Context</label>
            <textarea 
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="Describe modifications or environment..."
              className="w-full h-28 bg-white border border-black/10 rounded-2xl p-4 text-sm font-medium focus:outline-none focus:ring-1 focus:ring-black resize-none shadow-sm placeholder:text-black/20"
            />
          </div>

          {/* Strength Slider */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <label className="block text-[11px] uppercase tracking-wider font-bold text-black/40">Denoising Strength</label>
              <span className="text-xs font-mono font-bold text-black/60">x{strength}</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="0.9" 
              step="0.05" 
              value={strength} 
              onChange={e => setStrength(parseFloat(e.target.value))} 
              className="w-full accent-black h-1 bg-black/10 rounded-lg appearance-none cursor-pointer"
            />
            <p className="text-[10px] text-black/30 leading-tight">Lower = stays closer to original. Higher = allows more creative AI changes.</p>
          </div>

          {/* Advanced config button */}
          <div className="space-y-4">
            <button onClick={() => setShowSettings(!showSettings)} className="flex items-center justify-between w-full px-4 py-3 rounded-xl bg-white border border-black/10 text-black text-sm font-bold transition-all hover:bg-black hover:text-white">
              <div className="flex items-center gap-2"><SlidersHorizontal className="w-4 h-4" /> Config layout</div>
              {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            <AnimatePresence>
              {showSettings && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden space-y-4 pt-2">
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

        <div className="p-8 pt-0 border-t border-black/5 bg-[#fafafa]">
          <button onClick={generate} disabled={loading || !prompt || !baseImage} className="w-full bg-black text-white font-bold py-5 rounded-2xl active:scale-[0.98] transition-all disabled:opacity-30 shadow-xl shadow-black/10">
            {loading ? <Loader2 className="w-6 h-6 animate-spin mx-auto" /> : "Generate Modifications"}
          </button>
        </div>
      </aside>

      {/* Main View Area */}
      <div className="flex-1 flex flex-col bg-white overflow-hidden">
        <header className="h-16 border-b border-black/10 flex items-center justify-between px-10 shrink-0">
          <div className="flex items-center gap-4 text-sm font-medium">
            <span className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${loading ? "bg-yellow-400 animate-pulse" : "bg-green-500"}`} />
              {loading ? currentStatus : "Ready for generation"}
            </span>
          </div>
          <div className="flex items-center gap-3">
            {images.length > 1 && (
              <button onClick={downloadAllImages} className="flex items-center gap-1.5 text-xs text-black bg-black/5 px-3 py-1.5 rounded-full font-bold hover:bg-black hover:text-white transition-all shadow-sm">
                <Download className="w-3 h-3" /> Download All ({images.length})
              </button>
            )}
            <div className="flex items-center gap-1.5 text-xs text-black/50 bg-black/5 px-3 py-1.5 rounded-full font-bold">
              <Cpu className="w-3 h-3" /> A100 GPU Enabled
            </div>
          </div>
        </header>

        {/* Main Canvas */}
        <main className="flex-1 overflow-y-auto p-10 bg-[#f4f4f4] flex items-center justify-center">
          {images.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center text-black/20">
              <div className="w-16 h-16 rounded-3xl bg-black/5 flex items-center justify-center mb-4">
                <ImageIcon className="w-6 h-6" />
              </div>
              <p className="text-sm font-black tracking-tight">The Modification Canvas</p>
              <p className="text-[11px] font-bold uppercase tracking-wider mt-0.5">Upload and describe variations to begin</p>
            </div>
          )}

          {loading && images.length === 0 && (
            <div className="flex flex-col items-center justify-center">
              <div className="w-12 h-12 border-4 border-black/10 border-t-black rounded-full animate-spin mb-4" />
              <p className="text-sm font-black text-black">Computing Pixels...</p>
              <p className="text-[10px] font-bold uppercase tracking-wider text-black/40 mt-1 animate-pulse">{currentStatus}</p>
            </div>
          )}

          {images.length > 0 && (
            <div className={`grid gap-6 w-full h-full ${images.length === 1 ? 'place-items-center' : 'grid-cols-2'}`}>
              {images.map((img, i) => (
                <div key={i} className="relative group rounded-3xl overflow-hidden shadow-2xl border border-black/5 bg-white aspect-square flex items-center justify-center">
                  <img src={img.url} alt={`Generated ${i}`} className="w-full h-full object-cover" />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-all flex items-center justify-center gap-3 backdrop-blur-sm">
                    <button onClick={() => downloadImage(img.url, i)} className="bg-white/10 text-white px-6 py-3 rounded-xl font-bold border border-white/20 hover:bg-white/30 transition-all flex items-center gap-2">
                      <Download className="w-4 h-4" /> Download
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
