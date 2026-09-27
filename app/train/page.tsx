"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Loader2, Zap, Play, AlertCircle, ArrowDownToLine, SlidersHorizontal, ChevronUp, ChevronDown, Terminal
} from "lucide-react";

interface TrainEvent {
  type: "info" | "log" | "progress" | "done" | "error" | "warning";
  msg: string;
  step?: number;
  total?: number;
  pct?: number;
}

export default function TrainPage() {
  const [trainConfig, setTrainConfig] = useState({
    datasetName: "my-concept",
    triggerWord: "my_trigger_word",
    trainerApp: "", 
    className: "",
    steps: 1000,
    rank: 32,
    lr: 0.0001,
    resolution: 1024,
    batchSize: 2,
    gradientAccumulation: 4,
  });

  const [trainLogs, setTrainLogs] = useState<TrainEvent[]>([]);
  const [trainProgress, setTrainProgress] = useState<{ step: number; total: number; pct: number } | null>(null);
  const [trainRunning, setTrainRunning] = useState(false);
  const [trainDone, setTrainDone] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  const [datasets, setDatasets] = useState<string[]>([]);
  const [pipelines, setPipelines] = useState<any[]>([]);
  const [finishedLoras, setFinishedLoras] = useState<string[]>([]);
  const [activeCallId, setActiveCallId] = useState<string | null>(null);
  
  const trainLogEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    trainLogEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [trainLogs]);

  useEffect(() => {
    fetchDatasets();
    fetchPipelines();
    fetchFinishedLoras();
  }, []);

  useEffect(() => {
    if (pipelines.length > 0 && !trainConfig.trainerApp) {
      setTrainConfig(prev => ({
        ...prev,
        trainerApp: pipelines[0].appName,
        className: pipelines[0].className
      }));
    }
  }, [pipelines]);

  const fetchDatasets = async () => {
    try {
      const res = await fetch("/api/dataset?action=list");
      const data = await res.json();
      if (data.datasets) setDatasets(data.datasets);
    } catch (e) { console.error("Failed to fetch datasets", e); }
  };

  const fetchPipelines = async () => {
    try {
      const res = await fetch("/api/pipelines");
      const data = await res.json();
      if (data.pipelines) setPipelines(data.pipelines);
    } catch (e) { console.error("Failed to fetch pipelines", e); }
  };

  const fetchFinishedLoras = async () => {
    try {
      const res = await fetch("/api/train?action=list");
      const data = await res.json();
      if (data.outputs) setFinishedLoras(data.outputs);
    } catch {}
  };

  const downloadLora = async (name: string) => {
    const res = await fetch("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "download", datasetName: name }) });
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `${name}_lora.safetensors`; a.click();
  };

  const startTraining = async () => {
    setTrainRunning(true);
    setTrainDone(false);
    setTrainError(null);
    setTrainLogs([]);
    setTrainProgress(null);

    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "start", ...trainConfig }),
      });
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const ev: TrainEvent = JSON.parse(line);
            if (ev.type === "info" && (ev as any).callId) {
              setActiveCallId((ev as any).callId);
            }
            if (ev.type === "progress" && ev.step !== undefined && ev.total !== undefined) {
              setTrainProgress({ step: ev.step, total: ev.total, pct: ev.pct ?? 0 });
            }
            setTrainLogs(prev => [...prev, ev]);
            if (ev.type === "done") { setTrainDone(true); fetchFinishedLoras(); setActiveCallId(null); }
            if (ev.type === "error") { setTrainError(ev.msg); setActiveCallId(null); }
          } catch {}
        }
      }
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : String(err));
    } finally {
      setTrainRunning(false);
      setActiveCallId(null);
    }
  };

  const stopTraining = async () => {
    if (!activeCallId) return;
    if (!confirm("Are you sure you want to kill this training job? All progress will be lost and the instance will be terminated.")) return;
    
    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "stop", callId: activeCallId }),
      });
      if (res.ok) {
        setTrainLogs(prev => [...prev, { type: "error", msg: "Training terminated by user." }]);
        setTrainRunning(false);
        setActiveCallId(null);
      }
    } catch (e) { console.error("Stop error", e); }
  };

  const _log = (msg: string, kind = "info") => ({ type: kind as any, msg });

  return (
    <div className="flex h-screen overflow-hidden bg-white text-black font-sans">
      
      {/* Sidebar Controls */}
      <aside className="w-[380px] border-r border-black/10 flex flex-col bg-[#fafafa] shrink-0">
        <div className="p-8 flex-1 overflow-y-auto no-scrollbar space-y-8">
          <div>
            <h2 className="text-xl font-bold tracking-tight mb-1">Train LoRA</h2>
            <p className="text-xs text-black/40 font-medium tracking-tight">Fine-tune Flux.1 Dev on High-Performance GPU</p>
          </div>

          {/* Trainer Type selector */}
          <div className="space-y-3">
            <label className="block text-[11px] uppercase tracking-wider font-bold text-black/40">Training Pipeline</label>
            <select value={trainConfig.trainerApp} onChange={e => {
              const p = pipelines.find(p => p.appName === e.target.value);
              setTrainConfig({ 
                ...trainConfig, 
                trainerApp: e.target.value,
                className: p ? p.className : "" 
              });
            }} className="w-full bg-white border border-black/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-purple-400 appearance-none shadow-sm font-bold">
              {pipelines.map(p => <option key={p.id} value={p.appName}>{p.label}</option>)}
            </select>
          </div>

          {/* Dataset selector */}
          <div className="space-y-3">
            <label className="block text-[11px] uppercase tracking-wider font-bold text-black/40">Dataset</label>
            <select value={trainConfig.datasetName} onChange={e => setTrainConfig({ ...trainConfig, datasetName: e.target.value })} className="w-full bg-white border border-black/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-purple-400 appearance-none shadow-sm font-bold">
              {datasets.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          {/* Trigger word */}
          <div className="space-y-3">
            <label className="block text-[11px] uppercase tracking-wider font-bold text-black/40">Trigger Word</label>
            <input value={trainConfig.triggerWord} onChange={e => setTrainConfig({ ...trainConfig, triggerWord: e.target.value })} className="w-full bg-white border border-black/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-purple-400 shadow-sm font-bold" placeholder="e.g. my_trigger_word" />
          </div>

          {/* Params */}
          <div className="space-y-4">
            <button onClick={() => setShowSettings(!showSettings)} className="flex items-center justify-between w-full px-4 py-3 rounded-xl bg-white border border-black/10 text-black text-sm font-bold transition-all hover:bg-purple-600 hover:text-white hover:border-purple-600">
              <div className="flex items-center gap-2"><SlidersHorizontal className="w-4 h-4" /> Advanced Config</div>
              {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            <AnimatePresence>
              {showSettings && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden space-y-4 pt-2">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase font-bold text-black/40">Steps</label>
                      <input type="number" value={trainConfig.steps} onChange={e => setTrainConfig({ ...trainConfig, steps: parseInt(e.target.value) || 1000 })} className="w-full bg-white border border-black/10 rounded-lg px-3 py-2 text-xs font-mono" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase font-bold text-black/40">LoRA Rank</label>
                      <select value={trainConfig.rank} onChange={e => setTrainConfig({ ...trainConfig, rank: parseInt(e.target.value) })} className="w-full bg-white border border-black/10 rounded-lg px-2 py-2 text-xs">
                        {[4, 8, 16, 32, 64].map(r => <option key={r} value={r}>{r}</option>)}
                      </select>
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase font-bold text-black/40">Learning Rate</label>
                      <input type="number" step="0.00001" value={trainConfig.lr} onChange={e => setTrainConfig({ ...trainConfig, lr: parseFloat(e.target.value) || 1e-4 })} className="w-full bg-white border border-black/10 rounded-lg px-3 py-2 text-xs font-mono" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase font-bold text-black/40">Batch Size</label>
                      <select value={trainConfig.batchSize} onChange={e => setTrainConfig({ ...trainConfig, batchSize: parseInt(e.target.value) })} className="w-full bg-white border border-black/10 rounded-lg px-2 py-2 text-xs">
                        {[1, 2, 3, 4, 8].map(r => <option key={r} value={r}>{r}</option>)}
                      </select>
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] uppercase font-bold text-black/40">Resolution</label>
                      <select value={trainConfig.resolution} onChange={e => setTrainConfig({ ...trainConfig, resolution: parseInt(e.target.value) })} className="w-full bg-white border border-black/10 rounded-lg px-2 py-2 text-xs">
                        {[512, 768, 1024].map(r => <option key={r} value={r}>{r}px</option>)}
                      </select>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div className="p-8 pt-0 space-y-3 border-t border-black/5 bg-[#fafafa]">
          <button
            onClick={startTraining}
            disabled={trainRunning}
            className="w-full bg-purple-600 text-white font-bold py-5 rounded-2xl active:scale-[0.98] transition-all disabled:opacity-40 shadow-xl shadow-purple-500/30 flex items-center justify-center gap-3"
          >
            {trainRunning
              ? <><Loader2 className="w-6 h-6 animate-spin" /> Training…</>
              : <><Play className="w-6 h-6" /> Start Training</>
            }
          </button>
          {trainRunning && activeCallId && (
            <button onClick={stopTraining} className="w-full bg-white border border-red-200 text-red-500 font-bold py-4 rounded-2xl hover:bg-red-50 active:scale-[0.98] transition-all flex items-center justify-center gap-3">
              <AlertCircle className="w-5 h-5" /> Stop Training
            </button>
          )}
        </div>
      </aside>

      {/* Main View Area */}
      <div className="flex-1 flex flex-col bg-white overflow-hidden">
        <header className="h-16 border-b border-black/10 flex items-center justify-between px-10 shrink-0">
          <div className="flex items-center gap-4 text-sm font-medium">
            <span className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${trainRunning ? "bg-purple-500 animate-pulse" : "bg-green-500"}`} />
              {trainRunning
               ? `Training: ${trainProgress ? `${trainProgress.step}/${trainProgress.total} steps (${trainProgress.pct.toFixed(1)}%)` : "Initializing…"}`
               : trainDone ? "Training complete ✓" : "Configure & launch LoRA training"}
            </span>
          </div>
          {trainDone && (
            <button onClick={() => downloadLora(trainConfig.datasetName)} className="flex items-center gap-2 text-xs bg-purple-600 text-white px-4 py-1.5 rounded-full font-bold hover:bg-purple-700 transition-all">
              <ArrowDownToLine className="w-3 h-3" /> Download LoRA
            </button>
          )}
        </header>

        {/* Main Canvas */}
        <div className="flex-1 overflow-y-auto p-12 custom-scrollbar bg-[#f2f2f2]">
          <AnimatePresence mode="wait">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-8 h-full">
              
              {/* Progress bar */}
              {(trainRunning || trainDone) && (
                <div className="bg-white rounded-3xl border border-black/5 shadow-sm p-8 space-y-5">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-lg tracking-tight">{trainDone ? "Training Complete ✓" : "Training in Progress"}</h3>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs font-bold">
                      <span>Step {trainProgress?.step ?? 0} / {trainProgress?.total ?? trainConfig.steps}</span>
                      <span className="text-purple-600">{(trainProgress?.pct ?? 0).toFixed(1)}%</span>
                    </div>
                    <div className="h-3 bg-black/5 rounded-full overflow-hidden shadow-inner">
                      <motion.div initial={{ width: 0 }} animate={{ width: `${trainProgress?.pct ?? 0}%` }} className="h-full bg-purple-600 rounded-full" />
                    </div>
                  </div>
                </div>
              )}

              {/* Console Terminals Logs View */}
              <div className="flex-1 bg-black rounded-3xl p-8 overflow-hidden flex flex-col shadow-2xl border border-black/10">
                <div className="flex items-center justify-between mb-4 shrink-0">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500" />
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                    <span className="text-white/20 text-xs font-bold ml-2">CON_STREAM</span>
                  </div>
                  <Terminal className="w-5 h-5 text-white/40" />
                </div>
                <div className="flex-1 overflow-y-auto font-mono text-white/80 text-xs space-y-2 select-text custom-scrollbar">
                  {trainLogs.length === 0 ? (
                    <div className="text-white/20 italic">Awaiting streams logs output...</div>
                  ) : (
                    trainLogs.map((log, i) => (
                      <div key={i} className={`p-2 rounded-lg ${log.type === "error" ? "bg-red-950 text-red-400 border border-red-900/50" : log.type === "warning" ? "bg-yellow-950 text-yellow-400 border border-yellow-900/50" : "hover:bg-white/5"}`}>
                        {log.msg}
                      </div>
                    ))
                  )}
                  <div ref={trainLogEndRef} />
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
