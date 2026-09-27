"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileArchive, Loader2 } from "lucide-react";

interface TrainEvent {
  type: "info" | "log" | "progress" | "done" | "error";
  msg: string;
}

export default function SettingsPage() {
  const [settingsLogs, setSettingsLogs] = useState<TrainEvent[]>([]);
  const [settingsRunning, setSettingsRunning] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);

  const startDownloadFlux = async () => {
    setSettingsRunning(true);
    setSettingsError(null);
    setSettingsLogs([]);

    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "download_flux" }),
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
            setSettingsLogs(prev => [...prev, ev]);
            if (ev.type === "error") setSettingsError(ev.msg);
          } catch { }
        }
      }
    } catch (err) {
      setSettingsError(err instanceof Error ? err.message : String(err));
    } finally {
      setSettingsRunning(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-white overflow-hidden font-sans h-full">
      <header className="h-16 border-b border-black/10 flex items-center px-10 shrink-0">
        <span className="text-sm font-medium text-black/40">System Settings & Maintenance</span>
      </header>

      <div className="flex-1 overflow-y-auto p-12 custom-scrollbar bg-[#f2f2f2] flex flex-col gap-8">
        <AnimatePresence mode="wait">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-8 h-full">

            <div className="bg-white rounded-3xl border border-black/5 shadow-sm p-8 space-y-6">
              <div>
                <h3 className="font-bold text-lg tracking-tight">Maintenance Controls</h3>
                <p className="text-xs text-black/40 mt-0.5 font-medium">Trigger background seeding operations for shared volumes</p>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div className="bg-[#f8f8f8] border border-black/5 rounded-2xl p-6 space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-purple-50 rounded-xl"><FileArchive className="w-5 h-5 text-purple-600" /></div>
                    <div>
                      <h4 className="font-bold text-sm">Download Flux.1 weights</h4>
                      <p className="text-[10px] text-black/40 leading-tight">Syncs weight files to `flux-model` volume on CPU.</p>
                    </div>
                  </div>
                  <button 
                    onClick={startDownloadFlux} 
                    disabled={settingsRunning}
                    className="w-full bg-purple-600 text-white font-bold py-3 rounded-xl hover:bg-purple-700 disabled:opacity-40 transition-all flex items-center justify-center gap-2 text-xs"
                  >
                    {settingsRunning ? <><Loader2 className="w-4 h-4 animate-spin" /> Downloading…</> : "Trigger Download"}
                  </button>
                </div>
              </div>
            </div>

            {/* Settings Console */}
            {(settingsLogs.length > 0 || settingsRunning || settingsError) && (
              <div className="flex-1 bg-[#0d0d0d] rounded-3xl border border-white/5 overflow-hidden flex flex-col shadow-2xl min-h-0">
                <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1.5">
                      <div className="w-3 h-3 rounded-full bg-red-500/70" />
                      <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
                      <div className="w-3 h-3 rounded-full bg-green-500/70" />
                    </div>
                    <span className="text-[11px] font-mono font-bold text-white/30 uppercase tracking-widest border-l border-white/10 pl-3">Maintenance Stream</span>
                  </div>
                </div>
                <div className="flex-1 p-6 font-mono text-[11px] overflow-y-auto custom-scrollbar space-y-1">
                  {settingsError && (
                    <div className="text-red-400">✗ Error: {settingsError}</div>
                  )}
                  {settingsLogs.map((ev, i) => (
                    <div key={i} className={`flex gap-3 ${ev.type === "error" ? "text-red-400" : ev.type === "done" ? "text-green-400" : "text-white/60"}`}>
                      <span className="shrink-0 w-4">{ev.type === "done" ? "✓" : ev.type === "error" ? "✗" : "›"}</span>
                      <span className="break-all">{ev.msg}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
