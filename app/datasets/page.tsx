"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  FolderOpen, RefreshCw, SlidersHorizontal, ChevronDown, ChevronUp, FileArchive, Library, Save, CheckCircle2, Download
} from "lucide-react";

interface DatasetFile {
  name: string;
  caption: string;
  data?: string;
  source?: string;
}

export default function DatasetsPage() {
  const [activeDataset, setActiveDataset] = useState("default");
  const [datasets, setDatasets] = useState<string[]>([]);
  const [datasetFiles, setDatasetFiles] = useState<DatasetFile[]>([]);
  
  const [syncRunning, setSyncRunning] = useState(false);
  const [syncProgress, setSyncProgress] = useState(0);

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    fetchDatasetFiles(activeDataset);
  }, [activeDataset]);

  const fetchDatasets = async () => {
    try {
      const res = await fetch("/api/dataset?action=list");
      const data = await res.json();
      if (data.datasets) setDatasets(data.datasets);
    } catch (e) { console.error("Failed to fetch datasets", e); }
  };

  const fetchDatasetFiles = async (name: string) => {
    try {
      const res = await fetch(`/api/dataset?action=view&name=${name}`);
      const data = await res.json();
      if (data.files) setDatasetFiles(data.files);
    } catch (e) { console.error("Failed to fetch files", e); }
  };

  const syncDataset = async () => {
    const localOnly = datasetFiles.filter(f => f.source === "local");
    if (localOnly.length === 0) {
      alert("All images are already synced to Cloud Modal Volume!");
      return;
    }

    setSyncRunning(true);
    setSyncProgress(0);

    try {
      setSyncProgress(20);
      const filesPayload = localOnly.map(f => ({
        name: f.name,
        image: f.data, 
        caption: f.caption || ""
      }));

      setSyncProgress(50);
      const res = await fetch("/api/dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "sync",
          datasetName: activeDataset,
          files: filesPayload
        })
      });

      if (res.ok) {
         setSyncProgress(100);
         setTimeout(() => fetchDatasetFiles(activeDataset), 500);
      }
    } catch (e) {
      console.error("Sync error", e);
    } finally {
      setSyncRunning(false);
      setSyncProgress(0);
    }
  };

  const batchRename = async () => {
    if (!confirm(`Are you sure you want to rename all images in '${activeDataset}' sequentially?`)) return;
    try {
      const res = await fetch("/api/dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "rename", datasetName: activeDataset })
      });
      if (res.ok) fetchDatasetFiles(activeDataset);
    } catch (e) { console.error("Rename error", e); }
  };

  const batchCaption = async () => {
    const caption = window.prompt("Enter caption for all images in this dataset:");
    if (!caption) return;
    try {
      const res = await fetch("/api/dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "batch_caption", datasetName: activeDataset, caption })
      });
      if (res.ok) fetchDatasetFiles(activeDataset);
    } catch (e) { console.error("Caption error", e); }
  };

  const downloadDataset = async () => {
    try {
      const res = await fetch("/api/dataset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "export", datasetName: activeDataset }) });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = `${activeDataset}.zip`; a.click();
    } catch (e) { console.error("Export error", e); }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-white text-black font-sans">
      
      {/* Sidebar Controls */}
      <aside className="w-[380px] border-r border-black/10 flex flex-col bg-[#fafafa] shrink-0">
        <div className="p-8 flex-1 overflow-y-auto no-scrollbar space-y-8">
          <div>
            <h2 className="text-xl font-bold tracking-tight mb-1">Dataset Library</h2>
            <p className="text-xs text-black/40 font-medium tracking-tight">Manage and Curate locally cached folders</p>
          </div>

          {/* Dataset selector */}
          <div className="space-y-3">
            <label className="block text-[11px] uppercase tracking-wider font-bold text-black/40">Select Dataset</label>
            <select value={activeDataset} onChange={e => setActiveDataset(e.target.value)} className="w-full bg-white border border-black/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-1 focus:ring-purple-400 appearance-none shadow-sm font-bold">
              {datasets.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          {/* Sync Trigger */}
          <div className="space-y-4">
            <button
              onClick={syncDataset}
              disabled={syncRunning || datasetFiles.filter(f => f.source === "local").length === 0}
              className="w-full bg-black text-white px-4 py-3.5 rounded-xl font-bold text-sm tracking-tight hover:bg-black/90 active:scale-[0.98] transition-all disabled:opacity-30 shadow-sm flex items-center justify-center gap-2"
            >
              {syncRunning ? <Loader2Icon className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
              {syncRunning ? `Syncing (${syncProgress}%)` : "Sync with Cloud Volume"}
            </button>

            {/* Sync progress bar */}
            {syncRunning && (
              <div className="h-1.5 bg-black/5 rounded-full overflow-hidden">
                <motion.div initial={{ width: 0 }} animate={{ width: `${syncProgress}%` }} className="h-full bg-purple-600 rounded-full" />
              </div>
            )}
          </div>

          {/* Actions grid */}
          <div className="grid grid-cols-2 gap-3">
            <button onClick={batchRename} className="p-4 bg-white border border-black/10 rounded-xl text-xs font-bold text-black/70 hover:bg-black hover:text-white transition-all">
               Rename Set
            </button>
            <button onClick={batchCaption} className="p-4 bg-white border border-black/10 rounded-xl text-xs font-bold text-black/70 hover:bg-black hover:text-white transition-all">
               Batch Caption
            </button>
          </div>

          <button onClick={downloadDataset} className="w-full p-4 bg-white border border-black/10 rounded-xl text-xs font-bold text-black/70 hover:bg-black hover:text-white transition-all flex items-center justify-center gap-2">
             <FileArchive className="w-4 h-4" /> Download ZIP
          </button>
        </div>
      </aside>

      {/* Main View Area */}
      <div className="flex-1 flex flex-col bg-white overflow-hidden">
        <header className="h-16 border-b border-black/10 flex items-center justify-between px-10 shrink-0">
          <span className="text-sm font-medium text-black/40">
             {datasetFiles.length} curated samples in {activeDataset}
          </span>
          <button onClick={() => fetchDatasetFiles(activeDataset)} className="flex items-center gap-2 text-xs bg-black text-white px-4 py-1.5 rounded-full font-bold">
             <RefreshCw className="w-3 h-3" /> Reload List
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-12 custom-scrollbar bg-[#f2f2f2]">
          {datasetFiles.length > 0 ? (
            <div className="grid grid-cols-4 gap-8">
              {datasetFiles.map((file, i) => (
                <div key={i} className="bg-white rounded-3xl overflow-hidden border border-black/5 shadow-sm hover:shadow-xl transition-all group">
                  <div className="aspect-square bg-black/5 relative">
                    {file.data && <img src={`data:image/png;base64,${file.data}`} className="w-full h-full object-cover" />}
                    <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-md text-white text-[9px] font-bold px-3 py-1.5 rounded-full uppercase">
                       {file.source || "cloud"}
                    </div>
                  </div>
                  <div className="p-5">
                    <textarea defaultValue={file.caption} className="w-full h-20 bg-black/5 rounded-xl p-3 text-[11px] leading-relaxed focus:bg-white focus:ring-1 focus:ring-black outline-none transition-all resize-none shadow-inner" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-20">
              <FolderOpen className="w-20 h-20" />
              <h3 className="text-xl font-bold tracking-tight">Empty Dataset</h3>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Loader2Icon(props: any) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  )
}
