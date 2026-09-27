"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sparkles, LayoutDashboard, Library, Zap, Settings, Image } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  const isCurrent = (path: string) => {
    if (path === "/train" && (pathname === "/train" || pathname === "/")) return true;
    return pathname === path;
  };

  return (
    <nav className="w-16 border-r border-black/10 flex flex-col items-center py-6 gap-6 bg-white shrink-0">
      <div className="bg-black p-2 rounded-xl mb-4">
        <Sparkles className="w-6 h-6 text-white" />
      </div>

      <Link 
        href="/studio" 
        className={`p-3 rounded-xl transition-all ${isCurrent("/studio") ? "bg-black text-white shadow-lg" : "text-black/40 hover:bg-black/5"}`} 
        title="Creative Studio"
      >
        <LayoutDashboard className="w-6 h-6" />
      </Link>

      <Link 
        href="/img2img" 
        className={`p-3 rounded-xl transition-all ${isCurrent("/img2img") ? "bg-black text-white shadow-lg" : "text-black/40 hover:bg-black/5"}`} 
        title="Image to Image"
      >
        <Image className="w-6 h-6" />
      </Link>

      <Link 
        href="/datasets" 
        className={`p-3 rounded-xl transition-all ${isCurrent("/datasets") ? "bg-black text-white shadow-lg" : "text-black/40 hover:bg-black/5"}`} 
        title="Dataset Library"
      >
        <Library className="w-6 h-6" />
      </Link>

      <Link 
        href="/train" 
        className={`p-3 rounded-xl transition-all ${isCurrent("/train") ? "bg-purple-600 text-white shadow-lg shadow-purple-500/30" : "text-black/40 hover:bg-black/5"}`} 
        title="Train LoRA"
      >
        <Zap className="w-6 h-6" />
      </Link>
      
      <div className="flex-1" />
      
      <Link 
        href="/settings" 
        className={`p-3 rounded-xl transition-all ${isCurrent("/settings") ? "bg-black text-white shadow-lg" : "text-black/40 hover:bg-black/5"}`} 
        title="Settings"
      >
        <Settings className="w-6 h-6" />
      </Link>
    </nav>
  );
}
