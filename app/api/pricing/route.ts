import { NextResponse } from "next/server";

export interface GpuPricingSpec {
  id: string;
  name: string;
  vramGb: number;
  pricePerSec: number;
  pricePerHour: number;
}

// Fallback pricing extracted from https://modal.com/pricing
export const DEFAULT_GPU_PRICING: Record<string, GpuPricingSpec> = {
  "T4": { id: "T4", name: "Nvidia T4", vramGb: 16, pricePerSec: 0.000164, pricePerHour: 0.5904 },
  "L4": { id: "L4", name: "Nvidia L4", vramGb: 24, pricePerSec: 0.000222, pricePerHour: 0.7992 },
  "A10G": { id: "A10G", name: "Nvidia A10G", vramGb: 24, pricePerSec: 0.000306, pricePerHour: 1.1016 },
  "L40S": { id: "L40S", name: "Nvidia L40S", vramGb: 48, pricePerSec: 0.000542, pricePerHour: 1.9512 },
  "A100-40GB": { id: "A100-40GB", name: "Nvidia A100 (40 GB)", vramGb: 40, pricePerSec: 0.000583, pricePerHour: 2.0988 },
  "A100": { id: "A100", name: "Nvidia A100 (40 GB)", vramGb: 40, pricePerSec: 0.000583, pricePerHour: 2.0988 },
  "A100-80GB": { id: "A100-80GB", name: "Nvidia A100 (80 GB)", vramGb: 80, pricePerSec: 0.000694, pricePerHour: 2.4984 },
  "RTX-PRO-6000": { id: "RTX-PRO-6000", name: "Nvidia RTX PRO 6000", vramGb: 48, pricePerSec: 0.000842, pricePerHour: 3.0312 },
  "H100": { id: "H100", name: "Nvidia H100 SXM5", vramGb: 80, pricePerSec: 0.001097, pricePerHour: 3.9492 },
  "H200": { id: "H200", name: "Nvidia H200 SXM", vramGb: 141, pricePerSec: 0.001261, pricePerHour: 4.5396 },
  "B200": { id: "B200", name: "Nvidia B200", vramGb: 192, pricePerSec: 0.001736, pricePerHour: 6.2496 },
  "B300": { id: "B300", name: "Nvidia B300", vramGb: 288, pricePerSec: 0.001972, pricePerHour: 7.0992 },
};

export async function GET() {
  try {
    // Attempt dynamic fetch from modal.com/pricing with timeout
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3500);

    const res = await fetch("https://modal.com/pricing", {
      signal: controller.signal,
      headers: { "User-Agent": "Modal-Frontend/1.0" },
      next: { revalidate: 3600 }, // Cache 1 hour
    });
    clearTimeout(timeout);

    if (res.ok) {
      const html = await res.text();
      const livePricing = { ...DEFAULT_GPU_PRICING };

      // Helper regex to extract per-second prices
      const patterns: [string, RegExp][] = [
        ["T4", /Nvidia T4[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
        ["L4", /Nvidia L4[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
        ["A10G", /Nvidia A10[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
        ["L40S", /Nvidia L40S[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
        ["A100-40GB", /Nvidia A100,\s*40\s*GB[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
        ["A100-80GB", /Nvidia A100,\s*80\s*GB[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
        ["H100", /Nvidia H100 SXM5[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
        ["H200", /Nvidia H200 SXM[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
        ["B200", /Nvidia B200[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
        ["B300", /Nvidia B300[\s\S]*?\$([0-9.]+)\s*<\s*span[^>]*>\s*\/\s*sec/i],
      ];

      for (const [key, regex] of patterns) {
        const match = html.match(regex);
        if (match && match[1]) {
          const perSec = parseFloat(match[1]);
          if (!isNaN(perSec) && livePricing[key]) {
            livePricing[key].pricePerSec = perSec;
            livePricing[key].pricePerHour = parseFloat((perSec * 3600).toFixed(4));
          }
        }
      }

      // Keep alias A100 synced with A100-40GB
      livePricing["A100"] = { ...livePricing["A100-40GB"], id: "A100" };

      return NextResponse.json({
        source: "modal.com/pricing",
        updatedAt: new Date().toISOString(),
        pricing: livePricing,
      });
    }
  } catch {
    // Graceful fallback to cached official rates
  }

  return NextResponse.json({
    source: "cached",
    updatedAt: new Date().toISOString(),
    pricing: DEFAULT_GPU_PRICING,
  });
}
