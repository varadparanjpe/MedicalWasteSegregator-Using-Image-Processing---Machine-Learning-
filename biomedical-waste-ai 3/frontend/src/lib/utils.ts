import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/* ── Colour maps (used on both Category and Bin) ──────────────────────── */
export const CATEGORY_COLOR: Record<string, string> = {
  sharps_waste:         "#ef4444",
  infectious_waste:     "#f59e0b",
  pathological_waste:   "#a855f7",
  plastic_recyclable:   "#3b82f6",
  pharmaceutical_waste: "#10b981",
  general_waste:        "#9ca3af",
};

export const CATEGORY_LABEL: Record<string, string> = {
  sharps_waste:         "Sharps Waste",
  infectious_waste:     "Infectious Waste",
  pathological_waste:   "Pathological Waste",
  plastic_recyclable:   "Plastic Recyclable",
  pharmaceutical_waste: "Pharmaceutical Waste",
  general_waste:        "General Waste",
};

export const BIN_COLOR: Record<string, string> = {
  "White Bin":  "#e5e7eb",
  "Yellow Bin": "#f59e0b",
  "Red Bin":    "#ef4444",
  "Blue Bin":   "#3b82f6",
  "Black Bin":  "#6b7280",
};

export const BIN_EMOJI: Record<string, string> = {
  "White Bin":  "⬜",
  "Yellow Bin": "🟡",
  "Red Bin":    "🔴",
  "Blue Bin":   "🔵",
  "Black Bin":  "⬛",
};

/* ── Formatters ────────────────────────────────────────────────────────── */
export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
export const fmt = (v: number, d = 1) => `${(v * 100).toFixed(d)}`;

export function confColor(c: number): string {
  if (c >= 0.90) return "#10b981";
  if (c >= 0.80) return "#3b82f6";
  if (c >= 0.65) return "#f59e0b";
  return "#ef4444";
}

export function relTime(iso: string): string {
  const diff = Date.now() - new Date(iso + "Z").getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
