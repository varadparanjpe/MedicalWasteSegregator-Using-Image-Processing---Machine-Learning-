import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft, Timer, ShieldAlert, AlertTriangle, Trash2, ChevronDown,
  ChevronUp, Maximize2, CheckCircle2, Info, Flame, Eye, EyeOff,
} from "lucide-react";

import { Card, Badge } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ConfBar } from "@/components/ui/confidence-ring";
import { predictFile, predictBase64, type PredictResponse, type Detection } from "@/lib/api";
import {
  BIN_COLOR, BIN_EMOJI, CATEGORY_COLOR, CATEGORY_LABEL,
  pct, cn,
} from "@/lib/utils";

/* ── Pipeline step indicator ───────────────────────────────────────────── */
const STEPS = [
  "YOLOv8 Detection",
  "Object Cropping",
  "EfficientNet-B0 Classification",
  "Rule-Based Segregation",
  "Alert Evaluation",
];

function PipelineSteps() {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), 300); return () => clearTimeout(t); }, []);
  return (
    <div className={cn("glass rounded-xl px-5 py-4 transition-all duration-500", visible ? "opacity-100" : "opacity-0")}>
      <p className="text-xs font-semibold uppercase tracking-widest text-foreground/40 mb-3">Inference pipeline</p>
      <div className="flex items-center gap-0 overflow-x-auto">
        {STEPS.map((s, i) => (
          <div key={s} className="flex items-center shrink-0">
            <div className="flex flex-col items-center gap-1.5">
              <div
                className="h-6 w-6 rounded-full flex items-center justify-center text-[10px] font-bold text-white"
                style={{ background: `linear-gradient(135deg, #7c3aed, #ec4899)` }}
              >
                {i + 1}
              </div>
              <p className="text-[9px] text-foreground/40 text-center w-16 leading-tight">{s}</p>
            </div>
            {i < STEPS.length - 1 && (
              <div className="h-0.5 w-6 bg-gradient-to-r from-violet-500/50 to-fuchsia-500/50 mb-4 shrink-0" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Canvas overlay ─────────────────────────────────────────────────────── */
function BBoxCanvas({
  src, detections, width, height,
}: { src: string; detections: Detection[]; width: number; height: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef    = useRef<HTMLImageElement>(null);
  const [hov, setHov] = useState<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete) return;
    const ctx = canvas.getContext("2d")!;
    const scaleX = canvas.width / width, scaleY = canvas.height / height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((d) => {
      const col = CATEGORY_COLOR[d.category] ?? "#a78bfa";
      const isH = hov === d.id;
      const { x1, y1, x2, y2 } = d.bbox;
      const sx1 = x1 * scaleX, sy1 = y1 * scaleY, sx2 = x2 * scaleX, sy2 = y2 * scaleY;

      ctx.strokeStyle = col; ctx.lineWidth = isH ? 3 : 2;
      ctx.shadowColor = col; ctx.shadowBlur = isH ? 16 : 8;
      ctx.beginPath(); ctx.roundRect(sx1, sy1, sx2 - sx1, sy2 - sy1, 6); ctx.stroke();

      const label = `${CATEGORY_LABEL[d.category] ?? d.category}  ${pct(d.classification_confidence)}`;
      ctx.font = `bold ${isH ? 12 : 11}px Inter, sans-serif`;
      const tw = ctx.measureText(label).width + 14, th = 20;
      const lx = sx1, ly = sy1 - th - 4;
      ctx.fillStyle = col + "cc"; ctx.shadowBlur = 0;
      ctx.beginPath(); ctx.roundRect(lx, ly < 0 ? sy1 + 4 : ly, tw, th, 4); ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.fillText(label, lx + 7, (ly < 0 ? sy1 + 4 : ly) + 14);
    });
  }, [detections, width, height, hov]);

  return (
    <div className="relative w-full">
      <img ref={imgRef} src={src} alt="annotated"
        className="w-full rounded-xl object-contain bg-black/50"
        onLoad={() => {
          const c = canvasRef.current, i = imgRef.current;
          if (c && i) { c.width = i.naturalWidth; c.height = i.naturalHeight; }
        }}
      />
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full rounded-xl pointer-events-none" />
      {detections.map((d) => (
        <div key={d.id}
          className="absolute cursor-pointer"
          style={{
            left:   `${(d.bbox.x1 / width) * 100}%`,
            top:    `${(d.bbox.y1 / height) * 100}%`,
            width:  `${((d.bbox.x2 - d.bbox.x1) / width) * 100}%`,
            height: `${((d.bbox.y2 - d.bbox.y1) / height) * 100}%`,
          }}
          onMouseEnter={() => setHov(d.id)}
          onMouseLeave={() => setHov(null)}
        />
      ))}
    </div>
  );
}

/* ── Detection card ─────────────────────────────────────────────────────── */
function DetectionCard({ d, index }: { d: Detection; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const [showCam, setShowCam] = useState(false);
  const color = CATEGORY_COLOR[d.category] ?? "#a78bfa";
  const binColor = BIN_COLOR[d.bin] ?? "#e5e7eb";
  const isHaz  = d.alerts.includes("HAZARDOUS");
  const isLow  = d.alerts.includes("LOW_CONFIDENCE");

  return (
    <div
      className={cn(
        "glass rounded-2xl overflow-hidden border animate-fade-up",
        isHaz ? "border-red-500/30" : "border-white/[0.07]"
      )}
      style={{ animationDelay: `${index * 0.08}s` }}
    >
      <div className="flex items-center gap-4 p-4">
        {/* Thumbnail — toggle between crop and Grad-CAM */}
        {(d.cropped_image_b64 || d.gradcam_b64) && (
          <div className="relative shrink-0">
            <img
              src={showCam && d.gradcam_b64 ? d.gradcam_b64 : d.cropped_image_b64 || d.gradcam_b64}
              alt="crop"
              className="h-20 w-20 rounded-xl object-cover border border-white/10"
            />
            {d.gradcam_b64 && (
              <button
                onClick={() => setShowCam((v) => !v)}
                className={cn(
                  "absolute -bottom-1 -right-1 h-6 w-6 rounded-full flex items-center justify-center border border-white/10 transition-colors",
                  showCam ? "bg-orange-500 text-white" : "glass-heavy text-foreground/60 hover:text-white"
                )}
                title={showCam ? "Show crop" : "Show Grad-CAM"}
              >
                <Flame className="h-3 w-3" />
              </button>
            )}
            {isHaz && (
              <div className="absolute -top-1.5 -right-1.5 h-5 w-5 rounded-full bg-red-500 flex items-center justify-center shadow-glow-red">
                <ShieldAlert className="h-3 w-3 text-white" />
              </div>
            )}
          </div>
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className="text-sm font-semibold px-3 py-1 rounded-full border"
              style={{ color, background: color + "18", borderColor: color + "40" }}
            >
              {CATEGORY_LABEL[d.category] ?? d.category}
            </span>
            {isHaz && <Badge variant="hazard"><ShieldAlert className="h-3 w-3" />Hazardous</Badge>}
            {isLow && <Badge variant="warning"><AlertTriangle className="h-3 w-3" />Low confidence</Badge>}
            {!isHaz && !isLow && <Badge variant="success"><CheckCircle2 className="h-3 w-3" />Safe</Badge>}
          </div>

          <div className="mt-2.5 flex items-center gap-2">
            <Trash2 className="h-4 w-4 shrink-0" style={{ color: binColor }} />
            <span className="text-sm text-foreground/80">
              Dispose in{" "}
              <strong style={{ color: binColor }}>
                {BIN_EMOJI[d.bin]} {d.bin}
              </strong>
            </span>
          </div>

          <div className="mt-3">
            <ConfBar value={d.classification_confidence} label="EfficientNet-B0 confidence" />
          </div>
        </div>

        <button
          onClick={() => setExpanded((v) => !v)}
          className="shrink-0 h-8 w-8 rounded-full glass flex items-center justify-center text-foreground/40 hover:text-foreground transition-colors"
        >
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </button>
      </div>

      {expanded && (
        <div className="border-t border-white/[0.06] px-4 pb-4 pt-3 grid grid-cols-2 gap-3 animate-fade-in">
          {[
            ["Detection #",     String(d.id + 1)],
            ["YOLOv8 conf",     pct(d.detection_confidence)],
            ["Classifier conf", pct(d.classification_confidence)],
            ["Bounding box",    `[${d.bbox.x1.toFixed(0)}, ${d.bbox.y1.toFixed(0)}, ${d.bbox.x2.toFixed(0)}, ${d.bbox.y2.toFixed(0)}]`],
            ["Box size",        `${(d.bbox.x2 - d.bbox.x1).toFixed(0)} × ${(d.bbox.y2 - d.bbox.y1).toFixed(0)} px`],
            ["Alerts",          d.alerts.length ? d.alerts.join(", ") : "none"],
          ].map(([k, v]) => (
            <div key={k} className="bg-white/[0.03] rounded-lg px-3 py-2">
              <p className="text-[10px] text-foreground/40 uppercase tracking-wider">{k}</p>
              <p className="text-xs font-mono text-foreground/70 mt-0.5 break-all">{v}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Page ───────────────────────────────────────────────────────────────── */
export default function Prediction() {
  const nav = useNavigate();
  const [data, setData]       = useState<PredictResponse | null>(null);
  const [imageName, setName]  = useState("");
  const [fullscreen, setFull] = useState(false);

  const [explain, setExplain]   = useState(false);
  const [ensemble, setEnsemble] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const raw = sessionStorage.getItem("lastPrediction");
    if (!raw) { nav("/"); return; }
    setData(JSON.parse(raw));
    setName(sessionStorage.getItem("lastImageName") ?? "");
  }, [nav]);

  async function rerun(nextExplain: boolean, nextEnsemble: boolean) {
    const fileJson = sessionStorage.getItem("lastImageB64");
    if (!fileJson) { setExplain(nextExplain); setEnsemble(nextEnsemble); return; }
    setRefreshing(true);
    try {
      const res = await predictBase64(fileJson, {
        include_annotated: true, explain: nextExplain, use_ensemble: nextEnsemble,
      });
      setData(res);
      sessionStorage.setItem("lastPrediction", JSON.stringify(res));
      setExplain(nextExplain);
      setEnsemble(nextEnsemble);
    } catch (e) {
      console.error(e);
    } finally {
      setRefreshing(false);
    }
  }

  if (!data) return null;

  const hazardous = data.detections.some((d) => d.alerts.includes("HAZARDOUS"));
  const lowConf   = data.detections.some((d) => d.alerts.includes("LOW_CONFIDENCE"));

  return (
    <main className="min-h-screen bg-[#0c0414] bg-hero-glow text-foreground pt-24 pb-20 px-4 sm:px-6 page-enter">
      <div className="max-w-7xl mx-auto">

        {/* Top bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <Button variant="ghost" size="sm" onClick={() => nav("/")}>
            <ArrowLeft className="h-4 w-4" /> New image
          </Button>

          <div className="flex items-center gap-4 text-xs text-foreground/40 flex-wrap">
            <span className="flex items-center gap-1.5">
              <Timer className="h-3.5 w-3.5" />
              {data.processing_ms.toFixed(0)} ms
            </span>
            <span>{data.num_detections} detection{data.num_detections !== 1 ? "s" : ""}</span>
            <span>{data.image_width} × {data.image_height} px</span>
            {data.used_ensemble && <Badge variant="default">Ensemble</Badge>}
            {imageName && <span className="max-w-[180px] truncate text-foreground/30">{imageName}</span>}
          </div>
        </div>

        {/* Toggle row */}
        <div className="flex flex-wrap items-center gap-2 mb-6">
          <button
            onClick={() => rerun(!explain, ensemble)}
            disabled={refreshing}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium transition-all border",
              explain
                ? "bg-orange-500/20 text-orange-300 border-orange-500/40"
                : "glass text-foreground/60 hover:text-foreground border-white/10"
            )}
          >
            {explain ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
            Grad-CAM {explain ? "ON" : "OFF"}
          </button>

          <button
            onClick={() => rerun(explain, !ensemble)}
            disabled={refreshing}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium transition-all border",
              ensemble
                ? "bg-violet-500/20 text-violet-200 border-violet-500/40"
                : "glass text-foreground/60 hover:text-foreground border-white/10"
            )}
          >
            Ensemble (EfficientNet + ResNet50) {ensemble ? "ON" : "OFF"}
          </button>

          {refreshing && (
            <span className="text-xs text-foreground/40 animate-pulse ml-2">Re-running…</span>
          )}
        </div>

        {/* Global alert banners */}
        {(hazardous || lowConf) && (
          <div className="flex flex-wrap gap-3 mb-6">
            {hazardous && (
              <div className="flex-1 min-w-[260px] flex items-center gap-3 rounded-xl border border-red-500/30 bg-red-500/8 px-4 py-3 alert-pulse">
                <div className="h-8 w-8 rounded-lg bg-red-500/20 flex items-center justify-center shrink-0">
                  <ShieldAlert className="h-4 w-4 text-red-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-red-300">Hazardous waste detected</p>
                  <p className="text-xs text-red-400/60">Follow WHO disposal protocol immediately.</p>
                </div>
              </div>
            )}
            {lowConf && (
              <div className="flex-1 min-w-[260px] flex items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/8 px-4 py-3">
                <div className="h-8 w-8 rounded-lg bg-amber-500/20 flex items-center justify-center shrink-0">
                  <AlertTriangle className="h-4 w-4 text-amber-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-amber-300">Low-confidence detections</p>
                  <p className="text-xs text-amber-400/60">Human review advised — confidence below 80%.</p>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="mb-6"><PipelineSteps /></div>

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-6">
          <div className="space-y-4">
            <div className={cn(
              "glass rounded-2xl overflow-hidden p-3 transition-all duration-300",
              fullscreen && "fixed inset-4 z-50 flex items-center justify-center bg-black/90"
            )}>
              <div className="flex items-center justify-between px-2 pb-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-foreground/40">
                  Annotated preview
                </p>
                <button
                  onClick={() => setFull((v) => !v)}
                  className="text-foreground/30 hover:text-foreground transition-colors"
                >
                  <Maximize2 className="h-4 w-4" />
                </button>
              </div>

              {data.annotated_image_b64 ? (
                <BBoxCanvas
                  src={data.annotated_image_b64}
                  detections={data.detections}
                  width={data.image_width}
                  height={data.image_height}
                />
              ) : (
                <div className="aspect-video rounded-xl bg-white/5 flex items-center justify-center text-foreground/30 text-sm">
                  No image returned.
                </div>
              )}
            </div>

            {data.detections.length > 0 && (
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: "Detections",    value: String(data.num_detections), sub: "total objects" },
                  {
                    label: "Avg confidence",
                    value: pct(data.detections.reduce((s, d) => s + d.classification_confidence, 0) / data.detections.length),
                    sub: ensemble ? "Ensemble" : "EfficientNet-B0",
                  },
                  {
                    label: "Hazard items",
                    value: String(data.detections.filter((d) => d.alerts.includes("HAZARDOUS")).length),
                    sub: "require protocol",
                  },
                ].map(({ label, value, sub }) => (
                  <div key={label} className="glass rounded-xl px-4 py-3 text-center">
                    <p className="text-[10px] uppercase tracking-widest text-foreground/40">{label}</p>
                    <p className="text-2xl font-semibold text-foreground mt-1">{value}</p>
                    <p className="text-[10px] text-foreground/30 mt-0.5">{sub}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-3 overflow-y-auto max-h-[85vh] pr-1">
            {data.detections.length === 0 ? (
              <div className="glass rounded-2xl p-8 text-center">
                <Info className="h-8 w-8 text-foreground/20 mx-auto mb-3" />
                <p className="text-foreground/50 text-sm">
                  No biomedical waste objects detected in this image.
                </p>
                <Button variant="outline" size="sm" className="mt-4" onClick={() => nav("/")}>
                  Try another image
                </Button>
              </div>
            ) : (
              data.detections.map((d, i) => (
                <DetectionCard key={d.id} d={d} index={i} />
              ))
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
