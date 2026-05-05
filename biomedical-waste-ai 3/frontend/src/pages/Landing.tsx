import { useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Webcam from "react-webcam";
import {
  Camera, FileUp, Loader2, AlertCircle, RotateCcw,
  Zap, Shield, Activity,
} from "lucide-react";

import { Hero } from "@/components/ui/hero";
import { Dropzone } from "@/components/ui/dropzone";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { predictBase64, predictFile } from "@/lib/api";
import { cn } from "@/lib/utils";

type Mode = "upload" | "webcam";

/* ── Feature pills ──────────────────────────────────────────────────────── */
const FEATURES = [
  { icon: Zap,      label: "YOLOv8 Detection",          sub: "Object localisation" },
  { icon: Activity, label: "EfficientNet-B0",            sub: "Final classification" },
  { icon: Shield,   label: "Rule-based Segregation",     sub: "No LLM — pure logic" },
];

export default function Landing() {
  const nav = useNavigate();
  const camRef = useRef<Webcam>(null);

  const [mode, setMode]       = useState<Mode>("upload");
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [camReady, setCamReady] = useState(false);

  /* ── Helpers ────────────────────────────────────────────────────────── */
  const navigate = useCallback(
    (res: object, name: string) => {
      sessionStorage.setItem("lastPrediction", JSON.stringify(res));
      sessionStorage.setItem("lastImageName", name);
      nav("/predict");
    },
    [nav]
  );

  function fileToB64(f: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result));
      r.onerror = reject;
      r.readAsDataURL(f);
    });
  }

  async function handleFile(file: File) {
    setError(null);
    setPreview(URL.createObjectURL(file));
    setLoading(true);
    try {
      const b64 = await fileToB64(file);
      sessionStorage.setItem("lastImageB64", b64);
      const res = await predictFile(file);
      navigate(res, file.name);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed — is the backend running?");
      setLoading(false);
    }
  }

  async function handleCapture() {
    const shot = camRef.current?.getScreenshot();
    if (!shot) { setError("Webcam not ready — please allow camera access."); return; }
    setError(null);
    setLoading(true);
    try {
      sessionStorage.setItem("lastImageB64", shot);
      const res = await predictBase64(shot);
      navigate(res, "webcam_capture.jpg");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed — is the backend running?");
      setLoading(false);
    }
  }

  function reset() {
    setPreview(null);
    setError(null);
    setLoading(false);
  }

  /* ── Render ─────────────────────────────────────────────────────────── */
  return (
    <Hero>
      {/* Mode toggle */}
      <div className="flex items-center justify-center gap-2 mb-6">
        {(["upload", "webcam"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => { setMode(m); reset(); }}
            className={cn(
              "flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200",
              mode === m
                ? "bg-gradient-to-r from-violet-600 to-fuchsia-500 text-white shadow-btn-primary"
                : "glass text-foreground/60 hover:text-foreground hover:bg-white/5"
            )}
          >
            {m === "upload"
              ? <FileUp className="h-4 w-4" />
              : <Camera className="h-4 w-4" />}
            {m === "upload" ? "Upload image" : "Live webcam"}
          </button>
        ))}
      </div>

      {/* Input area */}
      <div className="w-full max-w-2xl mx-auto">
        {mode === "upload" ? (
          <Dropzone
            onFile={handleFile}
            preview={preview}
            className="min-h-[220px]"
          />
        ) : (
          <div className="glass rounded-2xl overflow-hidden border border-white/10">
            <div className="relative">
              <Webcam
                ref={camRef}
                audio={false}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.92}
                videoConstraints={{ facingMode: "environment", width: 1280, height: 720 }}
                onUserMedia={() => setCamReady(true)}
                onUserMediaError={() => setError("Camera access denied or unavailable.")}
                className="w-full aspect-video object-cover bg-black"
              />
              {/* Scanning overlay */}
              {camReady && (
                <div className="absolute inset-0 pointer-events-none">
                  {/* Corner brackets */}
                  {[
                    "top-4 left-4 border-t-2 border-l-2 rounded-tl-lg",
                    "top-4 right-4 border-t-2 border-r-2 rounded-tr-lg",
                    "bottom-4 left-4 border-b-2 border-l-2 rounded-bl-lg",
                    "bottom-4 right-4 border-b-2 border-r-2 rounded-br-lg",
                  ].map((cls) => (
                    <div key={cls} className={`absolute h-8 w-8 border-violet-400/60 ${cls}`} />
                  ))}
                  {/* Scan line */}
                  <div
                    className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-violet-400/60 to-transparent"
                    style={{ animation: "scanLine 2.5s ease-in-out infinite", top: "50%" }}
                  />
                </div>
              )}
            </div>
            <div className="p-4 flex items-center justify-between">
              <span className={cn(
                "flex items-center gap-2 text-xs",
                camReady ? "text-emerald-400" : "text-foreground/40"
              )}>
                <span className={cn(
                  "h-2 w-2 rounded-full",
                  camReady ? "bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.5)] animate-pulse" : "bg-white/20"
                )} />
                {camReady ? "Camera ready" : "Initialising…"}
              </span>
              <Button
                onClick={handleCapture}
                loading={loading}
                disabled={!camReady || loading}
                size="sm"
              >
                <Camera className="h-4 w-4" />
                Capture & analyse
              </Button>
            </div>
          </div>
        )}

        {/* Loading state */}
        {loading && (
          <div className="mt-5 glass rounded-xl px-5 py-4 flex items-center gap-4">
            <div className="relative shrink-0">
              <Loader2 className="h-6 w-6 text-violet-400 animate-spin" />
              <div className="absolute inset-0 blur-md bg-violet-500/40 rounded-full" />
            </div>
            <div className="text-left">
              <p className="text-sm font-medium text-foreground/80">Processing…</p>
              <p className="text-xs text-foreground/40 mt-0.5">
                YOLOv8 detecting → cropping → EfficientNet-B0 classifying
              </p>
            </div>
          </div>
        )}

        {/* Error state */}
        {error && (
          <div className="mt-5 rounded-xl border border-red-500/30 bg-red-500/8 px-5 py-4 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
            <div className="flex-1 text-left">
              <p className="text-sm font-medium text-red-300">Error</p>
              <p className="text-xs text-red-400/70 mt-0.5 break-all">{error}</p>
            </div>
            <button
              onClick={reset}
              className="shrink-0 text-red-400/60 hover:text-red-300 transition-colors"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>

      {/* Feature cards */}
      <div className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl mx-auto">
        {FEATURES.map(({ icon: Icon, label, sub }) => (
          <Card key={label} hoverable className="p-4 text-center">
            <div className="mx-auto mb-3 h-10 w-10 rounded-xl bg-gradient-to-br from-violet-600/20 to-fuchsia-600/20 flex items-center justify-center border border-white/10">
              <Icon className="h-5 w-5 text-violet-300" />
            </div>
            <p className="text-sm font-medium text-foreground/80">{label}</p>
            <p className="text-xs text-foreground/40 mt-1">{sub}</p>
          </Card>
        ))}
      </div>

      {/* Scan-line keyframe injected via style tag */}
      <style>{`
        @keyframes scanLine {
          0%   { transform: translateY(-80px); opacity: 0; }
          10%  { opacity: 1; }
          90%  { opacity: 1; }
          100% { transform: translateY(80px); opacity: 0; }
        }
      `}</style>
    </Hero>
  );
}
