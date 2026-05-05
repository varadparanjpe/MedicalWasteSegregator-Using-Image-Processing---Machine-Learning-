/**
 * Realtime webcam page.
 * - Auto-captures at 1–2 fps (frame skipping)
 * - Async inference with concurrency guard (one in-flight request at a time)
 * - Live bounding-box overlay drawn from last successful response
 * - Live latency + throughput metrics pulled from /api/perf
 */
import { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import {
  Play, Pause, Zap, Gauge, AlertTriangle,
  ShieldAlert, Camera, Activity,
} from "lucide-react";

import { Card, CardTitle, Badge } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { predictBase64, type PredictResponse, getPerf, type PerfStats } from "@/lib/api";
import {
  BIN_COLOR, BIN_EMOJI, CATEGORY_COLOR, CATEGORY_LABEL,
  cn, confColor, fmt,
} from "@/lib/utils";

const FPS_CHOICES = [0.5, 1, 2] as const;
type Fps = typeof FPS_CHOICES[number];

export default function Realtime() {
  const camRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inflight = useRef(false);
  const videoBox = useRef<{ w: number; h: number }>({ w: 0, h: 0 });

  const [running, setRunning]   = useState(false);
  const [fps, setFps]           = useState<Fps>(1);
  const [last, setLast]         = useState<PredictResponse | null>(null);
  const [perf, setPerf]         = useState<PerfStats | null>(null);
  const [frames, setFrames]     = useState(0);
  const [droppedFrames, setDropped] = useState(0);
  const [error, setError]       = useState<string | null>(null);

  /* ── Async inference loop (frame-skipped) ─────────────────────────── */
  useEffect(() => {
    if (!running) return;
    const interval = setInterval(async () => {
      if (inflight.current) {
        setDropped((d) => d + 1);
        return;
      }
      const shot = camRef.current?.getScreenshot();
      if (!shot) return;
      inflight.current = true;
      try {
        const res = await predictBase64(shot, { include_annotated: false });
        setLast(res);
        setFrames((f) => f + 1);
        setError(null);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Inference error");
      } finally {
        inflight.current = false;
      }
    }, 1000 / fps);
    return () => clearInterval(interval);
  }, [running, fps]);

  /* ── Poll /api/perf every 2 s ─────────────────────────────────────── */
  useEffect(() => {
    const poll = async () => {
      try { setPerf(await getPerf()); } catch { /* ignore */ }
    };
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, []);

  /* ── Draw boxes on overlay canvas ─────────────────────────────────── */
  const drawBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const vid = camRef.current?.video as HTMLVideoElement | undefined;
    if (!canvas || !vid || !last) return;
    const cw = vid.clientWidth, ch = vid.clientHeight;
    canvas.width = cw; canvas.height = ch;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, cw, ch);

    const sx = cw / last.image_width;
    const sy = ch / last.image_height;

    last.detections.forEach((d) => {
      const col = CATEGORY_COLOR[d.category] ?? "#a78bfa";
      const x1 = d.bbox.x1 * sx, y1 = d.bbox.y1 * sy;
      const x2 = d.bbox.x2 * sx, y2 = d.bbox.y2 * sy;

      ctx.strokeStyle = col;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = col; ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 6);
      ctx.stroke();

      // Pill
      const label = `${CATEGORY_LABEL[d.category] ?? d.category}  ${(d.classification_confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 11px Inter, sans-serif";
      const tw = ctx.measureText(label).width + 12;
      const th = 20;
      ctx.fillStyle = col + "dd";
      ctx.shadowBlur = 0;
      ctx.beginPath();
      ctx.roundRect(x1, Math.max(0, y1 - th - 4), tw, th, 4);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x1 + 6, Math.max(14, y1 - 8));
    });
  }, [last]);

  useEffect(() => { drawBoxes(); }, [drawBoxes]);
  useEffect(() => {
    const onResize = () => drawBoxes();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [drawBoxes]);

  /* ── Summary ─────────────────────────────────────────────────────── */
  const hazardCount = last?.detections.filter((d) => d.alerts.includes("HAZARDOUS")).length ?? 0;
  const lowConf     = last?.detections.filter((d) => d.alerts.includes("LOW_CONFIDENCE")).length ?? 0;

  return (
    <main className="min-h-screen bg-[#0c0414] bg-hero-glow text-foreground pt-24 pb-16 px-4 sm:px-6 page-enter">
      <div className="max-w-7xl mx-auto">
        <header className="mb-6">
          <h1 className="text-3xl font-semibold gradient-text flex items-center gap-3">
            <Camera className="h-7 w-7 text-violet-400" />
            Realtime detection
          </h1>
          <p className="text-sm text-foreground/40 mt-1">
            Auto-capture @ {fps} fps  ·  async inference  ·  frame skipping when busy
          </p>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-6">
          {/* ── Left: live video + overlay ── */}
          <Card className="p-3">
            <div className="relative rounded-xl overflow-hidden bg-black">
              <Webcam
                ref={camRef}
                audio={false}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.85}
                videoConstraints={{ facingMode: "environment", width: 1280, height: 720 }}
                className="w-full h-auto object-cover"
                onLoadedMetadata={() => {
                  const v = camRef.current?.video as HTMLVideoElement | undefined;
                  if (v) videoBox.current = { w: v.videoWidth, h: v.videoHeight };
                }}
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />

              {/* Status overlay */}
              <div className="absolute top-3 left-3 flex items-center gap-2">
                <div className={cn(
                  "flex items-center gap-1.5 glass-heavy rounded-full px-3 py-1.5 text-xs",
                  running ? "text-emerald-300" : "text-foreground/50"
                )}>
                  <div className={cn(
                    "h-2 w-2 rounded-full",
                    running
                      ? "bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.6)] animate-pulse"
                      : "bg-white/30"
                  )} />
                  {running ? "LIVE" : "PAUSED"}
                </div>
                {inflight.current && (
                  <div className="glass-heavy rounded-full px-3 py-1.5 text-xs text-violet-300">
                    inferring…
                  </div>
                )}
              </div>

              {/* Corner brackets */}
              {running && (
                <>
                  {[
                    "top-4 left-4 border-t-2 border-l-2 rounded-tl-lg",
                    "top-4 right-4 border-t-2 border-r-2 rounded-tr-lg",
                    "bottom-4 left-4 border-b-2 border-l-2 rounded-bl-lg",
                    "bottom-4 right-4 border-b-2 border-r-2 rounded-br-lg",
                  ].map((cls) => (
                    <div key={cls} className={`absolute h-6 w-6 border-violet-400/40 ${cls} pointer-events-none`} />
                  ))}
                </>
              )}
            </div>

            {/* Controls */}
            <div className="flex items-center justify-between mt-4 flex-wrap gap-3">
              <Button
                onClick={() => setRunning((v) => !v)}
                variant={running ? "danger" : "primary"}
              >
                {running ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {running ? "Pause" : "Start"}
              </Button>

              <div className="flex items-center gap-2">
                <span className="text-xs text-foreground/40 mr-2">Rate</span>
                {FPS_CHOICES.map((f) => (
                  <button
                    key={f}
                    onClick={() => setFps(f)}
                    className={cn(
                      "px-3 py-1.5 rounded-full text-xs font-medium transition-all",
                      fps === f
                        ? "bg-violet-500/20 text-violet-200 border border-violet-500/30"
                        : "text-foreground/40 hover:text-foreground hover:bg-white/5"
                    )}
                  >
                    {f} fps
                  </button>
                ))}
              </div>
            </div>
          </Card>

          {/* ── Right: live stats ── */}
          <div className="space-y-4">
            {/* Session counters */}
            <Card className="p-5">
              <CardTitle icon={<Activity className="h-4 w-4" />} className="mb-4">Session</CardTitle>
              <div className="grid grid-cols-2 gap-3">
                {[
                  ["Frames",  frames],
                  ["Dropped", droppedFrames],
                  ["Hazard",  hazardCount],
                  ["Low-conf", lowConf],
                ].map(([k, v]) => (
                  <div key={k as string} className="rounded-xl bg-white/[0.03] p-3">
                    <p className="text-[10px] uppercase tracking-widest text-foreground/40">{k as string}</p>
                    <p className="text-2xl font-semibold tabular-nums mt-1">{v as number}</p>
                  </div>
                ))}
              </div>
            </Card>

            {/* Latency */}
            {perf && (
              <Card className="p-5">
                <CardTitle icon={<Gauge className="h-4 w-4" />} className="mb-4">Latency / throughput</CardTitle>
                <div className="space-y-3">
                  {[
                    ["Mean",     perf.mean,   "ms"],
                    ["P50",      perf.p50,    "ms"],
                    ["P90",      perf.p90,    "ms"],
                    ["P99",      perf.p99,    "ms"],
                    ["Throughput", perf.throughput_fps, "fps"],
                  ].map(([k, v, unit]) => (
                    <div key={k as string} className="flex items-center justify-between text-sm">
                      <span className="text-foreground/50 text-xs uppercase tracking-widest">{k as string}</span>
                      <span className="font-mono font-medium">
                        {typeof v === "number" ? v.toFixed(2) : v} <span className="text-foreground/30 text-xs">{unit}</span>
                      </span>
                    </div>
                  ))}
                </div>
                <p className="text-[10px] text-foreground/30 mt-3">
                  Rolling window · {perf.samples} samples
                </p>
              </Card>
            )}

            {/* Last prediction */}
            {last && last.detections.length > 0 && (
              <Card className="p-5">
                <CardTitle icon={<Zap className="h-4 w-4" />} className="mb-4">Last frame</CardTitle>
                <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                  {last.detections.map((d) => {
                    const col = CATEGORY_COLOR[d.category] ?? "#a78bfa";
                    return (
                      <div
                        key={d.id}
                        className="flex items-center justify-between rounded-lg bg-white/[0.03] px-3 py-2 text-xs"
                      >
                        <span style={{ color: col }} className="font-medium">
                          {CATEGORY_LABEL[d.category] ?? d.category}
                        </span>
                        <div className="flex items-center gap-2">
                          <span
                            className="font-mono tabular-nums"
                            style={{ color: confColor(d.classification_confidence) }}
                          >
                            {fmt(d.classification_confidence)}%
                          </span>
                          <span className="text-foreground/40" style={{ color: BIN_COLOR[d.bin] }}>
                            {BIN_EMOJI[d.bin]}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-3 flex gap-2 flex-wrap">
                  {hazardCount > 0 && (
                    <Badge variant="hazard"><ShieldAlert className="h-3 w-3" /> {hazardCount} hazard</Badge>
                  )}
                  {lowConf > 0 && (
                    <Badge variant="warning"><AlertTriangle className="h-3 w-3" /> {lowConf} low-conf</Badge>
                  )}
                </div>
              </Card>
            )}

            {error && (
              <div className="rounded-xl border border-red-500/30 bg-red-500/8 px-4 py-3 text-xs text-red-300">
                {error}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
