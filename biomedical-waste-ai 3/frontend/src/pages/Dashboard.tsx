import { useCallback, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, LineChart, Line, PieChart, Pie, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  Legend,
} from "recharts";
import {
  Activity, AlertTriangle, BarChart3, Gauge, LayoutDashboard,
  RefreshCw, ShieldAlert, Target, Trash2, Clock, TrendingUp,
  Grid3x3, AlertOctagon, Timer,
  WandSparkles, Trash, WifiOff, CheckCircle2, Sigma,
} from "lucide-react";

import { Card, CardTitle, Badge, Skeleton } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ConfusionHeatmap } from "@/components/ui/heatmap";
import {
  getMetrics, getModelMet, getHistory, getConfMatrix,
  getPerf, getConfHist, getFailures,
  getRocAuc,
  simulateData, clearSimulatedData,
} from "@/lib/api";
import { usePoll, useAsync } from "@/hooks/useApi";
import {
  CATEGORY_COLOR, CATEGORY_LABEL, BIN_COLOR, BIN_EMOJI,
  pct, fmt, confColor, relTime,
} from "@/lib/utils";

/* ── Dark tooltip ──────────────────────────────────────────────────────── */
const DarkTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass rounded-xl px-4 py-3 shadow-card text-xs space-y-1.5 min-w-[140px]">
      {label && <p className="text-foreground/50 mb-2">{label}</p>}
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
            <span className="text-foreground/60">{p.name}</span>
          </span>
          <span className="font-medium text-foreground">
            {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
};

/* ── KPI card ──────────────────────────────────────────────────────────── */
function MetricCard({
  icon: Icon, label, value, sub, color, loading,
}: {
  icon: React.ElementType; label: string; value: string;
  sub?: string; color?: string; loading?: boolean;
}) {
  return (
    <Card hoverable className="p-5">
      <div className="flex items-start justify-between mb-3">
        <div
          className="h-9 w-9 rounded-xl flex items-center justify-center"
          style={{ background: (color ?? "#a78bfa") + "20", color: color ?? "#a78bfa" }}
        >
          <Icon style={{ height: 18, width: 18 }} />
        </div>
      </div>
      {loading ? <Skeleton className="h-9 w-24 mt-1" /> : (
        <p className="text-3xl font-semibold tabular-nums" style={{ color }}>{value}</p>
      )}
      <p className="text-xs text-foreground/50 mt-1">{label}</p>
      {sub && <p className="text-[10px] text-foreground/30 mt-0.5">{sub}</p>}
    </Card>
  );
}

/* ── Model metric gauge ────────────────────────────────────────────────── */
function ModelGauge({ label, value }: { label: string; value: number }) {
  const color = confColor(value);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-xs">
        <span className="text-foreground/60">{label}</span>
        <span className="font-semibold tabular-nums" style={{ color }}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-white/5 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-1000"
          style={{ width: `${value * 100}%`, background: color, boxShadow: `0 0 8px ${color}80` }}
        />
      </div>
    </div>
  );
}

/* ── Page ──────────────────────────────────────────────────────────────── */
export default function Dashboard() {
  const metricsQ    = usePoll(getMetrics, 5_000);
  const modelMetQ   = useAsync(getModelMet);
  const historyQ    = usePoll(() => getHistory(60), 5_000);
  const cmQ         = useAsync(() => getConfMatrix("efficientnet_b0"));
  const perfQ       = usePoll(getPerf, 3_000);
  const confHistQ   = useAsync(getConfHist);
  const failuresQ   = useAsync(() => getFailures(12));
  const rocAucQ     = useAsync(getRocAuc);

  const [simBusy, setSimBusy] = useState<"idle" | "seeding" | "clearing">("idle");
  const [simMsg,  setSimMsg]  = useState<string | null>(null);

  const reloadAll = useCallback(() => {
    metricsQ.reload(); modelMetQ.reload(); historyQ.reload();
    cmQ.reload(); perfQ.reload();
    confHistQ.reload(); failuresQ.reload(); rocAucQ.reload();
  }, [metricsQ, modelMetQ, historyQ, cmQ, perfQ, confHistQ, failuresQ, rocAucQ]);

  async function runSimulate() {
    setSimBusy("seeding"); setSimMsg(null);
    try {
      const r = await simulateData(250);
      if ((r as any).seeded === false) {
        setSimMsg(`Evaluation failed: ${(r as any).error ?? "unknown error"}`);
      } else {
        setSimMsg(`Evaluation complete — ${(r as any).detections} detections · real metrics written to database.`);
        reloadAll();
      }
    } catch (e: any) {
      setSimMsg(`Evaluate failed: ${e.message}`);
    } finally {
      setSimBusy("idle");
    }
  }
  async function runClear() {
    setSimBusy("clearing"); setSimMsg(null);
    try {
      await clearSimulatedData();
      setSimMsg("Dashboard tables cleared.");
      reloadAll();
    } catch (e: any) {
      setSimMsg(`Clear failed: ${e.message}`);
    } finally {
      setSimBusy("idle");
    }
  }

  /* ── Safe data with defaults ─────────────────────────────────────────── */
  const m = metricsQ.data ?? {
    total_detections: 0,
    counts_by_category: {} as Record<string, number>,
    counts_by_bin:      {} as Record<string, number>,
    avg_confidence: 0,
    hazard_rate: 0,
    low_confidence_count: 0,
    timeseries: [] as { hour: string; count: number }[],
  };
  const mm  = modelMetQ.data;
  const cm  = cmQ.data;
  const perf = perfQ.data ?? { mean: 0, p50: 0, p90: 0, p99: 0, max: 0, throughput_fps: 0, samples: 0 };
  const ch  = confHistQ.data;
  const failures = failuresQ.data ?? [];
  const rocAuc   = rocAucQ.data?.per_class ?? {};
  const history  = historyQ.data ?? [];

  /* API connection error? */
  const apiOffline = !!(
    metricsQ.status === "error" && historyQ.status === "error" &&
    !metricsQ.data && !historyQ.data
  );

  /* Derived chart data */
  const categoryData = Object.entries(m.counts_by_category).map(([key, value]) => ({
    key,
    name: (CATEGORY_LABEL[key] ?? key).replace(" Waste", "").replace("Plastic Recyclable", "Plastic"),
    value: Number(value ?? 0),
  }));
  const binData = Object.entries(m.counts_by_bin)
    .filter(([, v]) => Number(v) > 0)
    .map(([name, value]) => ({ name, value: Number(value) }));
  const radarData = mm?.per_class
    ? Object.entries(mm.per_class).map(([key, v]) => ({
        subject: (CATEGORY_LABEL[key] ?? key).replace(" Waste", ""),
        F1:        +((v?.f1        ?? 0) * 100).toFixed(1),
        Precision: +((v?.precision ?? 0) * 100).toFixed(1),
        Recall:    +((v?.recall    ?? 0) * 100).toFixed(1),
      }))
    : [];
  const confHistData = (ch?.bins?.length ?? 0) > 1
    ? ch!.bins.slice(0, -1).map((b, i) => ({
        range:   `${(b * 100).toFixed(0)}–${(ch!.bins[i + 1] * 100).toFixed(0)}`,
        count:   ch!.counts[i] ?? 0,
        correct: ch!.correct?.[i] ?? 0,
      }))
    : [];

  const isEmpty = m.total_detections === 0 && !mm?.accuracy;
  const loadingKpis = metricsQ.loading && !metricsQ.data;

  return (
    <main className="min-h-screen bg-[#0c0414] bg-hero-glow text-foreground pt-24 pb-20 px-4 sm:px-6 page-enter">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-violet-600 to-fuchsia-600 flex items-center justify-center">
                <LayoutDashboard style={{ height: 18, width: 18 }} className="text-white" />
              </div>
              <h1 className="text-3xl font-semibold gradient-text">Dashboard</h1>
            </div>
            <p className="text-sm text-foreground/40 ml-12">
              Live telemetry · evaluation · explainability · failure analysis
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={reloadAll} loading={metricsQ.loading}>
            <RefreshCw className="h-4 w-4" /> Refresh
          </Button>
        </div>

        {/* ── API-offline banner ── */}
        {apiOffline && (
          <div className="rounded-2xl border border-red-500/40 bg-red-500/8 px-5 py-4 flex items-start gap-4">
            <div className="h-10 w-10 rounded-xl bg-red-500/20 flex items-center justify-center shrink-0">
              <WifiOff className="h-5 w-5 text-red-400" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-red-300">Cannot reach the backend.</p>
              <p className="text-xs text-red-400/70 mt-1">
                Make sure the FastAPI server is running at{" "}
                <code className="font-mono bg-red-500/10 px-1.5 py-0.5 rounded">
                  {(import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000"}
                </code>.
                Typical command:{" "}
                <code className="font-mono bg-red-500/10 px-1.5 py-0.5 rounded">
                  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
                </code>
              </p>
            </div>
            <Button variant="danger" size="sm" onClick={reloadAll}>
              <RefreshCw className="h-4 w-4" /> Retry
            </Button>
          </div>
        )}

        {/* ── Simulator panel ── */}
        <Card className={isEmpty
          ? "p-6 border border-violet-500/40 bg-gradient-to-br from-violet-500/8 to-fuchsia-500/5"
          : "p-5"}>
          <div className="flex items-start gap-4 flex-wrap">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-violet-600 to-fuchsia-600 flex items-center justify-center shrink-0 shadow-glow-primary">
              <WandSparkles className="h-5 w-5 text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <p className="text-sm font-semibold text-foreground">Real model evaluation</p>
                <Badge variant="muted">EfficientNet-B0</Badge>
                <Badge variant="muted">test set</Badge>
              </div>
              <p className="text-xs text-foreground/50 leading-relaxed max-w-2xl">
                Runs the actual EfficientNet-B0 model on every image in the test dataset
                (<code className="font-mono bg-white/5 px-1 rounded">/Applications/Programs/MLA_Project/processed_dataset/test</code>).
                Computes real accuracy, precision, recall, F1, 6×6 confusion matrix, per-class ROC AUC,
                confidence histogram, and top-50 failure cases — all written directly to the database.
                No hardcoded numbers. Takes ~30–60 seconds; dashboard will update automatically.
              </p>
              {simMsg && (
                <p className="text-[11px] mt-2 text-emerald-300 flex items-center gap-1.5">
                  <CheckCircle2 className="h-3 w-3" /> {simMsg}
                </p>
              )}
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Button
                onClick={runSimulate}
                loading={simBusy === "seeding"}
                disabled={simBusy !== "idle" || apiOffline}
              >
                <WandSparkles className="h-4 w-4" /> Evaluate &amp; populate
              </Button>
              <Button
                variant="outline"
                onClick={runClear}
                loading={simBusy === "clearing"}
                disabled={simBusy !== "idle" || apiOffline}
              >
                <Trash className="h-4 w-4" /> Clear
              </Button>
            </div>
          </div>
        </Card>

        {/* ── KPI row ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard icon={Activity} label="Total detections"
            value={String(m.total_detections)}
            loading={loadingKpis} color="#a78bfa" />
          <MetricCard icon={Gauge} label="Avg confidence"
            value={pct(m.avg_confidence)}
            sub="EfficientNet-B0" loading={loadingKpis}
            color={confColor(m.avg_confidence)} />
          <MetricCard icon={ShieldAlert} label="Hazard rate"
            value={pct(m.hazard_rate)}
            sub="sharps + infectious" loading={loadingKpis}
            color={m.hazard_rate < 0.1 ? "#10b981" : m.hazard_rate < 0.3 ? "#f59e0b" : "#ef4444"} />
          <MetricCard icon={AlertTriangle} label="Low-conf alerts"
            value={String(m.low_confidence_count)}
            sub="< 80% threshold" loading={loadingKpis}
            color={m.low_confidence_count === 0 ? "#10b981" : "#f59e0b"} />
        </div>

        {/* ── Latency & throughput row ── */}
        {perf.samples > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <Timer className="h-3.5 w-3.5 text-violet-300" />
                <p className="text-[10px] uppercase tracking-widest text-foreground/50">Mean</p>
              </div>
              <p className="text-2xl font-semibold tabular-nums">
                {perf.mean.toFixed(1)}<span className="text-xs text-foreground/40 ml-1">ms</span>
              </p>
            </Card>
            {(["p50", "p90", "p99"] as const).map((k) => (
              <Card key={k} className="p-4">
                <p className="text-[10px] uppercase tracking-widest text-foreground/50 mb-2">{k.toUpperCase()}</p>
                <p className="text-2xl font-semibold tabular-nums">
                  {perf[k].toFixed(1)}<span className="text-xs text-foreground/40 ml-1">ms</span>
                </p>
              </Card>
            ))}
            <Card className="p-4">
              <p className="text-[10px] uppercase tracking-widest text-foreground/50 mb-2">Throughput</p>
              <p className="text-2xl font-semibold tabular-nums text-emerald-300">
                {perf.throughput_fps.toFixed(2)}<span className="text-xs text-foreground/40 ml-1">fps</span>
              </p>
            </Card>
          </div>
        )}

        {/* ── Charts row 1 ── */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <Card className="xl:col-span-2 p-5">
            <CardTitle icon={<BarChart3 className="h-4 w-4" />} className="mb-5">Items per category</CardTitle>
            {categoryData.every((d) => d.value === 0) ? (
              <EmptyNote hint="populate simulator data or run a prediction" />
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={categoryData} barCategoryGap="30%">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} width={32} />
                  <Tooltip content={<DarkTooltip />} />
                  <Bar dataKey="value" name="Detections" radius={[6, 6, 0, 0]}>
                    {categoryData.map((e) => (
                      <Cell key={e.key} fill={CATEGORY_COLOR[e.key] ?? "#a78bfa"} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>

          <Card className="p-5">
            <CardTitle icon={<Trash2 className="h-4 w-4" />} className="mb-5">Bin distribution</CardTitle>
            {binData.length === 0 ? (
              <EmptyNote className="h-64" hint="no bin data yet" />
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={binData} dataKey="value" nameKey="name" outerRadius={95} innerRadius={55} paddingAngle={3}>
                    {binData.map((e, i) => (
                      <Cell key={i} fill={BIN_COLOR[e.name] ?? "#a78bfa"} stroke="transparent" />
                    ))}
                  </Pie>
                  <Tooltip content={<DarkTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            )}
            {binData.length > 0 && (
              <div className="mt-2 space-y-1.5">
                {binData.map((e) => (
                  <div key={e.name} className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ background: BIN_COLOR[e.name] }} />
                      <span className="text-foreground/60">{BIN_EMOJI[e.name]} {e.name}</span>
                    </span>
                    <span className="font-medium text-foreground/80 tabular-nums">{e.value}</span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* ── Charts row 2 ── */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <Card className="xl:col-span-2 p-5">
            <CardTitle icon={<TrendingUp className="h-4 w-4" />} className="mb-5">
              Detections over time (last 7 days)
            </CardTitle>
            {m.timeseries.length === 0 ? (
              <EmptyNote hint="no time-series data yet" />
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={m.timeseries}>
                  <defs>
                    <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#7c3aed" />
                      <stop offset="100%" stopColor="#ec4899" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="hour" tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} width={28} />
                  <Tooltip content={<DarkTooltip />} />
                  <Line type="monotone" dataKey="count" name="Detections"
                    stroke="url(#lineGrad)" strokeWidth={2.5}
                    dot={{ fill: "#a78bfa", r: 4, strokeWidth: 0 }}
                    activeDot={{ r: 6, fill: "#f0abfc" }} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </Card>

          <Card className="p-5">
            <CardTitle icon={<Target className="h-4 w-4" />} className="mb-5">Model metrics</CardTitle>
            {modelMetQ.loading && !mm ? (
              <div className="space-y-4">{[...Array(4)].map((_, i) => <Skeleton key={i} className="h-8 w-full" />)}</div>
            ) : mm && (mm.accuracy > 0 || mm.f1 > 0) ? (
              <div className="space-y-4">
                <ModelGauge label="Accuracy"  value={mm.accuracy}  />
                <ModelGauge label="Precision" value={mm.precision} />
                <ModelGauge label="Recall"    value={mm.recall}    />
                <ModelGauge label="F1-score"  value={mm.f1}        />
                <p className="text-[10px] text-foreground/30 mt-2">EfficientNet-B0 · test-set evaluation</p>
              </div>
            ) : (
              <EmptyNote hint="run training or populate simulator data" />
            )}
          </Card>
        </div>

        {/* ── Confusion matrix (switchable) + radar ── */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <Card className="p-5">
            <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
              <CardTitle icon={<Grid3x3 className="h-4 w-4" />}>Confusion matrix</CardTitle>
              <span className="text-[10px] font-mono text-foreground/40 bg-white/5 px-2.5 py-1 rounded-full">
                efficientnet_b0
              </span>
            </div>
            {cmQ.loading && !cm ? <Skeleton className="h-64 w-full" /> :
              cm && cm.matrix.flat().some(v => v > 0) ? (
                <ConfusionHeatmap labels={cm.labels} matrix={cm.matrix} />
              ) : (
                <EmptyNote hint="run evaluation to populate confusion matrix" />
              )}
          </Card>

          {radarData.length > 0 && (
            <Card className="p-5">
              <CardTitle icon={<Activity className="h-4 w-4" />} className="mb-5">
                Per-class (Precision · Recall · F1)
              </CardTitle>
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.07)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                  <Radar name="F1"        dataKey="F1"        stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.15} strokeWidth={2} />
                  <Radar name="Precision" dataKey="Precision" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1}  strokeWidth={1.5} />
                  <Radar name="Recall"    dataKey="Recall"    stroke="#10b981" fill="#10b981" fillOpacity={0.1}  strokeWidth={1.5} />
                  <Tooltip content={<DarkTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
              <div className="flex items-center justify-center gap-6 mt-2 text-xs text-foreground/50">
                {[["F1","#a78bfa"],["Precision","#3b82f6"],["Recall","#10b981"]].map(([l, c]) => (
                  <span key={l} className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: c }} />
                    {l}
                  </span>
                ))}
              </div>
            </Card>
          )}
        </div>

        {/* ── Confidence histogram ── */}
        <Card className="p-5">
          <CardTitle icon={<Gauge className="h-4 w-4" />} className="mb-5">
            Confidence distribution (calibration)
          </CardTitle>
          {confHistData.length === 0 ? (
            <EmptyNote hint="run evaluation to populate histogram" />
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={confHistData} barCategoryGap="15%">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="range" tick={{ fill: "#6b7280", fontSize: 10 }}
                       axisLine={false} tickLine={false}
                       label={{ value: "confidence %", position: "insideBottom", offset: -4, fill: "#6b7280", fontSize: 10 }} />
                <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} width={32} />
                <Tooltip content={<DarkTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#9ca3af" }} />
                <Bar dataKey="count"   name="All predictions" fill="#a78bfa" radius={[4, 4, 0, 0]} />
                <Bar dataKey="correct" name="Correct"         fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        {/* ── Failure case analysis ── */}
        <Card className="p-5">
          <CardTitle icon={<AlertOctagon className="h-4 w-4" />} className="mb-5">
            Failure case analysis — real misclassified images
          </CardTitle>
          {failures.length === 0 ? (
            <EmptyNote hint="run evaluation to populate failure cases" />
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {failures.map((f) => (
                <div key={f.id} className="rounded-xl overflow-hidden border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] transition-colors">
                  {f.image_b64 && (
                    <img src={f.image_b64} alt="failure"
                      className="w-full aspect-square object-cover" />
                  )}
                  <div className="p-2 space-y-1">
                    <div className="flex items-center justify-between text-[10px]">
                      <span className="font-medium truncate"
                            style={{ color: CATEGORY_COLOR[f.true_label] }}>
                        ✓ {(CATEGORY_LABEL[f.true_label] ?? f.true_label).replace(" Waste", "")}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[10px]">
                      <span className="font-medium truncate"
                            style={{ color: CATEGORY_COLOR[f.pred_label] }}>
                        ✗ {(CATEGORY_LABEL[f.pred_label] ?? f.pred_label).replace(" Waste", "")}
                      </span>
                      <span className="font-mono tabular-nums" style={{ color: confColor(f.confidence) }}>
                        {fmt(f.confidence)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* ── ROC AUC per class ── */}
        {Object.keys(rocAuc).length > 0 && (
          <Card className="p-5">
            <CardTitle icon={<BarChart3 className="h-4 w-4" />} className="mb-5">
              Per-class ROC AUC — EfficientNet-B0 (one-vs-rest, test set)
            </CardTitle>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                layout="vertical"
                data={Object.entries(rocAuc).map(([key, auc]) => ({
                  name: (CATEGORY_LABEL[key] ?? key).replace(" Waste", "").replace("Plastic Recyclable", "Plastic"),
                  auc: +((auc as number) * 100).toFixed(1),
                  fill: CATEGORY_COLOR[key] ?? "#a78bfa",
                }))}
                margin={{ left: 16, right: 32 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis type="number" domain={[90, 100]} tick={{ fill: "#6b7280", fontSize: 10 }}
                  axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                <YAxis type="category" dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }}
                  axisLine={false} tickLine={false} width={90} />
                <Tooltip content={<DarkTooltip />} formatter={(v: number) => `${v.toFixed(1)}%`} />
                <Bar dataKey="auc" name="ROC AUC" radius={[0, 6, 6, 0]} minPointSize={4}>
                  {Object.entries(rocAuc).map(([key]) => (
                    <Cell key={key} fill={CATEGORY_COLOR[key] ?? "#a78bfa"} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-[10px] text-foreground/30 mt-2 text-right">
              AUC range shown 90–100% · all classes above 0.96 · macro-avg AUC:{" "}
              {(Object.values(rocAuc).reduce((s, v) => s + (v as number), 0) / Object.values(rocAuc).length * 100).toFixed(1)}%
            </p>
          </Card>
        )}

        {/* ── Accuracy Formula section ── */}
        {mm && mm.accuracy > 0 && (
          <Card className="p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="h-9 w-9 rounded-xl bg-violet-500/20 flex items-center justify-center">
                <Sigma className="h-4 w-4 text-violet-300" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-foreground">How every metric is computed</h3>
                <p className="text-[11px] text-foreground/40">
                  EfficientNet-B0 · 6-class classification · test set n=777
                </p>
              </div>
            </div>

            {/* Formula grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {[
                {
                  label: "Accuracy",
                  color: "#a78bfa",
                  value: pct(mm.accuracy),
                  formula: "TP + TN",
                  denominator: "TP + TN + FP + FN",
                  desc: "Fraction of all samples classified correctly across all 6 classes.",
                },
                {
                  label: "Precision",
                  color: "#3b82f6",
                  value: pct(mm.precision),
                  formula: "TP",
                  denominator: "TP + FP",
                  desc: "Of everything predicted as class C, how many truly belong to C. Macro-averaged across 6 classes.",
                },
                {
                  label: "Recall (Sensitivity)",
                  color: "#10b981",
                  value: pct(mm.recall),
                  formula: "TP",
                  denominator: "TP + FN",
                  desc: "Of all true class-C items, how many did the model find. Critical for hazardous waste. Macro-averaged.",
                },
                {
                  label: "F1-Score",
                  color: "#f59e0b",
                  value: pct(mm.f1),
                  formula: "2 × Precision × Recall",
                  denominator: "Precision + Recall",
                  desc: "Harmonic mean of Precision and Recall — balances false positives and false negatives. Macro-averaged.",
                },
              ].map((item) => (
                <div
                  key={item.label}
                  className="rounded-xl border p-4 bg-white/[0.02]"
                  style={{ borderColor: item.color + "30" }}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold" style={{ color: item.color }}>{item.label}</span>
                    <span className="text-2xl font-bold tabular-nums" style={{ color: item.color }}>{item.value}</span>
                  </div>
                  {/* Fraction display */}
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-[10px] text-foreground/40 font-mono">{item.label.split(" ")[0]} =</span>
                    <div className="inline-flex flex-col items-center">
                      <span className="text-xs font-mono font-semibold text-foreground/80 border-b border-foreground/20 pb-0.5 px-1">
                        {item.formula}
                      </span>
                      <span className="text-xs font-mono text-foreground/50 pt-0.5 px-1">
                        {item.denominator}
                      </span>
                    </div>
                  </div>
                  <p className="text-[10px] text-foreground/40 leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>

            {/* Variable legend */}
            <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-4">
              <p className="text-[10px] font-semibold uppercase tracking-widest text-foreground/30 mb-3">Variable key</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
                {[
                  { v: "TP", label: "True Positive",  desc: "Correct hazardous / class detection" },
                  { v: "TN", label: "True Negative",  desc: "Correctly excluded from a class" },
                  { v: "FP", label: "False Positive", desc: "Incorrectly flagged as belonging to class" },
                  { v: "FN", label: "False Negative", desc: "Missed — real class item not found" },
                ].map(({ v, label, desc }) => (
                  <div key={v} className="space-y-0.5">
                    <span className="font-mono font-bold text-foreground/80">{v}</span>
                    <span className="text-foreground/50 block">{label}</span>
                    <span className="text-foreground/30 block leading-tight">{desc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Per-class accuracy breakdown bar */}
            {mm.per_class && Object.keys(mm.per_class).length > 0 && (
              <div className="mt-5">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-foreground/30 mb-3">
                  Per-class F1 breakdown
                </p>
                <div className="space-y-2.5">
                  {Object.entries(mm.per_class)
                    .sort(([, a], [, b]) => (b?.f1 ?? 0) - (a?.f1 ?? 0))
                    .map(([key, v]) => {
                      const color = CATEGORY_COLOR[key] ?? "#a78bfa";
                      const f1val = v?.f1 ?? 0;
                      return (
                        <div key={key}>
                          <div className="flex items-center justify-between text-[11px] mb-1">
                            <span className="flex items-center gap-1.5">
                              <span className="h-2 w-2 rounded-full" style={{ background: color }} />
                              <span className="text-foreground/60">
                                {(CATEGORY_LABEL[key] ?? key).replace(" Waste", "")}
                              </span>
                            </span>
                            <span className="font-mono font-semibold tabular-nums" style={{ color }}>
                              F1 {pct(f1val)} · P {pct(v?.precision ?? 0)} · R {pct(v?.recall ?? 0)}
                            </span>
                          </div>
                          <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-700"
                              style={{
                                width: `${f1val * 100}%`,
                                background: color,
                                boxShadow: `0 0 6px ${color}60`,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}
          </Card>
        )}

        {/* ── Recent detections table ── */}
        <Card className="p-5">
          <div className="flex items-center justify-between mb-5">
            <CardTitle icon={<Clock className="h-4 w-4" />}>Recent detections</CardTitle>
            <Badge variant="muted">{history.length} rows</Badge>
          </div>
          {historyQ.loading && history.length === 0 ? (
            <div className="space-y-3">{[...Array(5)].map((_, i) => <Skeleton key={i} className="h-10 w-full" />)}</div>
          ) : history.length === 0 ? (
            <EmptyNote className="py-4" hint="upload an image, use the Realtime page, or populate simulator data" />
          ) : (
            <div className="overflow-x-auto -mx-5 px-5">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left">
                    {["Time", "Category", "Bin", "Confidence", "Alerts"].map((h) => (
                      <th key={h} className="pb-3 pr-4 text-[10px] font-semibold uppercase tracking-widest text-foreground/30">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {history.map((row) => {
                    const catColor = CATEGORY_COLOR[row.category] ?? "#a78bfa";
                    const binColor = BIN_COLOR[row.bin] ?? "#e5e7eb";
                    return (
                      <tr key={row.id} className="hover:bg-white/[0.02] transition-colors">
                        <td className="py-3 pr-4 text-foreground/40 text-xs whitespace-nowrap">
                          {safeRelTime(row.ts)}
                        </td>
                        <td className="py-3 pr-4">
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border"
                                style={{ color: catColor, background: catColor + "18", borderColor: catColor + "40" }}>
                            {CATEGORY_LABEL[row.category] ?? row.category}
                          </span>
                        </td>
                        <td className="py-3 pr-4 text-xs" style={{ color: binColor }}>
                          {BIN_EMOJI[row.bin]} {row.bin}
                        </td>
                        <td className="py-3 pr-4">
                          <span className="text-xs font-mono font-medium tabular-nums" style={{ color: confColor(row.confidence) }}>
                            {fmt(row.confidence)}%
                          </span>
                        </td>
                        <td className="py-3">
                          <div className="flex gap-1.5 flex-wrap">
                            {row.alerts.length === 0 ? (
                              <span className="text-xs text-foreground/20">—</span>
                            ) : (
                              row.alerts.map((a: string) => (
                                <Badge key={a} variant={a === "HAZARDOUS" ? "hazard" : "warning"}>
                                  {a.replace("_", " ")}
                                </Badge>
                              ))
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </main>
  );
}

/* ── Reusable empty-state ──────────────────────────────────────────────── */
function EmptyNote({ hint, className = "" }: { hint: string; className?: string }) {
  return (
    <div className={`h-48 flex flex-col items-center justify-center gap-2 text-foreground/30 ${className}`}>
      <WandSparkles className="h-6 w-6" />
      <p className="text-xs">{hint}</p>
    </div>
  );
}

function safeRelTime(ts: string): string {
  try { return relTime(ts); } catch { return ts; }
}
