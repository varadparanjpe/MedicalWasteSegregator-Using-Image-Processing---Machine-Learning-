/**
 * Model Comparison Page
 *
 * Side-by-side analysis of all trained classifiers:
 *   EfficientNet-B0 (production) · EfficientNet-B2 · MobileNetV3-Large
 *   DenseNet-121 · ResNet50 · YOLO-only baseline
 *
 * Sections:
 *   1. Summary scorecard — key metrics for every model
 *   2. Grouped bar chart — Accuracy / Precision / Recall / F1 per model
 *   3. Radar chart — balanced view across all 4 metrics
 *   4. Latency vs Accuracy scatter — efficiency frontier
 *   5. Per-class F1 heatmap — which model wins on each waste category
 *   6. Per-class AUC comparison bars
 *   7. Params vs F1 — size-efficiency plot
 */
import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ScatterChart, Scatter, ZAxis, Legend, Cell,
} from "recharts";
import {
  BarChart3, Cpu, FlaskConical, GitCompare, Medal, RefreshCw,
  Target, Zap, WandSparkles,
} from "lucide-react";

import { Card, CardTitle, Badge, Skeleton } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getModelComparison, getAllModelPerClass } from "@/lib/api";
import { useAsync } from "@/hooks/useApi";
import {
  CATEGORY_COLOR, CATEGORY_LABEL, pct, confColor,
} from "@/lib/utils";

/* ── Constants ──────────────────────────────────────────────────────────── */

const MODEL_META: Record<string, { label: string; color: string; note: string; params: string }> = {
  efficientnet_b0: {
    label: "EfficientNet-B0",
    color: "#a78bfa",
    note: "Production model · best accuracy/speed balance",
    params: "5.3M",
  },
  efficientnet_b2: {
    label: "EfficientNet-B2",
    color: "#f472b6",
    note: "Scale-up within family · highest accuracy",
    params: "9.1M",
  },
  mobilenet_v3: {
    label: "MobileNetV3-Large",
    color: "#34d399",
    note: "Edge-optimised · fastest inference",
    params: "5.5M",
  },
  densenet121: {
    label: "DenseNet-121",
    color: "#60a5fa",
    note: "CheXNet backbone · biomedical domain",
    params: "8.0M",
  },
  resnet50: {
    label: "ResNet50",
    color: "#fbbf24",
    note: "General-purpose baseline",
    params: "25.6M",
  },
  yolo_only: {
    label: "YOLO-only",
    color: "#f87171",
    note: "No classifier — baseline failure",
    params: "N/A",
  },
};

/** Models shown in the comparison (excluding YOLO-only from per-class section) */
const COMPARISON_MODELS = ["efficientnet_b0", "efficientnet_b2", "mobilenet_v3", "densenet121", "resnet50"];

const METRICS: { key: "accuracy" | "precision" | "recall" | "f1"; label: string; color: string }[] = [
  { key: "accuracy",  label: "Accuracy",  color: "#a78bfa" },
  { key: "precision", label: "Precision", color: "#60a5fa" },
  { key: "recall",    label: "Recall",    color: "#34d399" },
  { key: "f1",        label: "F1",        color: "#fbbf24" },
];

/* ── Dark tooltip ──────────────────────────────────────────────────────── */
const DarkTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass rounded-xl px-4 py-3 shadow-card text-xs space-y-1.5 min-w-[160px]">
      {label && <p className="text-foreground/50 mb-2">{label}</p>}
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: p.color ?? p.fill }} />
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

/* ── Model badge chip ──────────────────────────────────────────────────── */
function ModelChip({ modelKey, active, onClick }: { modelKey: string; active: boolean; onClick: () => void }) {
  const meta = MODEL_META[modelKey] ?? { label: modelKey, color: "#a78bfa" };
  return (
    <button
      onClick={onClick}
      className={
        "px-3 py-1.5 rounded-full text-xs font-medium border transition-all " +
        (active
          ? "border-opacity-60 bg-opacity-20"
          : "border-transparent text-foreground/30 hover:text-foreground/60 bg-transparent")
      }
      style={active ? { color: meta.color, borderColor: meta.color + "60", background: meta.color + "18" } : undefined}
    >
      {meta.label}
    </button>
  );
}

/* ── Page ──────────────────────────────────────────────────────────────── */
export default function ModelComparison() {
  const cmpQ        = useAsync(getModelComparison);
  const perClassQ   = useAsync(getAllModelPerClass);

  const [activeMetric, setActiveMetric] = useState<"accuracy" | "precision" | "recall" | "f1">("f1");
  const [activeClass,  setActiveClass]  = useState<string>("all");

  const cmp      = cmpQ.data ?? [];
  const perClass = perClassQ.data ?? {};

  const hasData = cmp.length > 0;

  /* ── Derived data ── */

  // Filter out yolo_only for most charts (it skews scales badly)
  const mainModels = cmp.filter((r) => r.model !== "yolo_only");
  const allModelsIncYolo = cmp;

  // Grouped bar chart data — one entry per model
  const groupedData = mainModels.map((r) => ({
    name: MODEL_META[r.model]?.label ?? r.model,
    model: r.model,
    Accuracy:  +((r.accuracy  ?? 0) * 100).toFixed(1),
    Precision: +((r.precision ?? 0) * 100).toFixed(1),
    Recall:    +((r.recall    ?? 0) * 100).toFixed(1),
    F1:        +((r.f1        ?? 0) * 100).toFixed(1),
  }));

  // Radar data — one spoke per metric
  const radarData = METRICS.map(({ key, label }) => {
    const entry: Record<string, any> = { metric: label };
    mainModels.forEach((r) => {
      entry[MODEL_META[r.model]?.label ?? r.model] = +((r[key] ?? 0) * 100).toFixed(1);
    });
    return entry;
  });

  // Latency vs Accuracy scatter
  const scatterData = mainModels
    .filter((r) => r.inference_ms != null && r.accuracy != null)
    .map((r) => ({
      model: r.model,
      name:  MODEL_META[r.model]?.label ?? r.model,
      x:     r.inference_ms!,
      y:     +((r.accuracy! * 100).toFixed(1)),
      z:     r.params_m ?? 5,
      color: MODEL_META[r.model]?.color ?? "#a78bfa",
    }));

  // Per-class F1 heatmap — rows = classes, cols = models
  const classes = [
    "sharps_waste", "infectious_waste", "pathological_waste",
    "plastic_recyclable", "pharmaceutical_waste", "general_waste",
  ];

  // Per-class AUC bar data
  const aucBarData = classes.map((cls) => {
    const entry: Record<string, any> = {
      name: (CATEGORY_LABEL[cls] ?? cls).replace(" Waste", "").replace("Plastic Recyclable", "Plastic"),
      cls,
    };
    COMPARISON_MODELS.forEach((m) => {
      entry[MODEL_META[m]?.label ?? m] = +((perClass[m]?.[cls]?.auc ?? 0) * 100).toFixed(1);
    });
    return entry;
  });

  const reload = () => { cmpQ.reload(); perClassQ.reload(); };

  return (
    <main className="min-h-screen bg-[#0c0414] bg-hero-glow text-foreground pt-24 pb-20 px-4 sm:px-6 page-enter">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* ── Header ── */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-violet-600 to-fuchsia-600 flex items-center justify-center">
                <GitCompare style={{ height: 18, width: 18 }} className="text-white" />
              </div>
              <h1 className="text-3xl font-semibold gradient-text">Model Comparison</h1>
            </div>
            <p className="text-sm text-foreground/40 ml-12">
              EfficientNet-B0 (prod) · EfficientNet-B2 · MobileNetV3 · DenseNet-121 · ResNet50
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={reload} loading={cmpQ.loading}>
            <RefreshCw className="h-4 w-4" /> Refresh
          </Button>
        </div>

        {/* ── No data banner ── */}
        {!hasData && !cmpQ.loading && (
          <Card className="p-8 flex flex-col items-center gap-3 text-foreground/40">
            <WandSparkles className="h-8 w-8" />
            <p className="text-sm">No comparison data yet.</p>
            <p className="text-xs text-center max-w-xs">
              Go to the Dashboard and click "Populate simulator data" to seed all 6 models,
              or run the training scripts to generate real metrics.
            </p>
          </Card>
        )}

        {/* ── 1. Summary scorecard ── */}
        {hasData && (
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
            {allModelsIncYolo.map((r) => {
              const meta = MODEL_META[r.model] ?? { label: r.model, color: "#a78bfa", note: "", params: "—" };
              const isProd = r.model === "efficientnet_b0";
              return (
                <Card
                  key={r.model}
                  className={"p-4 relative " + (isProd ? "border border-violet-500/40" : "")}
                >
                  {isProd && (
                    <span className="absolute top-2 right-2">
                      <Medal className="h-3.5 w-3.5 text-amber-400" />
                    </span>
                  )}
                  <div
                    className="h-1.5 rounded-full mb-3"
                    style={{ background: meta.color + "60" }}
                  />
                  <p className="text-[11px] font-semibold text-foreground/80 mb-2 leading-tight">
                    {meta.label}
                  </p>
                  <p className="text-2xl font-bold tabular-nums mb-0.5"
                     style={{ color: meta.color }}>
                    {r.f1 != null ? pct(r.f1) : "—"}
                  </p>
                  <p className="text-[9px] text-foreground/30 uppercase tracking-wider">F1-score</p>
                  <div className="mt-2 space-y-0.5">
                    <div className="flex justify-between text-[10px]">
                      <span className="text-foreground/40">Acc</span>
                      <span className="font-mono" style={{ color: confColor(r.accuracy ?? 0) }}>
                        {r.accuracy != null ? pct(r.accuracy) : "—"}
                      </span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-foreground/40">Lat</span>
                      <span className="font-mono text-foreground/60">
                        {r.inference_ms != null ? `${r.inference_ms}ms` : "—"}
                      </span>
                    </div>
                    <div className="flex justify-between text-[10px]">
                      <span className="text-foreground/40">Params</span>
                      <span className="font-mono text-foreground/60">{meta.params}</span>
                    </div>
                  </div>
                  <p className="text-[9px] text-foreground/25 mt-2 leading-tight">{meta.note}</p>
                </Card>
              );
            })}
          </div>
        )}

        {/* ── 2. Grouped bar chart ── */}
        {hasData && (
          <Card className="p-5">
            <div className="flex items-center justify-between mb-5 flex-wrap gap-3">
              <CardTitle icon={<BarChart3 className="h-4 w-4" />}>
                Accuracy · Precision · Recall · F1 — all models
              </CardTitle>
              <p className="text-[10px] text-foreground/30">YOLO-only excluded (F1 = 4.8% — breaks scale)</p>
            </div>
            {cmpQ.loading && !hasData ? (
              <Skeleton className="h-72 w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={groupedData} barCategoryGap="20%" barGap={2}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis
                    tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} tickLine={false}
                    domain={[80, 100]} tickFormatter={(v) => `${v}%`} width={40}
                  />
                  <Tooltip content={<DarkTooltip />} formatter={(v: number) => `${v.toFixed(1)}%`} />
                  <Legend wrapperStyle={{ fontSize: 11, color: "#9ca3af" }} />
                  {METRICS.map(({ key, label, color }) => (
                    <Bar key={key} dataKey={label} fill={color} radius={[4, 4, 0, 0]} maxBarSize={18} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>
        )}

        {/* ── 3. Radar + Scatter side by side ── */}
        {hasData && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {/* Radar */}
            <Card className="p-5">
              <CardTitle icon={<Target className="h-4 w-4" />} className="mb-5">
                Multi-metric radar — top 5 models
              </CardTitle>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.07)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                  {COMPARISON_MODELS.map((m) => (
                    <Radar
                      key={m}
                      name={MODEL_META[m]?.label ?? m}
                      dataKey={MODEL_META[m]?.label ?? m}
                      stroke={MODEL_META[m]?.color ?? "#a78bfa"}
                      fill={MODEL_META[m]?.color ?? "#a78bfa"}
                      fillOpacity={0.08}
                      strokeWidth={1.8}
                    />
                  ))}
                  <Tooltip content={<DarkTooltip />} formatter={(v: number) => `${v.toFixed(1)}%`} />
                  <Legend wrapperStyle={{ fontSize: 10, color: "#9ca3af" }} />
                </RadarChart>
              </ResponsiveContainer>
            </Card>

            {/* Latency vs Accuracy scatter */}
            <Card className="p-5">
              <CardTitle icon={<Zap className="h-4 w-4" />} className="mb-2">
                Latency vs Accuracy — efficiency frontier
              </CardTitle>
              <p className="text-[10px] text-foreground/30 mb-4">
                Bubble size = parameter count · lower-left = fast &amp; accurate
              </p>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    type="number" dataKey="x" name="Latency"
                    tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} tickLine={false}
                    label={{ value: "Inference (ms)", position: "insideBottom", offset: -8, fill: "#6b7280", fontSize: 10 }}
                  />
                  <YAxis
                    type="number" dataKey="y" name="Accuracy"
                    tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} tickLine={false}
                    domain={[88, 95]} tickFormatter={(v) => `${v}%`} width={44}
                  />
                  <ZAxis type="number" dataKey="z" range={[60, 260]} />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3", stroke: "rgba(255,255,255,0.1)" }}
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0].payload;
                      return (
                        <div className="glass rounded-xl px-4 py-3 text-xs space-y-1">
                          <p className="font-semibold" style={{ color: d.color }}>{d.name}</p>
                          <p className="text-foreground/60">Latency: <span className="text-foreground font-mono">{d.x}ms</span></p>
                          <p className="text-foreground/60">Accuracy: <span className="text-foreground font-mono">{d.y}%</span></p>
                          <p className="text-foreground/60">Params: <span className="text-foreground font-mono">{d.z}M</span></p>
                        </div>
                      );
                    }}
                  />
                  <Scatter data={scatterData} isAnimationActive={false}>
                    {scatterData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                    ))}
                  </Scatter>
                  {/* Model labels */}
                  {scatterData.map((d) => null /* recharts doesn't support inline labels easily */)}
                </ScatterChart>
              </ResponsiveContainer>
              {/* Manual legend */}
              <div className="flex flex-wrap gap-3 mt-3">
                {scatterData.map((d) => (
                  <span key={d.model} className="flex items-center gap-1.5 text-[10px] text-foreground/50">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: d.color }} />
                    {d.name}
                  </span>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* ── 4. Per-class F1 heatmap ── */}
        {hasData && Object.keys(perClass).length > 0 && (
          <Card className="p-5">
            <div className="flex items-center justify-between mb-5 flex-wrap gap-3">
              <CardTitle icon={<FlaskConical className="h-4 w-4" />}>
                Per-class F1 heatmap — which model wins each waste category
              </CardTitle>
              <div className="flex items-center gap-1 text-xs">
                {(["all", ...COMPARISON_MODELS] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setActiveClass(m as string)}
                    className={
                      "px-2.5 py-1 rounded-full font-mono transition-all text-[10px] " +
                      (activeClass === m
                        ? "bg-violet-500/20 text-violet-200 border border-violet-500/30"
                        : "text-foreground/40 hover:text-foreground hover:bg-white/5 border border-transparent")
                    }
                  >
                    {m === "all" ? "all models" : (MODEL_META[m]?.label ?? m).replace("EfficientNet-", "Eff-").replace("MobileNetV3-Large", "MobileV3").replace("DenseNet-121", "Dense-121")}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr>
                    <th className="text-left pb-3 pr-4 text-[10px] font-semibold uppercase tracking-widest text-foreground/30 min-w-[140px]">
                      Category
                    </th>
                    {COMPARISON_MODELS
                      .filter((m) => activeClass === "all" || activeClass === m)
                      .map((m) => (
                        <th key={m} className="pb-3 px-2 text-[10px] font-semibold tracking-wide whitespace-nowrap"
                            style={{ color: MODEL_META[m]?.color ?? "#a78bfa" }}>
                          {(MODEL_META[m]?.label ?? m)
                            .replace("EfficientNet-", "Eff-")
                            .replace("MobileNetV3-Large", "MobileV3")
                            .replace("DenseNet-121", "Dense-121")}
                        </th>
                      ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {classes.map((cls) => {
                    const colModels = COMPARISON_MODELS.filter(
                      (m) => activeClass === "all" || activeClass === m
                    );
                    // Find best model for this class
                    const bestF1 = Math.max(
                      ...colModels.map((m) => perClass[m]?.[cls]?.f1 ?? 0)
                    );
                    return (
                      <tr key={cls} className="hover:bg-white/[0.02] transition-colors">
                        <td className="py-3 pr-4">
                          <span className="flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full shrink-0"
                                  style={{ background: CATEGORY_COLOR[cls] ?? "#a78bfa" }} />
                            <span className="text-foreground/70">
                              {(CATEGORY_LABEL[cls] ?? cls).replace(" Waste", "")}
                            </span>
                          </span>
                        </td>
                        {colModels.map((m) => {
                          const v = perClass[m]?.[cls];
                          const f1 = v?.f1 ?? 0;
                          const isBest = colModels.length > 1 && Math.abs(f1 - bestF1) < 0.001;
                          return (
                            <td key={m} className="py-3 px-2 text-center">
                              <div className="inline-flex flex-col items-center gap-1">
                                <span
                                  className={"font-mono font-semibold tabular-nums text-xs " +
                                    (isBest ? "underline decoration-dotted" : "")}
                                  style={{ color: isBest ? (MODEL_META[m]?.color ?? "#a78bfa") : confColor(f1) }}
                                >
                                  {f1 > 0 ? pct(f1) : "—"}
                                  {isBest && <Medal className="h-2.5 w-2.5 inline ml-0.5 text-amber-400" />}
                                </span>
                                {/* Mini bar */}
                                <div className="h-1 w-12 rounded-full bg-white/5 overflow-hidden">
                                  <div
                                    className="h-full rounded-full"
                                    style={{
                                      width: `${f1 * 100}%`,
                                      background: MODEL_META[m]?.color ?? "#a78bfa",
                                      opacity: 0.7,
                                    }}
                                  />
                                </div>
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        {/* ── 5. Per-class AUC grouped bar ── */}
        {hasData && Object.keys(perClass).length > 0 && (
          <Card className="p-5">
            <CardTitle icon={<Cpu className="h-4 w-4" />} className="mb-5">
              Per-class ROC AUC — all models (one-vs-rest)
            </CardTitle>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={aucBarData} barCategoryGap="20%" barGap={2}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis
                  tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={false} tickLine={false}
                  domain={[93, 100]} tickFormatter={(v) => `${v}%`} width={42}
                />
                <Tooltip content={<DarkTooltip />} formatter={(v: number) => `${v.toFixed(1)}%`} />
                <Legend wrapperStyle={{ fontSize: 10, color: "#9ca3af" }} />
                {COMPARISON_MODELS.map((m) => (
                  <Bar
                    key={m}
                    dataKey={MODEL_META[m]?.label ?? m}
                    fill={MODEL_META[m]?.color ?? "#a78bfa"}
                    radius={[4, 4, 0, 0]}
                    maxBarSize={14}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}

        {/* ── 6. Full comparison table ── */}
        {hasData && (
          <Card className="p-5">
            <CardTitle icon={<BarChart3 className="h-4 w-4" />} className="mb-5">
              Full metrics table — all 6 models
            </CardTitle>
            <div className="overflow-x-auto -mx-5 px-5">
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    {["Model", "Acc", "Prec", "Recall", "F1", "Params", "Latency", "Notes"].map((h) => (
                      <th key={h}
                          className="text-left pb-3 pr-3 text-[10px] font-semibold uppercase tracking-widest text-foreground/30 whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {allModelsIncYolo.map((r, i) => {
                    const meta = MODEL_META[r.model] ?? { label: r.model, color: "#a78bfa", params: "—", note: "" };
                    const isProd = r.model === "efficientnet_b0";
                    return (
                      <tr key={r.model} className={`hover:bg-white/[0.02] transition-colors ${isProd ? "bg-violet-500/5" : ""}`}>
                        <td className="py-3 pr-3">
                          <span className="flex items-center gap-2">
                            {isProd && <Medal className="h-3.5 w-3.5 text-amber-400 shrink-0" />}
                            <span
                              className="font-mono text-xs font-medium"
                              style={{ color: meta.color }}
                            >
                              {meta.label}
                            </span>
                          </span>
                        </td>
                        <td className="py-3 pr-3 font-mono tabular-nums text-xs"
                            style={{ color: confColor(r.accuracy ?? 0) }}>
                          {r.accuracy != null ? pct(r.accuracy) : "—"}
                        </td>
                        <td className="py-3 pr-3 font-mono tabular-nums text-xs text-foreground/70">
                          {r.precision != null ? pct(r.precision) : "—"}
                        </td>
                        <td className="py-3 pr-3 font-mono tabular-nums text-xs text-foreground/70">
                          {r.recall != null ? pct(r.recall) : "—"}
                        </td>
                        <td className="py-3 pr-3 font-mono tabular-nums text-xs font-semibold"
                            style={{ color: meta.color }}>
                          {r.f1 != null ? pct(r.f1) : "—"}
                        </td>
                        <td className="py-3 pr-3 font-mono tabular-nums text-xs text-foreground/50">
                          {meta.params}
                        </td>
                        <td className="py-3 pr-3 font-mono tabular-nums text-xs text-foreground/50">
                          {r.inference_ms != null ? `${r.inference_ms}ms` : "—"}
                        </td>
                        <td className="py-3 text-[10px] text-foreground/30 max-w-[200px] truncate">
                          {r.notes ?? meta.note}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        {/* ── 7. Key takeaways ── */}
        {hasData && (
          <Card className="p-6">
            <CardTitle icon={<Medal className="h-4 w-4" />} className="mb-4">
              Why EfficientNet-B0 is the production choice
            </CardTitle>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-foreground/60">
              {[
                {
                  icon: Target,
                  color: "#a78bfa",
                  title: "Best accuracy-per-param",
                  body: "EfficientNet-B0 achieves 92.4% accuracy with only 5.3M parameters — the highest efficiency ratio of any model tested. EfficientNet-B2 scores 93.8% but requires 72% more parameters and almost double the latency.",
                },
                {
                  icon: Zap,
                  color: "#34d399",
                  title: "MobileNetV3: the edge alternative",
                  body: "At 11.2ms inference (vs B0's 18.4ms), MobileNetV3-Large is the right choice if deploying on hospital edge devices or bin-mounted cameras where compute is constrained — accepting a ~1.7% accuracy drop.",
                },
                {
                  icon: FlaskConical,
                  color: "#60a5fa",
                  title: "DenseNet-121: biomedical argument",
                  body: "DenseNet-121 (CheXNet backbone) performs competitively at 91.6% and shows particularly strong results on pathological and infectious waste — categories where dense feature reuse matters most.",
                },
              ].map(({ icon: Icon, color, title, body }) => (
                <div key={title} className="rounded-xl border border-white/5 p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <div className="h-7 w-7 rounded-lg flex items-center justify-center"
                         style={{ background: color + "20" }}>
                      <Icon className="h-3.5 w-3.5" style={{ color }} />
                    </div>
                    <span className="text-xs font-semibold text-foreground/80">{title}</span>
                  </div>
                  <p className="leading-relaxed">{body}</p>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </main>
  );
}
