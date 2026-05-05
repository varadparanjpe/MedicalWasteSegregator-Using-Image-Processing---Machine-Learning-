const API_URL = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";

/* ─────────────────────────────────────────── Types ─────────────────────── */
export interface BBox { x1: number; y1: number; x2: number; y2: number; }

export interface Detection {
  id: number;
  bbox: BBox;
  detection_confidence: number;
  category: string;
  classification_confidence: number;
  bin: string;
  alerts: string[];
  cropped_image_b64?: string;
  gradcam_b64?: string;
}

export interface PredictResponse {
  success: boolean;
  image_width: number;
  image_height: number;
  num_detections: number;
  detections: Detection[];
  processing_ms: number;
  annotated_image_b64?: string;
  used_ensemble?: boolean;
}

export interface HistoryItem {
  id: number; ts: string; category: string; bin: string;
  confidence: number; alerts: string[];
}

export interface DashMetrics {
  total_detections: number;
  counts_by_category: Record<string, number>;
  counts_by_bin: Record<string, number>;
  avg_confidence: number;
  hazard_rate: number;
  low_confidence_count: number;
  timeseries: { hour: string; count: number }[];
}

export interface ModelMetrics {
  accuracy: number; precision: number; recall: number; f1: number;
  per_class: Record<string, { precision: number; recall: number; f1: number }>;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
  model: string;
  ts: string | null;
}

export interface ComparisonRow {
  model: string;
  accuracy?: number; precision?: number; recall?: number; f1?: number;
  params_m?: number; inference_ms?: number; notes?: string; ts?: string;
}

export interface PerfStats {
  p50: number; p90: number; p99: number;
  mean: number; max: number;
  throughput_fps: number;
  samples: number;
}

export interface ConfidenceHist {
  bins: number[];
  counts: number[];
  correct: number[] | null;
}

export interface FailureCase {
  id: number;
  image_path: string;
  true_label: string;
  pred_label: string;
  confidence: number;
  reason: string | null;
  image_b64: string | null;
}

export interface AblationRow {
  setting: string;
  accuracy: number;
  f1: number;
  notes?: string;
  ts?: string;
}

export interface TrainingEpoch {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
}

export interface RocAucResponse {
  per_class: Record<string, number>;
}

export interface PerClassEntry {
  precision: number;
  recall: number;
  f1: number;
  auc: number;
}

/** { model_name: { class_name: PerClassEntry } } */
export type AllModelPerClass = Record<string, Record<string, PerClassEntry>>;

/* ─────────────────────────────────────────── Helpers ───────────────────── */
async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const res = await fetch(input, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

/* ─────────────────────────────────────────── Endpoints ─────────────────── */
export async function predictFile(
  file: File, opts: { explain?: boolean; use_ensemble?: boolean } = {}
): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const params = new URLSearchParams();
  if (opts.explain)      params.set("explain", "true");
  if (opts.use_ensemble) params.set("use_ensemble", "true");
  const qs = params.toString() ? `?${params.toString()}` : "";
  return request<PredictResponse>(`${API_URL}/api/predict${qs}`, {
    method: "POST",
    body: fd,
  });
}

export async function predictBase64(
  image_b64: string,
  opts: { explain?: boolean; use_ensemble?: boolean; include_annotated?: boolean } = {}
): Promise<PredictResponse> {
  return request<PredictResponse>(`${API_URL}/api/predict/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_b64,
      include_annotated: opts.include_annotated ?? true,
      use_ensemble: !!opts.use_ensemble,
      explain: !!opts.explain,
    }),
  });
}

export const getMetrics         = () => request<DashMetrics>(`${API_URL}/api/metrics`);
export const getModelMet        = () => request<ModelMetrics>(`${API_URL}/api/model-metrics`);
export const getHistory         = (limit = 50) => request<HistoryItem[]>(`${API_URL}/api/history?limit=${limit}`);
export const getHealth          = () => request<{ status: string; pipeline: boolean }>(`${API_URL}/api/health`);
export const getConfMatrix      = (model = "efficientnet_b0") => request<ConfusionMatrix>(`${API_URL}/api/confusion-matrix?model=${model}`);
export const getModelComparison = () => request<ComparisonRow[]>(`${API_URL}/api/model-comparison`);
export const getPerf            = () => request<PerfStats>(`${API_URL}/api/perf`);
export const getConfHist        = () => request<ConfidenceHist>(`${API_URL}/api/confidence-hist`);
export const getFailures        = (limit = 24) => request<FailureCase[]>(`${API_URL}/api/failures?limit=${limit}`);
export const getAblation         = () => request<AblationRow[]>(`${API_URL}/api/ablation`);
export const getTrainingHistory    = () => request<TrainingEpoch[]>(`${API_URL}/api/training-history`);
export const getRocAuc             = () => request<RocAucResponse>(`${API_URL}/api/roc-auc`);
export const getAllModelPerClass    = () => request<AllModelPerClass>(`${API_URL}/api/model-comparison/per-class`);

/* ── Simulator (capstone demo data) ─────────────────────────────────── */
export const simulateData =
  (n = 250) => request<Record<string, unknown>>(`${API_URL}/api/simulate?n_detections=${n}`, { method: "POST" });

export const clearSimulatedData =
  () => request<Record<string, unknown>>(`${API_URL}/api/simulate/clear`, { method: "POST" });
