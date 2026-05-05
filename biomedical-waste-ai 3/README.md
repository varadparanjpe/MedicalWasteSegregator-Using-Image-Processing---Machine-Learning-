# Biomedical Waste Classification and Real-Time Segregation System

Production-grade end-to-end AI system for biomedical waste detection, classification, and rule-based bin segregation aligned with WHO guidelines. **Zero LLMs** — every decision is deterministic.

---

## 🧠 Architecture

```
Camera / Image Input
        ↓
YOLOv8 (Ultralytics)  →  Object Detection (bounding boxes only)
        ↓
Crop each detected object
        ↓
Resize to 224 × 224
        ↓
EfficientNet-B0 (PyTorch)  →  Final Classification
        ↓
Confidence Check  (threshold = 0.80)
        ↓
Rule-Based Segregation  (BIN_MAP — pure dict lookup)
        ↓
Alert System   (HAZARDOUS / LOW_CONFIDENCE)
        ↓
Grad-CAM overlay (optional, explainability)
        ↓
Response + SQLite persistence
```

**Hard constraints honored:**
- No LLMs. No OpenAI / Anthropic / HuggingFace LLM calls anywhere.
- YOLOv8 is used **only** for detection. Never for classification.
- EfficientNet-B0 is used **only after cropping**.
- Segregation is **purely rule-based** (`BIN_MAP`).

---

## ✅ Capstone checklist coverage

| Area | Feature | Location |
|------|---------|----------|
| **Core** | YOLOv8 + EfficientNet-B0 pipeline | `backend/utils/pipeline.py` |
| Core | Rule-based `BIN_MAP` segregation | `backend/config.py`, `backend/utils/rules.py` |
| Core | FastAPI + SQLite | `backend/main.py`, `backend/database/db.py` |
| Core | Dark UI dashboard | `frontend/src/` |
| **Evaluation** | Accuracy / Precision / Recall / F1 | `ml/training/train_efficientnet.py` |
| Evaluation | Per-class metrics | Auto-saved by training → `/api/model-metrics` |
| Evaluation | Confusion matrix + heatmap PNG | `ml/training/confusion_matrix.py` |
| Evaluation | Model comparison (EfficientNet vs ResNet50 vs YOLO-only) | `ml/training/compare_resnet50.py`, `yolo_only_baseline.py` |
| **Dashboard** | Bar chart, line, pie | `frontend/src/pages/Dashboard.tsx` |
| Dashboard | Confusion matrix heatmap | `frontend/src/components/ui/heatmap.tsx` |
| Dashboard | Hazard rate, low-conf alerts, confidence histogram | ✓ |
| Dashboard | Latency (P50/P90/P99) + throughput (fps) | `backend/utils/perf.py` → `/api/perf` |
| **Explainability** | Grad-CAM on EfficientNet-B0 | `backend/models/gradcam.py`; toggle on Prediction page |
| **Realtime** | Live webcam @ 1–2 fps, frame-skip, async inference | `frontend/src/pages/Realtime.tsx` |
| **Data / training** | Class-weighted loss | `ml/training/class_weights.py`, `train_efficientnet.py --class-weights` |
| Data / training | Ablation (aug vs no-aug) | `ml/training/ablation.py` |
| Data / training | Stratified split + aug | handled by `MLA_PreProcessing.py` |
| **Error handling** | No detection / low-conf / invalid image | pipeline returns `detections: []`, API `400`s invalid |
| **Advanced ML** | Ensemble (EfficientNet + ResNet50 soft-voting) | `backend/models/ensemble.py`, toggle on Prediction page |
| Advanced ML | YOLO-only vs two-stage study | `ml/training/yolo_only_baseline.py` |
| Advanced ML | Confidence calibration histogram | `train_efficientnet.py` → `/api/confidence-hist` |
| **Failure analysis** | Misclassified grid w/ reasons | `ml/training/failure_analysis.py` → `/api/failures` |
| **Deployment** | Dockerfile × 2 + docker-compose | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |
| Deployment | Swagger / OpenAPI docs | `http://localhost:8000/docs` |

---

## 📁 Project structure

```
biomedical-waste-ai/
├── backend/
│   ├── main.py                       # FastAPI app (all endpoints)
│   ├── config.py                     # Paths, thresholds, BIN_MAP, categories
│   ├── schemas.py                    # Pydantic models for every endpoint
│   ├── Dockerfile
│   ├── models/
│   │   ├── detection_model.py        # YOLOv8 wrapper
│   │   ├── classification_model.py   # EfficientNet-B0 wrapper
│   │   ├── ensemble.py               # EfficientNet + ResNet50 soft-voting
│   │   └── gradcam.py                # Grad-CAM + heatmap overlay
│   ├── database/
│   │   └── db.py                     # SQLite — 7 tables, all dashboard data
│   ├── utils/
│   │   ├── pipeline.py               # End-to-end inference
│   │   ├── rules.py                  # BIN_MAP + alerts
│   │   ├── image_ops.py              # Crop / b64 / render
│   │   └── perf.py                   # Rolling latency + throughput tracker
│   ├── weights/                      # yolov8n.pt, efficientnet_b0.pth, resnet50.pth
│   └── requirements.txt
├── ml/
│   └── training/
│       ├── common.py                 # Shared loaders / transforms
│       ├── train_efficientnet.py     # Primary classifier (w/ class-weights flag)
│       ├── train_yolo.py             # YOLO fine-tune
│       ├── compare_resnet50.py       # ResNet50 baseline
│       ├── yolo_only_baseline.py     # Proves two-stage > one-stage
│       ├── confusion_matrix.py       # CM + metrics + PNG → DB
│       ├── ablation.py               # aug vs no-aug study
│       ├── class_weights.py          # Inverse-frequency weights
│       ├── failure_analysis.py       # Misclassified viewer
│       └── evaluate.py               # Standalone eval
├── frontend/
│   ├── src/
│   │   ├── components/ui/
│   │   │   ├── hero.tsx              # Dark glassy Hero
│   │   │   ├── navbar.tsx            # Floating pill w/ API health dot
│   │   │   ├── heatmap.tsx           # Confusion-matrix heatmap
│   │   │   ├── confidence-ring.tsx   # ConfBar + Ring
│   │   │   ├── dropzone.tsx
│   │   │   ├── button.tsx
│   │   │   └── card.tsx
│   │   ├── pages/
│   │   │   ├── Landing.tsx
│   │   │   ├── Prediction.tsx        # Grad-CAM + Ensemble toggles
│   │   │   ├── Realtime.tsx          # Live webcam @ 1–2 fps
│   │   │   └── Dashboard.tsx         # Full analytics
│   │   ├── hooks/useApi.ts           # useAsync + usePoll
│   │   ├── lib/{api.ts, utils.ts}
│   │   ├── App.tsx, main.tsx, index.css
│   │   └── vite-env.d.ts
│   ├── Dockerfile + nginx.conf
│   ├── package.json, vite.config.ts, tailwind.config.js
│   └── tsconfig.json
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick start

### Option A — Docker (single command)

```bash
docker compose up --build
# Frontend:  http://localhost:5173
# Backend :  http://localhost:8000/docs   (Swagger)
```

### Option B — Native

```bash
# 1. Preprocess the dataset (uses your existing script)
python MLA_PreProcessing.py
# → /Users/varadparanjpe/Downloads/processed_dataset/{train,val,test}/<category>/

# 2. Train EfficientNet-B0 (populates model metrics, CM, latency, confidence hist)
cd biomedical-waste-ai
python ml/training/train_efficientnet.py \
    --data-root /Users/varadparanjpe/Downloads/processed_dataset \
    --out backend/weights/efficientnet_b0.pth \
    --epochs 30 --batch-size 32 --class-weights

# 3. (Optional) ResNet50 baseline + YOLO-only study
python ml/training/compare_resnet50.py \
    --data-root /Users/varadparanjpe/Downloads/processed_dataset \
    --out backend/weights/resnet50.pth --epochs 20

python ml/training/yolo_only_baseline.py \
    --data-root /Users/varadparanjpe/Downloads/processed_dataset \
    --yolo backend/weights/yolov8n.pt

# 4. (Optional) Ablation + failure analysis
python ml/training/ablation.py \
    --data-root /Users/varadparanjpe/Downloads/processed_dataset --epochs 8

python ml/training/failure_analysis.py \
    --data-root /Users/varadparanjpe/Downloads/processed_dataset \
    --weights   backend/weights/efficientnet_b0.pth --top 24

# 5. Run the API
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 6. Run the frontend
cd ../frontend
npm install
npm run dev      # http://localhost:5173
```

---

## 🧪 API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health`          | GET  | Liveness                                                           |
| `/api/predict`         | POST | Multipart image. Query: `?explain=true&use_ensemble=true`           |
| `/api/predict/base64`  | POST | JSON body — used by webcam & Grad-CAM re-run                       |
| `/api/history`         | GET  | Last N detections                                                   |
| `/api/metrics`         | GET  | Aggregated telemetry (category/bin/hazard/low-conf/timeseries)      |
| `/api/model-metrics`   | GET  | Accuracy / Precision / Recall / F1 + per-class                      |
| `/api/confusion-matrix`| GET  | `?model=efficientnet_b0\|resnet50` — labels + 6×6 matrix            |
| `/api/model-comparison`| GET  | EfficientNet vs ResNet50 vs YOLO-only table                         |
| `/api/confidence-hist` | GET  | Binned confidence distribution (calibration curve)                  |
| `/api/failures`        | GET  | Misclassified examples with thumbnails + reasons                    |
| `/api/ablation`        | GET  | aug vs no-aug rows                                                  |
| `/api/perf`            | GET  | Mean / P50 / P90 / P99 latency + throughput (fps)                   |

Full OpenAPI / Swagger UI → `http://localhost:8000/docs`.

---

## ♻️ Segregation rules (WHO-aligned)

| Category              | Bin        | Alert       |
|-----------------------|------------|-------------|
| sharps_waste          | White ⬜   | HAZARDOUS   |
| infectious_waste      | Yellow 🟡  | HAZARDOUS   |
| pathological_waste    | Yellow 🟡  | —           |
| plastic_recyclable    | Red 🔴     | —           |
| pharmaceutical_waste  | Blue 🔵    | —           |
| general_waste         | Black ⬛   | —           |

Confidence < 0.80 ⇒ `LOW_CONFIDENCE` alert attached to that detection.

---

## 🖥️ Frontend pages

1. **Landing** (`/`) — Hero component, drag-drop, file picker, webcam capture, feature cards.
2. **Prediction** (`/predict`) — Annotated image with canvas bbox overlay, pipeline step indicator, per-detection cards with ConfBar, **Grad-CAM toggle** (shows heat-map overlay on each crop), **Ensemble toggle** (soft-vote with ResNet50). Fullscreen toggle. Expandable technical-details grid.
3. **Realtime** (`/realtime`) — Live webcam auto-capture at 0.5 / 1 / 2 fps. Async inference with in-flight guard; dropped-frame counter. Live bbox overlay. Live latency panel (P50/P90/P99 + fps throughput from `/api/perf`).
4. **Dashboard** (`/dashboard`) — Every dashboard feature from the capstone checklist:
   - KPI cards (total, avg conf, hazard rate, low-conf)
   - Latency row (mean / P50 / P90 / P99 / throughput)
   - Bar chart: items per category · pie: bin distribution
   - Line: detections over time (7 days)
   - Model-metrics gauges
   - **Confusion-matrix heatmap**
   - Per-class radar (Precision / Recall / F1)
   - **Confidence histogram** (all vs correct — for calibration)
   - **Model comparison table** (EfficientNet vs ResNet50 vs YOLO-only) with medal icon for best
   - **Ablation study table** (aug vs no-aug)
   - **Failure-case grid** with true/pred labels + auto-detected reason
   - Recent detections live table

---

## 📊 What each ML script writes to the database

| Script | Populates |
|--------|-----------|
| `train_efficientnet.py` | `model_metrics`, `confusion_matrix`, `model_comparison` (EfficientNet row), `confidence_hist` |
| `compare_resnet50.py`   | `model_comparison` (ResNet50 row), `confusion_matrix` (resnet50) |
| `yolo_only_baseline.py` | `model_comparison` (YOLO-only row) |
| `confusion_matrix.py`   | `confusion_matrix`, `model_metrics`, `model_comparison`, `confidence_hist` |
| `ablation.py`           | `ablation_results` |
| `failure_analysis.py`   | `failure_cases` |

Run them in any order — the dashboard refreshes automatically (5 s poll).

---

## 🔐 No-LLM guarantee

Every classifier output is either:
- `argmax(softmax(conv_logits))` for EfficientNet-B0 / ResNet50, or
- `(p_effnet + p_resnet).argmax()` for the ensemble.

Every bin decision is `BIN_MAP[category]` — a Python dict.
Every alert is `confidence < 0.80` or `category in {sharps_waste, infectious_waste}`.
No prompt. No token. No sampling temperature. Just CNNs and `if` statements.
