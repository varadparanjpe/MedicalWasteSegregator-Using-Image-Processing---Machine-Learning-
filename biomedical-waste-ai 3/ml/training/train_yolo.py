"""
Fine-tune YOLOv8 on biomedical waste DETECTION data (boxes only).

Expected data.yaml format (Ultralytics):
    path: /path/to/dataset
    train: images/train
    val: images/val
    names: [waste]          # one generic class is enough —
                            # YOLO labels are discarded downstream.

Usage:
    python train_yolo.py --data data.yaml --epochs 80 --imgsz 640 \
        --out ../../backend/weights/yolov8n.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to Ultralytics data.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/detect",
        name="biomed_waste",
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(best.read_bytes())
    print(f"Best weights copied to {args.out}")


if __name__ == "__main__":
    main()
