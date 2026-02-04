#!/usr/bin/env python3
"""
Container detection model training script (YOLOv8).
Dataset: datasets/Final_Container_Project.v7i.yolov8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_data = root / "datasets" / "Final_Container_Project.v7i.yolov8" / "data.yaml"

    parser = argparse.ArgumentParser(description="Train YOLOv8 container detection model.")
    parser.add_argument("--data", type=Path, default=default_data, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device: 0,1,... or 'cpu'")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--name", type=str, default="container_detector", help="Run name")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="Project dir")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--cache", type=str, default="ram", help="Cache: ram, disk, or False")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("CONTAINER YOLO MODEL TRAINING")
    print("=" * 70)

    if not args.data.exists():
        print(f"❌ Dataset not found: {args.data}")
        return

    print(f"\n✓ Dataset: {args.data}")
    print(f"✓ Base model: {args.model}")
    print("\n⚙️  Training params:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch}")
    print(f"  - Image Size: {args.imgsz}")
    print(f"  - Device: {args.device}")
    print(f"  - Patience: {args.patience}")
    print(f"  - Workers: {args.workers}")
    print(f"  - Cache: {args.cache}")

    model = YOLO(args.model)
    print("\n⏳ Training starts...")

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        name=args.name,
        project=str(args.project),
        save=True,
        verbose=True,
        plots=True,
        workers=args.workers,
        cache=args.cache,
    )

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED")
    print("=" * 70)

    best_model = Path(args.project) / args.name / "weights" / "best.pt"
    if best_model.exists():
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        target = models_dir / "container_best.pt"
        target.write_bytes(best_model.read_bytes())
        print(f"\n✓ Best model copied: {target}")

    print("\n📊 Results:")
    print(f"  - mAP50: {results.results_dict.get('metrics/mAP50', 'N/A')}")
    print(f"  - Precision: {results.results_dict.get('metrics/precision', 'N/A')}")
    print(f"  - Recall: {results.results_dict.get('metrics/recall', 'N/A')}")


if __name__ == "__main__":
    main()
