#!/usr/bin/env python3
"""
Konteyner ISO karakter tespiti modeli (YOLOv8) eğitimi.
Dataset: datasets/konteyner_karakter_okuma (nc=1, names=['char'])
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _normalize_cache(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no", "off", "none"}:
            return False
        if lowered in {"true", "1", "yes", "on"}:
            return True
    return value


def _read_dataset_meta(data_path: Path):
    nc = None
    names = None
    try:
        for line in data_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("nc:"):
                try:
                    nc = int(stripped.split(":", 1)[1].strip())
                except Exception:
                    pass
            if stripped.startswith("names:"):
                names = stripped.split(":", 1)[1].strip()
    except Exception:
        pass
    return nc, names


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_data = root / "datasets" / "konteyner_karakter_okuma" / "data.yaml"
    default_copy = root / "models" / "konteyner_karakter_best.pt"

    parser = argparse.ArgumentParser(description="Train YOLOv8 container ISO character detector.")
    parser.add_argument("--data", type=Path, default=default_data, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device: 0,1,... or 'cpu'")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--name", type=str, default="konteyner_char_detector", help="Run name")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Project dir")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--cache", type=str, default="False", help="Cache: ram, disk, True/False")
    parser.add_argument("--copy-to", type=Path, default=default_copy, help="Copy best.pt to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_value = _normalize_cache(args.cache)

    print("=" * 70)
    print("KONTEYNER ISO KARAKTER MODELİ EĞİTİMİ")
    print("=" * 70)

    if not args.data.exists():
        print(f"❌ Dataset not found: {args.data}")
        return

    print(f"\n✓ Dataset: {args.data}")
    nc, names = _read_dataset_meta(args.data)
    if nc is not None:
        print(f"  - nc: {nc}")
    if names:
        print(f"  - names: {names}")
    print(f"✓ Base model: {args.model}")
    print("\n⚙️  Training params:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch}")
    print(f"  - Image Size: {args.imgsz}")
    print(f"  - Device: {args.device}")
    print(f"  - Patience: {args.patience}")
    print(f"  - Workers: {args.workers}")
    print(f"  - Cache: {cache_value}")
    print(f"  - Run name: {args.name}")

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
        cache=cache_value,
    )

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED")
    print("=" * 70)

    best_model = Path(args.project) / "detect" / args.name / "weights" / "best.pt"
    if best_model.exists() and args.copy_to:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        args.copy_to.write_bytes(best_model.read_bytes())
        print(f"\n✓ Best model copied: {args.copy_to}")
    else:
        print(f"\n⚠ Best model not found at: {best_model}")

    print("\n📊 Results:")
    print(f"  - mAP50: {results.results_dict.get('metrics/mAP50', 'N/A')}")
    print(f"  - Precision: {results.results_dict.get('metrics/precision', 'N/A')}")
    print(f"  - Recall: {results.results_dict.get('metrics/recall', 'N/A')}")


if __name__ == "__main__":
    main()
