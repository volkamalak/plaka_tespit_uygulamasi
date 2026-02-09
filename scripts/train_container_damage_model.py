#!/usr/bin/env python3
"""
Container damage detector training script (YOLO26/YOLOv8 family).

Example:
  .venv/bin/python scripts/train_container_damage_model.py \
      --data datasets/container_damage/data.yaml \
      --model yolo26n.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _resolve_default_data(root: Path) -> Path:
    """Pick the best available dataset yaml in this repo."""
    candidates = [
        root / "datasets" / "Container damage.v1i.yolov8" / "data.yaml",
        root / "datasets" / "container_damage" / "data.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
    default_data = _resolve_default_data(root)
    default_copy = root / "models" / "konteyner_hasar.pt"

    parser = argparse.ArgumentParser(description="Train YOLO container damage detector.")
    parser.add_argument("--data", type=Path, default=default_data, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device: 0,1,... or 'cpu'")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--name", type=str, default="container_damage_detector", help="Run name")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Project dir")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--cache", type=str, default="False", help="Cache: ram, disk, True/False")
    parser.add_argument("--copy-to", type=Path, default=default_copy, help="Copy best.pt to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data = args.data.expanduser().resolve()
    args.project = args.project.expanduser()
    if not args.project.is_absolute():
        args.project = (Path(__file__).resolve().parents[1] / args.project).resolve()
    if args.copy_to:
        args.copy_to = args.copy_to.expanduser().resolve()

    cache_value = _normalize_cache(args.cache)

    print("=" * 70)
    print("CONTAINER DAMAGE MODEL TRAINING")
    print("=" * 70)

    if not args.data.exists():
        print(f"Dataset not found: {args.data}")
        print("Tip: create data.yaml with damage classes, e.g. dent/scratch/crack/rust/deformation/breakage")
        return

    print(f"\nDataset: {args.data}")
    nc, names = _read_dataset_meta(args.data)
    if nc is not None:
        print(f"  - nc: {nc}")
    if names:
        print(f"  - names: {names}")
    print(f"Base model: {args.model}")
    print("\nTraining params:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch}")
    print(f"  - Image Size: {args.imgsz}")
    print(f"  - Device: {args.device}")
    print(f"  - Patience: {args.patience}")
    print(f"  - Workers: {args.workers}")
    print(f"  - Cache: {cache_value}")
    print(f"  - Run name: {args.name}")

    model = YOLO(args.model)
    print("\nTraining starts...")

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
    print("TRAINING COMPLETED")
    print("=" * 70)

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        best_model = Path(save_dir) / "weights" / "best.pt"
    else:
        best_model = Path(args.project) / "detect" / args.name / "weights" / "best.pt"
    if best_model.exists() and args.copy_to:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        args.copy_to.write_bytes(best_model.read_bytes())
        print(f"\nBest model copied to: {args.copy_to}")
    else:
        print(f"\nBest model not found at: {best_model}")

    print("\nResults:")
    print(f"  - mAP50: {results.results_dict.get('metrics/mAP50', 'N/A')}")
    print(f"  - Precision: {results.results_dict.get('metrics/precision', 'N/A')}")
    print(f"  - Recall: {results.results_dict.get('metrics/recall', 'N/A')}")


if __name__ == "__main__":
    main()
