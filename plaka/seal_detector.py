"""
Container seal presence detector.
Supports YOLO detection or classification models.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


class ContainerSealDetector:
    """Detects whether a container seal is present."""

    def __init__(
        self,
        model_path=None,
        present_labels=None,
        absent_labels=None,
    ):
        project_root = Path(__file__).resolve().parents[1]
        default_path = project_root / "models" / "muhur_bulma.pt"
        fallback_path = project_root / "models" / "container_seal.pt"
        self.model_path = Path(model_path) if model_path else default_path
        self.fallback_path = fallback_path
        self.model = None
        self.present_labels = {s.lower() for s in (present_labels or ["seal", "present", "var"])}
        self.absent_labels = {s.lower() for s in (absent_labels or ["no_seal", "absent", "yok"])}
        self.load_model()

    def load_model(self):
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(str(self.model_path))
                print(f"Container seal model loaded: {self.model_path}")
            elif self.fallback_path.exists():
                self.model_path = self.fallback_path
                self.model = YOLO(str(self.model_path))
                print(f"Container seal model loaded: {self.model_path}")
            else:
                print(f"WARN: Container seal model not found: {self.model_path}")
                self.model = None
        except Exception as exc:
            print(f"Container seal model load error: {exc}")
            self.model = None

    def _class_name(self, class_id):
        if not self.model:
            return ""
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return str(names.get(class_id, ""))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return ""

    def detect_seal(self, image, conf_threshold=0.25):
        result = {
            "success": False,
            "present": None,
            "confidence": 0.0,
            "class_name": "",
            "boxes": [],
            "error": None,
        }

        if self.model is None:
            result["error"] = "Seal model not loaded"
            return result

        if image is None or getattr(image, "size", 0) == 0:
            result["error"] = "Empty image"
            return result

        try:
            results = self.model(image, conf=conf_threshold, verbose=False)
        except Exception as exc:
            result["error"] = f"Seal detection error: {exc}"
            return result

        if not results:
            result["error"] = "No result"
            return result

        r0 = results[0]

        # Classification path
        probs = getattr(r0, "probs", None)
        if probs is not None:
            try:
                top_idx = int(probs.top1)
                top_conf = float(probs.top1conf)
            except Exception:
                top_idx = None
                top_conf = 0.0

            if top_idx is None:
                result["error"] = "Classification output not available"
                return result

            class_name = self._class_name(top_idx)
            class_key = class_name.lower()

            if class_key in self.present_labels:
                present = True
            elif class_key in self.absent_labels:
                present = False
            else:
                present = True

            result.update({
                "success": True,
                "present": present,
                "confidence": top_conf,
                "class_name": class_name,
                "boxes": [],
            })
            return result

        # Detection path
        present_best = (None, 0.0, "")
        absent_best = (None, 0.0, "")
        boxes = []

        for box in getattr(r0, "boxes", []):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self._class_name(cls)
            class_key = class_name.lower()

            boxes.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": cls,
                "class_name": class_name,
            })

            if class_key in self.present_labels:
                if conf > present_best[1]:
                    present_best = ((x1, y1, x2, y2), conf, class_name)
            elif class_key in self.absent_labels:
                if conf > absent_best[1]:
                    absent_best = ((x1, y1, x2, y2), conf, class_name)

        if present_best[0] is not None:
            result.update({
                "success": True,
                "present": True,
                "confidence": present_best[1],
                "class_name": present_best[2],
                "boxes": boxes,
            })
            return result

        if absent_best[0] is not None:
            result.update({
                "success": True,
                "present": False,
                "confidence": absent_best[1],
                "class_name": absent_best[2],
                "boxes": boxes,
            })
            return result

        if boxes:
            # If labels are unknown, fall back to presence by any detection.
            best_box = max(boxes, key=lambda b: b["confidence"])
            result.update({
                "success": True,
                "present": True,
                "confidence": best_box["confidence"],
                "class_name": best_box["class_name"],
                "boxes": boxes,
            })
            return result

        result["error"] = "No seal detected"
        result["boxes"] = boxes
        return result
if __name__ == "__main__":
    import cv2

    detector = ContainerSealDetector()

    test_images = [
        "test_images/container_with_seal.jpg",
        "test_images/container_without_seal.jpg",
    ]

    for test_image in test_images:
        img = cv2.imread(test_image)
        if img is not None:
            result = detector.detect_seal(img)
            if result["success"]:
                status = "Seal Present" if result["present"] else "No Seal"
                print(f"{test_image}: {status} (Confidence: {result['confidence']:.2f})")
            else:
                print(f"{test_image}: Detection failed - {result['error']}")
                
