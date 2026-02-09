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
        min_present_conf=0.2,
        min_box_area_ratio=0.001,
        fallback_conf=0.1,
        fallback_imgsz=1280,
        tile_imgsz=1600,
    ):
        project_root = Path(__file__).resolve().parents[1]
        default_path = project_root / "models" / "muhur_bulma.pt"
        fallback_path = project_root / "models" / "container_seal.pt"
        self.model_path = Path(model_path) if model_path else default_path
        self.fallback_path = fallback_path
        self.model = None
        self.present_labels = {s.lower() for s in (present_labels or ["seal", "present", "var"])}
        self.absent_labels = {s.lower() for s in (absent_labels or ["no_seal", "absent", "yok"])}
        self.min_present_conf = float(min_present_conf)
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.fallback_conf = float(fallback_conf)
        self.fallback_imgsz = int(fallback_imgsz) if fallback_imgsz else None
        self.tile_imgsz = int(tile_imgsz) if tile_imgsz else None
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
            "low_confidence": False,
            "error": None,
        }

        if self.model is None:
            result["error"] = "Seal model not loaded"
            return result

        if image is None:
            result["error"] = "Empty image"
            return result

        # Numpy images have .size as int; PIL images have .size as tuple.
        if hasattr(image, "shape"):
            if image.shape[0] == 0 or image.shape[1] == 0:
                result["error"] = "Empty image"
                return result
        elif isinstance(getattr(image, "size", None), int) and image.size == 0:
            result["error"] = "Empty image"
            return result

        def _infer(img, conf, imgsz=None, offset=(0, 0), full_shape=None):
            """Run model once and parse result. Returns (result, has_boxes, is_cls, fatal)."""
            try:
                kwargs = {"conf": conf, "verbose": False}
                if imgsz is not None:
                    kwargs["imgsz"] = imgsz
                results = self.model(img, **kwargs)
            except Exception as exc:
                return ({
                    "success": False,
                    "present": None,
                    "confidence": 0.0,
                    "class_name": "",
                    "boxes": [],
                    "low_confidence": False,
                    "error": f"Seal detection error: {exc}",
                }, False, False, True)

            if not results:
                return ({
                    "success": False,
                    "present": None,
                    "confidence": 0.0,
                    "class_name": "",
                    "boxes": [],
                    "low_confidence": False,
                    "error": "No result",
                }, False, False, True)

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
                    return ({
                        "success": False,
                        "present": None,
                        "confidence": 0.0,
                        "class_name": "",
                        "boxes": [],
                        "low_confidence": False,
                        "error": "Classification output not available",
                    }, False, True, True)

                class_name = self._class_name(top_idx)
                class_key = class_name.lower()

                if class_key in self.present_labels:
                    present = True
                elif class_key in self.absent_labels:
                    present = False
                else:
                    present = True

                if present and top_conf < self.min_present_conf:
                    return ({
                        "success": True,
                        "present": None,
                        "confidence": top_conf,
                        "class_name": class_name,
                        "boxes": [],
                        "low_confidence": True,
                        "error": None,
                    }, False, True, False)

                return ({
                    "success": True,
                    "present": present,
                    "confidence": top_conf,
                    "class_name": class_name,
                    "boxes": [],
                    "low_confidence": False,
                    "error": None,
                }, False, True, False)

            # Detection path
            present_best = (None, 0.0, "")
            absent_best = (None, 0.0, "")
            boxes = []
            offset_x, offset_y = offset
            full_area = None
            if full_shape is not None and len(full_shape) >= 2:
                full_area = int(full_shape[0]) * int(full_shape[1])

            for box in getattr(r0, "boxes", []):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y
                if full_area and self.min_box_area_ratio > 0:
                    bw = max(0, x2 - x1)
                    bh = max(0, y2 - y1)
                    if full_area > 0 and (bw * bh) / float(full_area) < self.min_box_area_ratio:
                        continue
                conf_val = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self._class_name(cls)
                class_key = class_name.lower()

                boxes.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf_val,
                    "class_id": cls,
                    "class_name": class_name,
                })

                if class_key in self.present_labels:
                    if conf_val > present_best[1]:
                        present_best = ((x1, y1, x2, y2), conf_val, class_name)
                elif class_key in self.absent_labels:
                    if conf_val > absent_best[1]:
                        absent_best = ((x1, y1, x2, y2), conf_val, class_name)

            if present_best[0] is not None:
                if present_best[1] < self.min_present_conf:
                    return ({
                        "success": True,
                        "present": None,
                        "confidence": present_best[1],
                        "class_name": present_best[2],
                        "boxes": boxes,
                        "low_confidence": True,
                        "error": None,
                    }, True, False, False)
                return ({
                    "success": True,
                    "present": True,
                    "confidence": present_best[1],
                    "class_name": present_best[2],
                    "boxes": boxes,
                    "low_confidence": False,
                    "error": None,
                }, True, False, False)

            if absent_best[0] is not None:
                return ({
                    "success": True,
                    "present": False,
                    "confidence": absent_best[1],
                    "class_name": absent_best[2],
                    "boxes": boxes,
                    "low_confidence": False,
                    "error": None,
                }, True, False, False)

            if boxes:
                # If labels are unknown, fall back to presence by any detection.
                best_box = max(boxes, key=lambda b: b["confidence"])
                if best_box["confidence"] < self.min_present_conf:
                    return ({
                        "success": True,
                        "present": None,
                        "confidence": best_box["confidence"],
                        "class_name": best_box["class_name"],
                        "boxes": boxes,
                        "low_confidence": True,
                        "error": None,
                    }, True, False, False)
                return ({
                    "success": True,
                    "present": True,
                    "confidence": best_box["confidence"],
                    "class_name": best_box["class_name"],
                    "boxes": boxes,
                    "low_confidence": False,
                    "error": None,
                }, True, False, False)

            return ({
                "success": False,
                "present": None,
                "confidence": 0.0,
                "class_name": "",
                "boxes": [],
                "low_confidence": False,
                "error": "No seal detected",
            }, False, False, False)

        full_shape = getattr(image, "shape", None)
        primary, has_boxes, is_cls, fatal = _infer(image, conf_threshold, full_shape=full_shape)
        if fatal:
            return primary
        if is_cls or has_boxes:
            return primary

        fallback_conf = min(conf_threshold, self.fallback_conf)
        fallback_imgsz = self.fallback_imgsz

        # Fallback: run once on full image with lower confidence & larger size.
        fallback_full, has_boxes, is_cls, fatal = _infer(
            image,
            fallback_conf,
            imgsz=fallback_imgsz,
            full_shape=full_shape,
        )
        if fatal:
            return fallback_full
        if is_cls or has_boxes:
            return fallback_full

        # Fallback: tiled search (2x2) for small objects.
        best = None
        best_conf = -1.0
        if hasattr(image, "shape") and len(image.shape) >= 2:
            height, width = image.shape[:2]
            for row in range(2):
                for col in range(2):
                    x1 = int(width * col / 2)
                    x2 = int(width * (col + 1) / 2)
                    y1 = int(height * row / 2)
                    y2 = int(height * (row + 1) / 2)
                    crop = image[y1:y2, x1:x2]
                    if getattr(crop, "size", 0) == 0:
                        continue
                    cand, has_boxes, is_cls, fatal = _infer(
                        crop,
                        fallback_conf,
                        imgsz=self.tile_imgsz or fallback_imgsz,
                        offset=(x1, y1),
                        full_shape=full_shape,
                    )
                    if fatal:
                        return cand
                    if is_cls:
                        return cand
                    if has_boxes:
                        conf_val = float(cand.get("confidence", 0.0) or 0.0)
                        if conf_val > best_conf:
                            best = cand
                            best_conf = conf_val

        if best is not None:
            return best

        # No detections -> treat as seal absent, not an error.
        result.update({
            "success": True,
            "present": False,
            "confidence": 0.0,
            "class_name": "",
            "boxes": [],
            "low_confidence": False,
            "error": None,
        })
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
