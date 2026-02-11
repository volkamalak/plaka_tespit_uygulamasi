"""
Container damage detector.
Supports YOLO detection or classification models.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


class ContainerDamageDetector:
    """Detects container damage and reports detected damage types."""

    def __init__(
        self,
        model_path=None,
        damage_labels=None,
        no_damage_labels=None,
        min_damage_conf=0.50,
        min_class_margin=0.08,
        decision_margin=0.10,
        overlap_iou=0.65,
        secondary_type_window=0.15,
        min_box_area_ratio=0.0002,
        fallback_conf=0.1,
        fallback_imgsz=1280,
        tile_imgsz=1600,
    ):
        project_root = Path(__file__).resolve().parents[1]
        preferred_path = project_root / "models" / "konteyner_hasar.pt"
        default_path = project_root / "models" / "container_damage.pt"
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = preferred_path if preferred_path.exists() else default_path
        self.fallback_path = default_path
        self.model = None

        default_damage = {
            "damage",
            "damaged",
            "dent",
            "dented",
            "scratch",
            "scratched",
            "crack",
            "cracked",
            "hole",
            "corrosion",
            "rust",
            "deformation",
            "bent",
            "tear",
            "broken",
            "breakage",
            "paint_damage",
            "paint_peel",
        }
        default_no_damage = {
            "no_damage",
            "healthy",
            "normal",
            "intact",
            "undamaged",
            "no_defect",
            "no_damage_detected",
        }

        self.damage_labels = {self._normalize_key(s) for s in (damage_labels or default_damage)}
        self.no_damage_labels = {
            self._normalize_key(s) for s in (no_damage_labels or default_no_damage)
        }
        self.min_damage_conf = float(min_damage_conf)
        self.min_class_margin = float(min_class_margin)
        self.decision_margin = float(decision_margin)
        self.overlap_iou = float(overlap_iou)
        self.secondary_type_window = float(secondary_type_window)
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.fallback_conf = float(fallback_conf)
        self.fallback_imgsz = int(fallback_imgsz) if fallback_imgsz else None
        self.tile_imgsz = int(tile_imgsz) if tile_imgsz else None

        # Canonical output names for common classes from various datasets.
        self.class_aliases = {
            "damage": "damage",
            "damaged": "damage",
            "dent": "dent",
            "dented": "dent",
            "scratch": "scratch",
            "scratched": "scratch",
            "crack": "crack",
            "cracked": "crack",
            "fracture": "crack",
            "hole": "hole",
            "puncture": "hole",
            "corrosion": "corrosion",
            "rust": "corrosion",
            "deformation": "deformation",
            "bent": "deformation",
            "bend": "deformation",
            "paint_damage": "paint_damage",
            "paint_peel": "paint_damage",
            "peeling": "paint_damage",
            "tear": "tear",
            "broken": "broken",
            "break": "broken",
            "breakage": "breakage",
            # Numeric labels seen in some public container-damage datasets.
            "0": "dent",
            "1": "scratch",
            "2": "crack",
            "3": "corrosion",
            "4": "deformation",
            "5": "breakage",
            "object": "no_damage",
            "no_damage": "no_damage",
            "healthy": "no_damage",
            "normal": "no_damage",
            "intact": "no_damage",
            "undamaged": "no_damage",
            "no_defect": "no_damage",
            "no_damage_detected": "no_damage",
        }

        self.load_model()

    @staticmethod
    def _normalize_key(name):
        key = str(name or "").strip().lower()
        key = key.replace("-", "_").replace(" ", "_")
        while "__" in key:
            key = key.replace("__", "_")
        return key

    def load_model(self):
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(str(self.model_path))
                print(f"Container damage model loaded: {self.model_path}")
            elif self.fallback_path.exists():
                self.model_path = self.fallback_path
                self.model = YOLO(str(self.model_path))
                print(f"Container damage model loaded: {self.model_path}")
            else:
                print(f"WARN: Container damage model not found: {self.model_path}")
                self.model = None
        except Exception as exc:
            print(f"Container damage model load error: {exc}")
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

    def _canonical_type(self, class_name):
        key = self._normalize_key(class_name)
        if key in self.class_aliases:
            return self.class_aliases[key]

        # Fallback to token-level matching for labels like "major_dent".
        for token in key.split("_"):
            if token in self.class_aliases:
                return self.class_aliases[token]

        if key in self.no_damage_labels:
            return "no_damage"
        if key in self.damage_labels:
            return key
        return key

    @staticmethod
    def _summarize_types(detections):
        if not detections:
            return [], []

        grouped = {}
        for item in detections:
            d_type = item.get("damage_type") or "damage"
            conf = float(item.get("confidence", 0.0) or 0.0)
            stat = grouped.setdefault(d_type, {"type": d_type, "count": 0, "max_confidence": 0.0})
            stat["count"] += 1
            if conf > stat["max_confidence"]:
                stat["max_confidence"] = conf

        summary = list(grouped.values())
        summary.sort(key=lambda x: (x["max_confidence"], x["count"]), reverse=True)
        damage_types = [item["type"] for item in summary]
        return summary, damage_types

    @staticmethod
    def _box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0

        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _suppress_overlaps(self, boxes):
        """Suppress heavy overlaps to reduce cross-class flicker."""
        if not boxes:
            return []
        if self.overlap_iou <= 0:
            return list(boxes)

        ordered = sorted(
            boxes,
            key=lambda item: float(item.get("confidence", 0.0) or 0.0),
            reverse=True,
        )
        kept = []
        for cand in ordered:
            cand_bbox = cand.get("bbox")
            if not cand_bbox:
                continue
            is_overlap = False
            for prev in kept:
                prev_bbox = prev.get("bbox")
                if prev_bbox and self._box_iou(cand_bbox, prev_bbox) >= self.overlap_iou:
                    is_overlap = True
                    break
            if not is_overlap:
                kept.append(cand)
        return kept

    def _classification_second_conf(self, probs):
        top5conf = getattr(probs, "top5conf", None)
        if top5conf is not None:
            try:
                if hasattr(top5conf, "detach"):
                    values = top5conf.detach().cpu().float().reshape(-1).tolist()
                else:
                    values = [float(x) for x in top5conf]
                if len(values) > 1:
                    values = sorted((float(v) for v in values), reverse=True)
                    return values[1]
            except Exception:
                pass

        data = getattr(probs, "data", None)
        if data is not None:
            try:
                if hasattr(data, "detach"):
                    values = data.detach().cpu().float().reshape(-1).tolist()
                else:
                    values = [float(x) for x in data]
                if len(values) > 1:
                    values.sort(reverse=True)
                    return float(values[1])
            except Exception:
                pass

        return 0.0

    def _filter_secondary_types(self, summary):
        if not summary:
            return [], []
        leader_conf = float(summary[0].get("max_confidence", 0.0) or 0.0)
        min_keep = max(self.min_damage_conf, leader_conf - self.secondary_type_window)
        filtered = [item for item in summary if float(item.get("max_confidence", 0.0) or 0.0) >= min_keep]
        damage_types = [item.get("type", "") for item in filtered if item.get("type")]
        return filtered, damage_types

    def detect_damage(self, image, conf_threshold=0.25):
        result = {
            "success": False,
            "has_damage": None,
            "damage_type": "",
            "damage_types": [],
            "summary": [],
            "confidence": 0.0,
            "class_name": "",
            "boxes": [],
            "low_confidence": False,
            "error": None,
        }

        if self.model is None:
            result["error"] = "Damage model not loaded"
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
                    "has_damage": None,
                    "damage_type": "",
                    "damage_types": [],
                    "summary": [],
                    "confidence": 0.0,
                    "class_name": "",
                    "boxes": [],
                    "low_confidence": False,
                    "error": f"Damage detection error: {exc}",
                }, False, False, True)

            if not results:
                return ({
                    "success": False,
                    "has_damage": None,
                    "damage_type": "",
                    "damage_types": [],
                    "summary": [],
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
                        "has_damage": None,
                        "damage_type": "",
                        "damage_types": [],
                        "summary": [],
                        "confidence": 0.0,
                        "class_name": "",
                        "boxes": [],
                        "low_confidence": False,
                        "error": "Classification output not available",
                    }, False, True, True)

                class_name = self._class_name(top_idx)
                d_type = self._canonical_type(class_name)
                has_damage = d_type != "no_damage"
                second_conf = self._classification_second_conf(probs)
                class_margin = max(0.0, top_conf - second_conf)

                if top_conf < self.min_damage_conf or class_margin < self.min_class_margin:
                    return ({
                        "success": True,
                        "has_damage": None,
                        "damage_type": d_type if d_type != "no_damage" else "",
                        "damage_types": [d_type] if d_type and d_type != "no_damage" else [],
                        "summary": (
                            [{"type": d_type, "count": 1, "max_confidence": top_conf}]
                            if d_type and d_type != "no_damage"
                            else []
                        ),
                        "confidence": top_conf,
                        "class_name": class_name,
                        "boxes": [],
                        "low_confidence": True,
                        "error": None,
                    }, False, True, False)

                return ({
                    "success": True,
                    "has_damage": has_damage,
                    "damage_type": d_type if has_damage else "",
                    "damage_types": [d_type] if has_damage and d_type else [],
                    "summary": [{"type": d_type, "count": 1, "max_confidence": top_conf}] if has_damage and d_type else [],
                    "confidence": top_conf,
                    "class_name": class_name,
                    "boxes": [],
                    "low_confidence": False,
                    "error": None,
                }, False, True, False)

            # Detection path
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
                d_type = self._canonical_type(class_name)

                item = {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf_val,
                    "class_id": cls,
                    "class_name": class_name,
                    "damage_type": d_type,
                }
                boxes.append(item)

            boxes = self._suppress_overlaps(boxes)

            damage_boxes = [item for item in boxes if item.get("damage_type") != "no_damage"]
            no_damage_boxes = [item for item in boxes if item.get("damage_type") == "no_damage"]
            best_damage = (
                max(damage_boxes, key=lambda b: float(b.get("confidence", 0.0) or 0.0))
                if damage_boxes
                else None
            )
            no_damage_best = (
                max(no_damage_boxes, key=lambda b: float(b.get("confidence", 0.0) or 0.0))
                if no_damage_boxes
                else None
            )

            if best_damage and no_damage_best:
                dmg_conf = float(best_damage.get("confidence", 0.0) or 0.0)
                no_dmg_conf = float(no_damage_best.get("confidence", 0.0) or 0.0)
                gap = dmg_conf - no_dmg_conf
                if no_dmg_conf >= self.min_damage_conf and gap <= -self.decision_margin:
                    return ({
                        "success": True,
                        "has_damage": False,
                        "damage_type": "",
                        "damage_types": [],
                        "summary": [],
                        "confidence": no_dmg_conf,
                        "class_name": no_damage_best.get("class_name", ""),
                        "boxes": boxes,
                        "low_confidence": False,
                        "error": None,
                    }, True, False, False)
                if (
                    max(dmg_conf, no_dmg_conf) >= self.min_damage_conf
                    and abs(gap) < self.decision_margin
                ):
                    summary, damage_types = self._summarize_types(damage_boxes)
                    summary, damage_types = self._filter_secondary_types(summary)
                    return ({
                        "success": True,
                        "has_damage": None,
                        "damage_type": best_damage.get("damage_type", "") if gap >= 0 else "",
                        "damage_types": damage_types if gap >= 0 else [],
                        "summary": summary if gap >= 0 else [],
                        "confidence": max(dmg_conf, no_dmg_conf),
                        "class_name": best_damage.get("class_name", "") if gap >= 0 else no_damage_best.get("class_name", ""),
                        "boxes": boxes,
                        "low_confidence": True,
                        "error": None,
                    }, True, False, False)

            if best_damage:
                summary, damage_types = self._summarize_types(damage_boxes)
                summary, damage_types = self._filter_secondary_types(summary)
                best_conf = float(best_damage.get("confidence", 0.0) or 0.0)

                if best_conf < self.min_damage_conf:
                    return ({
                        "success": True,
                        "has_damage": None,
                        "damage_type": best_damage.get("damage_type", ""),
                        "damage_types": damage_types,
                        "summary": summary,
                        "confidence": best_conf,
                        "class_name": best_damage.get("class_name", ""),
                        "boxes": boxes,
                        "low_confidence": True,
                        "error": None,
                    }, True, False, False)

                return ({
                    "success": True,
                    "has_damage": True,
                    "damage_type": best_damage.get("damage_type", ""),
                    "damage_types": damage_types,
                    "summary": summary,
                    "confidence": best_conf,
                    "class_name": best_damage.get("class_name", ""),
                    "boxes": boxes,
                    "low_confidence": False,
                    "error": None,
                }, True, False, False)

            if no_damage_best is not None:
                no_dmg_conf = float(no_damage_best.get("confidence", 0.0) or 0.0)
                if no_dmg_conf < self.min_damage_conf:
                    return ({
                        "success": True,
                        "has_damage": None,
                        "damage_type": "",
                        "damage_types": [],
                        "summary": [],
                        "confidence": no_dmg_conf,
                        "class_name": no_damage_best.get("class_name", ""),
                        "boxes": boxes,
                        "low_confidence": True,
                        "error": None,
                    }, True, False, False)
                return ({
                    "success": True,
                    "has_damage": False,
                    "damage_type": "",
                    "damage_types": [],
                    "summary": [],
                    "confidence": no_dmg_conf,
                    "class_name": no_damage_best.get("class_name", ""),
                    "boxes": boxes,
                    "low_confidence": False,
                    "error": None,
                }, True, False, False)

            if boxes:
                best = max(boxes, key=lambda b: float(b.get("confidence", 0.0) or 0.0))
                summary, damage_types = self._summarize_types(boxes)
                summary, damage_types = self._filter_secondary_types(summary)
                best_conf = float(best.get("confidence", 0.0) or 0.0)
                if best_conf < self.min_damage_conf:
                    return ({
                        "success": True,
                        "has_damage": None,
                        "damage_type": best.get("damage_type", ""),
                        "damage_types": damage_types,
                        "summary": summary,
                        "confidence": best_conf,
                        "class_name": best.get("class_name", ""),
                        "boxes": boxes,
                        "low_confidence": True,
                        "error": None,
                    }, True, False, False)
                return ({
                    "success": True,
                    "has_damage": True,
                    "damage_type": best.get("damage_type", ""),
                    "damage_types": damage_types,
                    "summary": summary,
                    "confidence": best_conf,
                    "class_name": best.get("class_name", ""),
                    "boxes": boxes,
                    "low_confidence": False,
                    "error": None,
                }, True, False, False)

            return ({
                "success": False,
                "has_damage": None,
                "damage_type": "",
                "damage_types": [],
                "summary": [],
                "confidence": 0.0,
                "class_name": "",
                "boxes": [],
                "low_confidence": False,
                "error": "No damage detected",
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
        best_score = (-1, -1.0)
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
                        has_damage = cand.get("has_damage") is True
                        score = (1 if has_damage else 0, conf_val)
                        if score > best_score:
                            best = cand
                            best_score = score

        if best is not None:
            return best

        # No detections -> do not force "no damage"; report as low confidence.
        result.update({
            "success": True,
            "has_damage": None,
            "damage_type": "",
            "damage_types": [],
            "summary": [],
            "confidence": 0.0,
            "class_name": "",
            "boxes": [],
            "low_confidence": True,
            "error": None,
        })
        return result
    


