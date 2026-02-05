"""
Container detection + reading pipeline.
Detects container number region and reads ISO 6346 number via character model.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from .character_detector import CharacterDetector
from .container_number import normalize_container_number


class ContainerNumberDetector:
    """YOLO-based container number detector + reader."""

    def __init__(
        self,
        model_path=None,
        use_character_model=True,
        character_model_path="models/konteyner_karakter_best.pt",
        cnrs_character_model_path="models/cnrs_recognition.pt",
        use_cnrs_character_model=True,
        fallback_character_model_path="models/character_best.pt",
        use_fallback_character_model=True,
        cnrs_model_path="models/cnrs_detection.pt",
        use_cnrs_detection_fallback=True,
        detect_conf=0.25,
        detect_low_conf=0.12,
        detect_imgsz=960,
        detect_iou=0.5,
        min_bbox_area_ratio=0.0005,
    ):
        project_root = Path(__file__).resolve().parents[1]
        preferred_path = project_root / "models" / "cnrs_detection.pt"
        default_path = project_root / "models" / "konteyner_ROI.pt"
        fallback_path = project_root / "models" / "find_knt_ISO_best.pt"
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = preferred_path if preferred_path.exists() else default_path
        self.fallback_path = fallback_path
        self.model = None
        self.cnrs_model_path = Path(cnrs_model_path)
        self.use_cnrs_detection_fallback = use_cnrs_detection_fallback
        self.cnrs_model = None
        self.use_character_model = use_character_model
        self.character_model_path = character_model_path
        self.char_detector = None
        self.cnrs_character_model_path = cnrs_character_model_path
        self.use_cnrs_character_model = use_cnrs_character_model
        self.cnrs_char_detector = None
        self.fallback_character_model_path = fallback_character_model_path
        self.use_fallback_character_model = use_fallback_character_model
        self.fallback_char_detector = None
        self.detect_conf = detect_conf
        self.detect_low_conf = detect_low_conf
        self.detect_imgsz = detect_imgsz
        self.detect_iou = detect_iou
        self.min_bbox_area_ratio = min_bbox_area_ratio

        self.load_model()

    def load_model(self):
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(str(self.model_path))
                print(f"Container number model loaded: {self.model_path}")
            elif self.fallback_path.exists():
                self.model_path = self.fallback_path
                self.model = YOLO(str(self.model_path))
                print(f"Container number model loaded: {self.model_path}")
            else:
                print(f"WARN: Container number model not found: {self.model_path}")
                self.model = None
        except Exception as exc:
            print(f"Container number model load error: {exc}")
            self.model = None

        if (
            self.use_cnrs_detection_fallback
            and self.cnrs_model_path.exists()
            and self.cnrs_model_path.resolve() != self.model_path.resolve()
        ):
            try:
                self.cnrs_model = YOLO(str(self.cnrs_model_path))
                print(f"CNRS detection model loaded: {self.cnrs_model_path}")
            except Exception as exc:
                print(f"CNRS detection model load error: {exc}")
                self.cnrs_model = None

    def load_character_model(self):
        if not self.use_character_model:
            return
        try:
            if Path(self.character_model_path).exists():
                self.char_detector = CharacterDetector(self.character_model_path)
            else:
                self.char_detector = None
        except Exception as exc:
            print(f"Container character model load error: {exc}")
            self.char_detector = None

    def load_fallback_character_model(self):
        if not (self.use_character_model and self.use_fallback_character_model):
            return
        try:
            if Path(self.fallback_character_model_path).exists():
                self.fallback_char_detector = CharacterDetector(self.fallback_character_model_path)
            else:
                self.fallback_char_detector = None
        except Exception as exc:
            print(f"Fallback character model load error: {exc}")
            self.fallback_char_detector = None

    def load_cnrs_character_model(self):
        if not (self.use_character_model and self.use_cnrs_character_model):
            return
        try:
            if Path(self.cnrs_character_model_path).exists():
                self.cnrs_char_detector = CharacterDetector(self.cnrs_character_model_path)
            else:
                self.cnrs_char_detector = None
        except Exception as exc:
            print(f"CNRS character model load error: {exc}")
            self.cnrs_char_detector = None

    @staticmethod
    def _prepare_for_char_model(img):
        h, w = img.shape[:2]
        pad_x = max(2, int(w * 0.08))
        pad_y = max(2, int(h * 0.10))
        img = cv2.copyMakeBorder(
            img, pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE,
        )
        h2, w2 = img.shape[:2]
        if max(h2, w2) < 180:
            scale = 2.5
        elif max(h2, w2) < 260:
            scale = 2.0
        elif max(h2, w2) < 340:
            scale = 1.5
        else:
            scale = 1.0
        if scale != 1.0:
            img = cv2.resize(
                img,
                (int(w2 * scale), int(h2 * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        return img

    @staticmethod
    def _score_char_result(model_data):
        if not model_data or not model_data.get("success") or not model_data.get("text"):
            return -1.0
        conf = float(model_data.get("confidence", 0.0) or 0.0)
        check_ok = model_data.get("check_digit_ok")
        if check_ok is True:
            conf += 0.12
        elif check_ok is False:
            conf -= 0.04
        return conf

    @staticmethod
    def _class_name(model, class_id):
        if model is None:
            return ""
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return str(names.get(class_id, ""))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return ""

    def _allowed_labels_for_model(self, model):
        names = getattr(model, "names", None)
        if not names:
            return None
        if isinstance(names, dict):
            all_names = [str(v).lower() for v in names.values()]
        else:
            all_names = [str(v).lower() for v in names]
        candidates = {"container_number_h", "container_number_v", "container_number"}
        if any(n in candidates for n in all_names):
            return {n for n in candidates if n in all_names}
        return None

    @staticmethod
    def _score_candidate(confidence, area_ratio):
        if area_ratio is None:
            size_score = 0.0
        else:
            size_score = min(1.0, float(area_ratio) / 0.02)
        return 0.7 * float(confidence) + 0.3 * size_score

    def _detect_boxes(
        self,
        image,
        model,
        conf_threshold,
        low_conf_threshold,
        imgsz,
        iou,
        min_bbox_area_ratio,
    ):
        if model is None or image is None or getattr(image, "size", 0) == 0:
            return []

        h, w = image.shape[:2]
        frame_area = float(h * w) if h and w else 0.0
        allowed = self._allowed_labels_for_model(model)

        def _run(conf):
            try:
                return model(
                    image,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    verbose=False,
                )
            except Exception:
                return []

        def _collect(results):
            boxes = []
            for r in results or []:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self._class_name(model, cls)
                    if allowed and class_name.lower() not in allowed:
                        continue
                    area = float((x2 - x1) * (y2 - y1))
                    area_ratio = area / frame_area if frame_area else 0.0
                    if min_bbox_area_ratio and area_ratio < min_bbox_area_ratio:
                        continue
                    score = self._score_candidate(conf, area_ratio)
                    boxes.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "area_ratio": area_ratio,
                        "score": score,
                        "class_id": cls,
                        "class_name": class_name,
                    })
            boxes.sort(key=lambda b: b["score"], reverse=True)
            return boxes

        boxes = _collect(_run(conf_threshold))
        if not boxes and low_conf_threshold is not None and low_conf_threshold < conf_threshold:
            boxes = _collect(_run(low_conf_threshold))
        return boxes

    def read_number_from_crop(
        self,
        crop,
        model_conf_threshold=0.12,
        image_is_rgb=False,
        full_image=None,
        coords=None,
        use_full_image_for_char=False,
        sort_by="x",
        debug_dir=None,
    ):
        if crop is None or getattr(crop, "size", 0) == 0:
            return "", 0.0, None

        crop_bgr = crop
        if image_is_rgb:
            try:
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            except Exception:
                crop_bgr = crop

        full_bgr = full_image
        if full_image is not None and image_is_rgb:
            try:
                full_bgr = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
            except Exception:
                full_bgr = full_image

        model_data = None

        # Character model
        if self.use_character_model:
            if self.char_detector is None:
                self.load_character_model()
            char_source = crop_bgr
            if use_full_image_for_char and full_bgr is not None:
                char_source = full_bgr
            elif full_bgr is not None and coords is not None and len(coords) == 4:
                x1, y1, x2, y2 = coords
                fh, fw = full_bgr.shape[:2]
                x1 = max(0, min(int(x1), fw - 1))
                y1 = max(0, min(int(y1), fh - 1))
                x2 = max(0, min(int(x2), fw))
                y2 = max(0, min(int(y2), fh))
                if x2 > x1 and y2 > y1:
                    char_source = full_bgr[y1:y2, x1:x2]

            char_input = self._prepare_for_char_model(char_source)

            primary_data = None
            if self.char_detector is not None:
                primary_data = self.char_detector.detect_characters(
                    char_input, conf_threshold=model_conf_threshold, sort_by=sort_by
                )
                if primary_data:
                    primary_data["source_model"] = "container_char"

            if primary_data and primary_data.get("success") and primary_data.get("text"):
                norm = normalize_container_number(primary_data.get("text"))
                if norm:
                    primary_data["normalized"] = norm
                    primary_data["text"] = norm["text"]
                    primary_data["formatted"] = norm["formatted"]
                    primary_data["check_digit_ok"] = norm["check_digit_ok"]

            model_data = primary_data

            need_fallback = (
                not model_data
                or not model_data.get("success")
                or not model_data.get("text")
                or (model_data.get("check_digit_ok") is False)
            )

            fallback_candidates = []

            if need_fallback and self.use_cnrs_character_model:
                if self.cnrs_char_detector is None:
                    self.load_cnrs_character_model()
                if self.cnrs_char_detector is not None:
                    cnrs_data = self.cnrs_char_detector.detect_characters(
                        char_input, conf_threshold=model_conf_threshold, sort_by=sort_by
                    )
                    if cnrs_data:
                        cnrs_data["source_model"] = "cnrs_char"
                    if cnrs_data and cnrs_data.get("success") and cnrs_data.get("text"):
                        norm = normalize_container_number(cnrs_data.get("text"))
                        if norm:
                            cnrs_data["normalized"] = norm
                            cnrs_data["text"] = norm["text"]
                            cnrs_data["formatted"] = norm["formatted"]
                            cnrs_data["check_digit_ok"] = norm["check_digit_ok"]
                    fallback_candidates.append(cnrs_data)

            if need_fallback and self.use_fallback_character_model:
                if self.fallback_char_detector is None:
                    self.load_fallback_character_model()
                if self.fallback_char_detector is not None:
                    fallback_data = self.fallback_char_detector.detect_characters(
                        char_input, conf_threshold=model_conf_threshold, sort_by=sort_by
                    )
                    if fallback_data:
                        fallback_data["source_model"] = "plate_char_fallback"
                    if fallback_data and fallback_data.get("success") and fallback_data.get("text"):
                        norm = normalize_container_number(fallback_data.get("text"))
                        if norm:
                            fallback_data["normalized"] = norm
                            fallback_data["text"] = norm["text"]
                            fallback_data["formatted"] = norm["formatted"]
                            fallback_data["check_digit_ok"] = norm["check_digit_ok"]
                    fallback_candidates.append(fallback_data)

            for cand in fallback_candidates:
                if self._score_char_result(cand) > self._score_char_result(model_data):
                    model_data = cand

            if model_data is None:
                model_data = {
                    "success": False,
                    "text": "",
                    "confidence": 0.0,
                    "characters": [],
                    "processing_time": 0.0,
                    "error": "Character model not found",
                }

            if debug_dir and char_input is not None:
                try:
                    Path(debug_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(Path(debug_dir) / "container_char_input.jpg"), char_input)
                    if model_data and model_data.get("characters"):
                        drawer = self.char_detector or self.cnrs_char_detector or self.fallback_char_detector
                        if drawer is not None:
                            annotated = drawer.draw_detections(
                                char_input, model_data["characters"]
                            )
                            cv2.imwrite(
                                str(Path(debug_dir) / "container_char_detections.jpg"), annotated
                            )
                except Exception as exc:
                    print(f"Debug save error: {exc}")

        # Choose best candidate
        best_text = ""
        best_conf = 0.0
        candidates = []

        if model_data and model_data.get("success") and model_data.get("text"):
            conf = float(model_data.get("confidence", 0.0) or 0.0)
            if model_data.get("check_digit_ok"):
                conf = min(1.0, conf + 0.05)
            candidates.append(("model", model_data["text"], conf))

        if candidates:
            _, best_text, best_conf = max(candidates, key=lambda x: x[2])

        return best_text, best_conf, model_data

    def detect_container_number(
        self,
        image_path,
        conf_threshold=None,
        low_conf_threshold=None,
        imgsz=None,
        iou=None,
        min_bbox_area_ratio=None,
        read_text=True,
    ):
        result = {
            "success": False,
            "image": None,
            "coordinates": [],
            "confidence": [],
            "area_ratio": [],
            "scores": [],
            "number_texts": [],
            "number_confidence": [],
            "number_sources": [],
            "model_data": [],
            "processing_time": 0.0,
            "error": None,
        }

        if self.model is None and self.cnrs_model is None:
            result["error"] = "Container number model not loaded"
            return result

        if not Path(image_path).exists():
            result["error"] = f"Image not found: {image_path}"
            return result

        try:
            start_time = time.time()
            image = cv2.imread(str(image_path))
            if image is None:
                result["error"] = "Image load failed"
                return result

            annotated = image.copy()
            conf_threshold = self.detect_conf if conf_threshold is None else conf_threshold
            low_conf_threshold = self.detect_low_conf if low_conf_threshold is None else low_conf_threshold
            imgsz = self.detect_imgsz if imgsz is None else imgsz
            iou = self.detect_iou if iou is None else iou
            min_bbox_area_ratio = (
                self.min_bbox_area_ratio if min_bbox_area_ratio is None else min_bbox_area_ratio
            )

            candidates = self._detect_boxes(
                image,
                self.model,
                conf_threshold,
                low_conf_threshold,
                imgsz,
                iou,
                min_bbox_area_ratio,
            )
            if not candidates and self.cnrs_model is not None:
                candidates = self._detect_boxes(
                    image,
                    self.cnrs_model,
                    conf_threshold,
                    low_conf_threshold,
                    imgsz,
                    iou,
                    min_bbox_area_ratio,
                )

            for cand in candidates:
                x1, y1, x2, y2 = cand["bbox"]
                conf = cand["confidence"]
                area_ratio = cand["area_ratio"]
                score = cand["score"]

                result["coordinates"].append((x1, y1, x2, y2))
                result["confidence"].append(conf)
                result["area_ratio"].append(area_ratio)
                result["scores"].append(score)

                number_text = ""
                number_conf = 0.0
                number_source = None
                model_data = None

                if read_text:
                    crop = image[y1:y2, x1:x2]
                    class_name = (cand.get("class_name") or "").lower()
                    sort_by = "y" if class_name.endswith("_v") else "x"
                    number_text, number_conf, model_data = self.read_number_from_crop(
                        crop,
                        image_is_rgb=False,
                        full_image=image,
                        coords=(x1, y1, x2, y2),
                        sort_by=sort_by,
                    )
                    if model_data and model_data.get("success") and model_data.get("text"):
                        number_source = "model"

                result["number_texts"].append(number_text)
                result["number_confidence"].append(number_conf)
                result["number_sources"].append(number_source)
                result["model_data"].append(model_data)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 128, 255), 2)

                label = "Container"
                if number_text:
                    label = f"{number_text} ({conf:.2f})"
                else:
                    label = f"Container {conf:.2f}"

                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    (0, 128, 255),
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

            result["processing_time"] = time.time() - start_time
            result["success"] = True
            result["image"] = annotated
            if not result["coordinates"]:
                result["error"] = "Container number not detected"

        except Exception as exc:
            result["error"] = f"Detection error: {exc}"

        return result

    def detect_container_number_in_image(
        self,
        image,
        conf_threshold=None,
        low_conf_threshold=None,
        imgsz=None,
        iou=None,
        min_bbox_area_ratio=None,
        read_text=False,
    ):
        result = {
            "success": False,
            "image": None,
            "coordinates": [],
            "confidence": [],
            "area_ratio": [],
            "scores": [],
            "number_texts": [],
            "number_confidence": [],
            "number_sources": [],
            "model_data": [],
            "processing_time": 0.0,
            "error": None,
        }

        if self.model is None and self.cnrs_model is None:
            result["error"] = "Container number model not loaded"
            return result

        if image is None or getattr(image, "size", 0) == 0:
            result["error"] = "Empty image"
            return result

        try:
            start_time = time.time()
            annotated = image.copy()
            conf_threshold = self.detect_conf if conf_threshold is None else conf_threshold
            low_conf_threshold = self.detect_low_conf if low_conf_threshold is None else low_conf_threshold
            imgsz = self.detect_imgsz if imgsz is None else imgsz
            iou = self.detect_iou if iou is None else iou
            min_bbox_area_ratio = (
                self.min_bbox_area_ratio if min_bbox_area_ratio is None else min_bbox_area_ratio
            )

            candidates = self._detect_boxes(
                image,
                self.model,
                conf_threshold,
                low_conf_threshold,
                imgsz,
                iou,
                min_bbox_area_ratio,
            )
            if not candidates and self.cnrs_model is not None:
                candidates = self._detect_boxes(
                    image,
                    self.cnrs_model,
                    conf_threshold,
                    low_conf_threshold,
                    imgsz,
                    iou,
                    min_bbox_area_ratio,
                )

            for cand in candidates:
                x1, y1, x2, y2 = cand["bbox"]
                conf = cand["confidence"]
                area_ratio = cand["area_ratio"]
                score = cand["score"]

                result["coordinates"].append((x1, y1, x2, y2))
                result["confidence"].append(conf)
                result["area_ratio"].append(area_ratio)
                result["scores"].append(score)

                number_text = ""
                number_conf = 0.0
                number_source = None
                model_data = None

                if read_text:
                    crop = image[y1:y2, x1:x2]
                    class_name = (cand.get("class_name") or "").lower()
                    sort_by = "y" if class_name.endswith("_v") else "x"
                    number_text, number_conf, model_data = self.read_number_from_crop(
                        crop,
                        image_is_rgb=False,
                        full_image=image,
                        coords=(x1, y1, x2, y2),
                        sort_by=sort_by,
                    )
                    if model_data and model_data.get("success") and model_data.get("text"):
                        number_source = "model"

                result["number_texts"].append(number_text)
                result["number_confidence"].append(number_conf)
                result["number_sources"].append(number_source)
                result["model_data"].append(model_data)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 128, 255), 2)

            result["processing_time"] = time.time() - start_time
            result["success"] = True
            result["image"] = annotated
            if not result["coordinates"]:
                result["error"] = "Container number not detected"
        except Exception as exc:
            result["error"] = f"Detection error: {exc}"

        return result

    def save_result(self, image, output_path):
        try:
            cv2.imwrite(str(output_path), image)
            return True
        except Exception as exc:
            print(f"Save error: {exc}")
            return False
