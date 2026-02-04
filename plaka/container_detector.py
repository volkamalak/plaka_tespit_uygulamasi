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
        model_path="models/find_knt_ISO_best.pt",
        use_character_model=True,
        character_model_path="models/iso_karakter_okuma_best.pt",
    ):
        self.model_path = model_path
        self.model = None
        self.use_character_model = use_character_model
        self.character_model_path = character_model_path
        self.char_detector = None

        self.load_model()

    def load_model(self):
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                print(f"Container number model loaded: {self.model_path}")
            else:
                print(f"WARN: Container number model not found: {self.model_path}")
                self.model = None
        except Exception as exc:
            print(f"Container number model load error: {exc}")
            self.model = None

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

    def read_number_from_crop(
        self,
        crop,
        model_conf_threshold=0.12,
        image_is_rgb=False,
        full_image=None,
        coords=None,
        use_full_image_for_char=False,
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
            if self.char_detector is not None:
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
                model_data = self.char_detector.detect_characters(
                    char_input, conf_threshold=model_conf_threshold
                )

                if model_data and model_data.get("success") and model_data.get("text"):
                    norm = normalize_container_number(model_data.get("text"))
                    if norm:
                        model_data["normalized"] = norm
                        model_data["text"] = norm["text"]
                        model_data["formatted"] = norm["formatted"]
                        model_data["check_digit_ok"] = norm["check_digit_ok"]

                if debug_dir:
                    try:
                        Path(debug_dir).mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(Path(debug_dir) / "container_char_input.jpg"), char_input)
                        if model_data and model_data.get("characters"):
                            annotated = self.char_detector.draw_detections(
                                char_input, model_data["characters"]
                            )
                            cv2.imwrite(
                                str(Path(debug_dir) / "container_char_detections.jpg"), annotated
                            )
                    except Exception as exc:
                        print(f"Debug save error: {exc}")
            else:
                model_data = {
                    "success": False,
                    "text": "",
                    "confidence": 0.0,
                    "characters": [],
                    "processing_time": 0.0,
                    "error": "Character model not found",
                }

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

    def detect_container_number(self, image_path, conf_threshold=0.25, read_text=True):
        result = {
            "success": False,
            "image": None,
            "coordinates": [],
            "confidence": [],
            "number_texts": [],
            "number_confidence": [],
            "number_sources": [],
            "model_data": [],
            "processing_time": 0.0,
            "error": None,
        }

        if self.model is None:
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
            results = self.model(image, conf=conf_threshold, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    result["coordinates"].append((x1, y1, x2, y2))
                    result["confidence"].append(conf)

                    number_text = ""
                    number_conf = 0.0
                    number_source = None
                    model_data = None

                    if read_text:
                        crop = image[y1:y2, x1:x2]
                        number_text, number_conf, model_data = self.read_number_from_crop(
                            crop,
                            image_is_rgb=False,
                            full_image=image,
                            coords=(x1, y1, x2, y2),
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

    def detect_container_number_in_image(self, image, conf_threshold=0.25, read_text=False):
        result = {
            "success": False,
            "image": None,
            "coordinates": [],
            "confidence": [],
            "number_texts": [],
            "number_confidence": [],
            "number_sources": [],
            "model_data": [],
            "processing_time": 0.0,
            "error": None,
        }

        if self.model is None:
            result["error"] = "Container number model not loaded"
            return result

        if image is None or getattr(image, "size", 0) == 0:
            result["error"] = "Empty image"
            return result

        try:
            start_time = time.time()
            annotated = image.copy()
            results = self.model(image, conf=conf_threshold, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    result["coordinates"].append((x1, y1, x2, y2))
                    result["confidence"].append(conf)

                    number_text = ""
                    number_conf = 0.0
                    number_source = None
                    model_data = None

                    if read_text:
                        crop = image[y1:y2, x1:x2]
                        number_text, number_conf, model_data = self.read_number_from_crop(
                            crop,
                            image_is_rgb=False,
                            full_image=image,
                            coords=(x1, y1, x2, y2),
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
