"""
Plaka Tespit Modülü
YOLO kullanarak resimlerde plaka tespiti yapar
OCR ile plaka numarasını okur
Character model ile metin okur
"""

import cv2
import time
from ultralytics import YOLO
from pathlib import Path
from .ocr import PlakaOCR
from .character_detector import CharacterDetector


class PlakaDetector:
    """YOLO kullanarak plaka tespiti yapan sınıf"""

    def __init__(
        self,
        model_path='models/find_plate_best.pt',
        use_ocr=True,
        ocr_languages=['tur', 'eng'],
        use_character_model=True,
        character_model_path='models/character_best.pt',
    ):
        """
        PlakaDetector sınıfını başlatır

        Args:
            model_path (str): YOLO model dosyasının yolu
            use_ocr (bool): OCR kullanılsın mı?
            ocr_languages (list): OCR dilleri (tesseract kodları)
        """
        self.model_path = model_path
        self.model = None
        self.use_ocr = use_ocr
        self.ocr = None
        self.use_character_model = use_character_model
        self.character_model_path = character_model_path
        self.char_detector = None

        self.load_model()

        if self.use_ocr:
            self.load_ocr(ocr_languages)

    def load_character_model(self):
        """Character modelini yükler (lazy)."""
        if not self.use_character_model:
            return
        try:
            if Path(self.character_model_path).exists():
                self.char_detector = CharacterDetector(self.character_model_path)
            else:
                self.char_detector = None
        except Exception as e:
            print(f"Character model yükleme hatası: {e}")
            self.char_detector = None

    def load_model(self):
        """YOLO modelini yükler"""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                print(f"Model başarıyla yüklendi: {self.model_path}")
            else:
                print(f"UYARI: Model dosyası bulunamadı: {self.model_path}")
                print("Lütfen best.pt modelini models/ klasörüne yerleştirin.")
                self.model = None
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            self.model = None

    def load_ocr(self, languages):
        """OCR modülünü yükler"""
        try:
            print("OCR modülü yükleniyor...")
            self.ocr = PlakaOCR(languages=languages)
            print("OCR modülü başarıyla yüklendi!")
        except Exception as e:
            print(f"OCR yükleme hatası: {e}")
            self.ocr = None
            self.use_ocr = False

    def detect_plate(self, image_path, conf_threshold=0.25, read_text=True):
        """
        Verilen resimdeki plakayı tespit eder ve okur

        Args:
            image_path (str): Resim dosyasının yolu
            conf_threshold (float): Güven eşiği (0-1 arası)
            read_text (bool): Plaka metnini oku

        Returns:
            dict: Tespit sonuçları
                - success (bool): Tespit başarılı mı?
                - image (numpy.ndarray): İşlenmiş resim
                - coordinates (list): Plaka koordinatları [(x1, y1, x2, y2), ...]
                - confidence (list): Güven skorları
                - plate_texts (list): Okunan plaka metinleri
                - ocr_confidence (list): OCR güven skorları
                - processing_time (float): İşlem süresi (saniye)
                - error (str): Hata mesajı (varsa)
        """
        result = {
            'success': False,
            'image': None,
            'coordinates': [],
            'confidence': [],
            'plate_texts': [],
            'ocr_confidence': [],
            'processing_time': 0,
            'error': None
        }

        # Model kontrolü
        if self.model is None:
            result['error'] = "Model yüklü değil. Lütfen best.pt modelini models/ klasörüne yerleştirin."
            return result

        # Resim kontrolü
        if not Path(image_path).exists():
            result['error'] = f"Resim bulunamadı: {image_path}"
            return result

        try:
            # Zamanı başlat
            start_time = time.time()

            # Resmi yükle
            image = cv2.imread(str(image_path))
            if image is None:
                result['error'] = "Resim yüklenemedi"
                return result

            # Kopyasını al (orijinali korumak için)
            annotated_image = image.copy()

            # YOLO ile tespit yap
            results = self.model(image, conf=conf_threshold, verbose=False)

            # Sonuçları işle
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Koordinatları al
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # Koordinatları kaydet
                    result['coordinates'].append((x1, y1, x2, y2))
                    result['confidence'].append(conf)

                    # OCR ile plaka metnini oku
                    plate_text = ""
                    ocr_conf = 0.0

                    if read_text and self.use_ocr and self.ocr:
                        ocr_result = self.ocr.read_plate_from_coordinates(
                            image, (x1, y1, x2, y2)
                        )
                        if ocr_result['success']:
                            plate_text = ocr_result['text']
                            ocr_conf = ocr_result['confidence']

                    result['plate_texts'].append(plate_text)
                    result['ocr_confidence'].append(ocr_conf)

                    # Çerçeve çiz
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Plaka metnini yaz (eğer varsa)
                    if plate_text:
                        label = f'{plate_text} ({conf:.2f})'
                        # Arka plan kutusu
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        cv2.rectangle(
                            annotated_image,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1),
                            (0, 255, 0),
                            -1
                        )
                        cv2.putText(
                            annotated_image, label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 2
                        )
                    else:
                        # Sadece güven skorunu yaz
                        label = f'Plaka {conf:.2f}'
                        cv2.putText(
                            annotated_image, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2
                        )

            # İşlem süresini hesapla
            end_time = time.time()
            result['processing_time'] = end_time - start_time

            # Sonucu güncelle
            result['success'] = True
            result['image'] = annotated_image

            if len(result['coordinates']) == 0:
                result['error'] = "Plaka tespit edilemedi"

        except Exception as e:
            result['error'] = f"Tespit hatası: {str(e)}"

        return result

    def detect_plate_in_image(self, image, conf_threshold=0.25, read_text=False):
        """
        Bellekteki görüntüde plaka tespit eder.

        Args:
            image (numpy.ndarray): BGR görüntü
            conf_threshold (float): Güven eşiği
            read_text (bool): OCR oku

        Returns:
            dict: detect_plate ile aynı yapı
        """
        result = {
            'success': False,
            'image': None,
            'coordinates': [],
            'confidence': [],
            'plate_texts': [],
            'ocr_confidence': [],
            'processing_time': 0,
            'error': None
        }

        if self.model is None:
            result['error'] = "Model yüklü değil."
            return result

        if image is None or getattr(image, "size", 0) == 0:
            result['error'] = "Görüntü boş"
            return result

        try:
            start_time = time.time()
            annotated_image = image.copy()
            results = self.model(image, conf=conf_threshold, verbose=False)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    result['coordinates'].append((x1, y1, x2, y2))
                    result['confidence'].append(conf)

                    plate_text = ""
                    ocr_conf = 0.0
                    if read_text and self.use_ocr and self.ocr:
                        ocr_result = self.ocr.read_plate_from_coordinates(
                            image, (x1, y1, x2, y2)
                        )
                        if ocr_result['success']:
                            plate_text = ocr_result['text']
                            ocr_conf = ocr_result['confidence']

                    result['plate_texts'].append(plate_text)
                    result['ocr_confidence'].append(ocr_conf)

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            end_time = time.time()
            result['processing_time'] = end_time - start_time
            result['success'] = True
            result['image'] = annotated_image
            if len(result['coordinates']) == 0:
                result['error'] = "Plaka tespit edilemedi"
        except Exception as e:
            result['error'] = f"Tespit hatası: {str(e)}"

        return result

    def read_plate_text(
        self,
        plate_image,
        model_conf_threshold=0.15,
        image_is_rgb=False,
        full_image=None,
        coords=None,
        use_full_image_for_char=False,
        debug_dir=None,
        use_ocr=None,
    ):
        """
        Plakayı hem Character Model hem OCR ile okur.

        Args:
            plate_image (numpy.ndarray): Plaka görüntüsü (BGR varsayılan)
            model_conf_threshold (float): Character model güven eşiği
            image_is_rgb (bool): Görüntü RGB ise True (GUI'den gelenler için)

        Returns:
            tuple: (plate_text, plate_conf, model_data, ocr_data)
        """
        if plate_image is None or getattr(plate_image, "size", 0) == 0:
            return "", 0.0, None, None

        # Gerekirse RGB -> BGR
        plate_bgr = plate_image
        if image_is_rgb:
            try:
                plate_bgr = cv2.cvtColor(plate_image, cv2.COLOR_RGB2BGR)
            except Exception:
                plate_bgr = plate_image

        full_bgr = full_image
        if full_image is not None and image_is_rgb:
            try:
                full_bgr = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
            except Exception:
                full_bgr = full_image

        model_data = None
        ocr_data = None

        def _prepare_for_char_model(img):
            """Character model için pad + büyütme."""
            h, w = img.shape[:2]
            pad_x = max(2, int(w * 0.08))
            pad_y = max(2, int(h * 0.10))
            img = cv2.copyMakeBorder(
                img, pad_y, pad_y, pad_x, pad_x,
                borderType=cv2.BORDER_REPLICATE
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
                    interpolation=cv2.INTER_CUBIC
                )
            return img

        # Character model
        if self.use_character_model:
            if self.char_detector is None:
                self.load_character_model()
            if self.char_detector is not None:
                # İstenirse character model tam görüntüde çalışır
                char_source = plate_bgr
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

                char_input = _prepare_for_char_model(char_source)
                model_data = self.char_detector.detect_characters(
                    char_input, conf_threshold=model_conf_threshold
                )

                if debug_dir:
                    try:
                        Path(debug_dir).mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(Path(debug_dir) / "char_input.jpg"), char_input)
                        if model_data and model_data.get("characters"):
                            annotated = self.char_detector.draw_detections(
                                char_input, model_data["characters"]
                            )
                            cv2.imwrite(str(Path(debug_dir) / "char_detections.jpg"), annotated)
                    except Exception as e:
                        print(f"Debug kaydetme hatası: {e}")
            else:
                model_data = {
                    "success": False,
                    "text": "",
                    "confidence": 0.0,
                    "characters": [],
                    "processing_time": 0.0,
                    "error": "Character model bulunamadı",
                }

        # OCR (BGR ve GRAY)
        use_ocr_flag = self.use_ocr if use_ocr is None else use_ocr
        if use_ocr_flag and self.ocr:
            ocr_bgr = self.ocr.read_plate(plate_bgr)
            try:
                gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = plate_bgr
            ocr_gray = self.ocr.read_plate(gray)

            # En iyi OCR sonucunu seç
            best_ocr = ocr_bgr
            if ocr_gray and ocr_gray.get("success"):
                if (not ocr_bgr or not ocr_bgr.get("success")) or (
                    ocr_gray.get("confidence", 0.0) > ocr_bgr.get("confidence", 0.0)
                ):
                    best_ocr = ocr_gray

            ocr_data = {
                "best": best_ocr,
                "bgr": ocr_bgr,
                "gray": ocr_gray,
            }

        # En iyi sonucu seç (varsa)
        plate_text = ""
        plate_conf = 0.0
        candidates = []
        if model_data and model_data.get("success") and model_data.get("text"):
            candidates.append(("model", model_data["text"], model_data.get("confidence", 0.0)))
        if ocr_data and ocr_data.get("best") and ocr_data["best"].get("success") and ocr_data["best"].get("text"):
            candidates.append((
                "ocr",
                ocr_data["best"]["text"],
                ocr_data["best"].get("confidence", 0.0)
            ))

        if candidates:
            _, plate_text, plate_conf = max(candidates, key=lambda x: x[2])

        return plate_text, plate_conf, model_data, ocr_data

    def detect_plate_in_stream(
        self,
        cap,
        conf_threshold=0.25,
        burst_frames=10,
        read_text=True,
        min_bbox_area_ratio=0.002,
        detect_every_n=1,
    ):
        """
        Video/stream içinde N frame boyunca çalışıp en iyi sonucu döndürür.
        """
        result = {
            "success": False,
            "best_plate_text": "",
            "best_plate_conf": 0.0,
            "best_bbox": None,
            "best_detector_conf": 0.0,
            "best_sharpness": 0.0,
            "best_plate_crop": None,
            "last_frame": None,
            "processing_time": 0.0,
            "debug": {"candidates": []},
            "error": None,
        }

        start_time = time.time()

        if self.model is None:
            result["error"] = "Model yüklü değil."
            result["processing_time"] = time.time() - start_time
            return result

        if cap is None or not hasattr(cap, "isOpened") or not cap.isOpened():
            result["error"] = "Video kaynağı açılamadı."
            result["processing_time"] = time.time() - start_time
            return result

        if burst_frames <= 0:
            result["error"] = "burst_frames 0'dan büyük olmalı."
            result["processing_time"] = time.time() - start_time
            return result

        if detect_every_n < 1:
            detect_every_n = 1

        raw_candidates = []
        sharpness_values = []
        last_bbox = None
        last_det_conf = 0.0
        last_frame = None

        try:
            for frame_idx in range(burst_frames):
                ok, frame = cap.read()
                if not ok or frame is None:
                    if frame_idx == 0:
                        result["error"] = "Frame okunamadı."
                    break

                last_frame = frame
                h, w = frame.shape[:2]
                frame_area = float(h * w) if h and w else 0.0

                # Belirli aralıklarla tespit yap, aralarda bbox'u reuse et
                run_detect = (frame_idx % detect_every_n == 0) or last_bbox is None
                bbox = None
                det_conf = 0.0

                if run_detect:
                    best_score = None
                    best_bbox = None
                    best_conf = 0.0

                    # detect_plate_in_image ile aynı çekirdek mantık (YOLO + bbox seçimi)
                    results = self.model(frame, conf=conf_threshold, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])

                            bw = max(0, x2 - x1)
                            bh = max(0, y2 - y1)
                            area = float(bw * bh)
                            if frame_area <= 0:
                                continue
                            area_norm = area / frame_area

                            # Çok küçük bbox'ları ele
                            if area_norm < min_bbox_area_ratio:
                                continue

                            score = 0.7 * conf + 0.3 * area_norm
                            if best_score is None or score > best_score:
                                best_score = score
                                best_bbox = (x1, y1, x2, y2)
                                best_conf = conf

                    if best_bbox:
                        last_bbox = best_bbox
                        last_det_conf = best_conf
                        bbox = best_bbox
                        det_conf = best_conf
                    else:
                        last_bbox = None
                        last_det_conf = 0.0
                else:
                    bbox = last_bbox
                    det_conf = last_det_conf

                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox
                x1 = max(0, min(int(x1), w - 1))
                y1 = max(0, min(int(y1), h - 1))
                x2 = max(0, min(int(x2), w))
                y2 = max(0, min(int(y2), h))
                if x2 <= x1 or y2 <= y1:
                    continue

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop is None or plate_crop.size == 0:
                    continue

                # Keskinlik skoru (Laplacian variance)
                try:
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                except Exception:
                    gray = plate_crop
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                plate_text = ""
                plate_conf = 0.0
                if read_text:
                    plate_text, plate_conf, _, _ = self.read_plate_text(
                        plate_crop,
                        image_is_rgb=False,
                        use_ocr=False,
                    )

                raw_candidates.append({
                    "frame_idx": frame_idx,
                    "text": plate_text,
                    "plate_conf": float(plate_conf or 0.0),
                    "det_conf": float(det_conf or 0.0),
                    "sharpness": sharpness,
                    "bbox": (x1, y1, x2, y2),
                    "_crop": plate_crop.copy(),
                })
                sharpness_values.append(sharpness)

        except Exception as e:
            result["error"] = f"Stream tespit hatası: {e}"
            result["processing_time"] = time.time() - start_time
            return result

        if not raw_candidates:
            if result["error"] is None:
                result["error"] = "Aday bulunamadı."
            result["processing_time"] = time.time() - start_time
            result["last_frame"] = last_frame
            return result

        min_sh = min(sharpness_values) if sharpness_values else 0.0
        max_sh = max(sharpness_values) if sharpness_values else 0.0
        denom = max_sh - min_sh

        best_score = None
        best_candidate = None

        for cand in raw_candidates:
            if denom > 0:
                sharp_norm = (cand["sharpness"] - min_sh) / denom
            else:
                sharp_norm = 0.0

            score = (
                0.55 * cand["plate_conf"]
                + 0.35 * sharp_norm
                + 0.10 * cand["det_conf"]
            )

            result["debug"]["candidates"].append({
                "frame_idx": cand["frame_idx"],
                "text": cand["text"],
                "plate_conf": cand["plate_conf"],
                "det_conf": cand["det_conf"],
                "sharpness": cand["sharpness"],
                "score": score,
                "bbox": cand["bbox"],
            })

            if best_score is None or score > best_score:
                best_score = score
                best_candidate = cand

        if not best_candidate:
            result["error"] = "En iyi aday seçilemedi."
            result["processing_time"] = time.time() - start_time
            result["last_frame"] = last_frame
            return result

        result["success"] = True
        result["best_plate_text"] = best_candidate["text"]
        result["best_plate_conf"] = best_candidate["plate_conf"]
        result["best_bbox"] = best_candidate["bbox"]
        result["best_detector_conf"] = best_candidate["det_conf"]
        result["best_sharpness"] = best_candidate["sharpness"]
        result["best_plate_crop"] = best_candidate.get("_crop")
        result["last_frame"] = last_frame
        result["processing_time"] = time.time() - start_time

        return result

    def save_result(self, image, output_path):
        """
        İşlenmiş resmi kaydeder

        Args:
            image (numpy.ndarray): Kaydedilecek resim
            output_path (str): Çıktı dosyasının yolu

        Returns:
            bool: Başarılı mı?
        """
        try:
            cv2.imwrite(str(output_path), image)
            return True
        except Exception as e:
            print(f"Resim kaydetme hatası: {e}")
            return False


if __name__ == "__main__":
    # Test kodu
    detector = PlakaDetector(use_ocr=True)

    # Test resmi varsa çalıştır
    test_image = "test.jpg"
    if Path(test_image).exists():
        result = detector.detect_plate(test_image)

        if result['success']:
            print(f"Tespit başarılı!")
            print(f"İşlem süresi: {result['processing_time']:.3f} saniye")
            print(f"Bulunan plaka sayısı: {len(result['coordinates'])}")

            for i, (coords, conf, text, ocr_conf) in enumerate(zip(
                result['coordinates'],
                result['confidence'],
                result['plate_texts'],
                result['ocr_confidence']
            )):
                print(f"\nPlaka {i+1}:")
                print(f"  Koordinatlar: {coords}")
                print(f"  Tespit Güveni: {conf:.2f}")
                if text:
                    print(f"  Plaka Numarası: {text}")
                    print(f"  OCR Güveni: {ocr_conf:.2f}")
        else:
            print(f"Hata: {result['error']}")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
    print("\n✅ Eğitim tamamlandı!")
    
