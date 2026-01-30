"""
OCR Modülü
Tesseract ve EasyOCR kullanarak plaka karakterlerini okur.
"""

from pathlib import Path
import re

import cv2
import numpy as np
import pytesseract

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class PlakaOCR:
    """Tesseract + EasyOCR tabanlı plaka OCR sınıfı."""

    def __init__(self, languages=None, min_conf=20):
        """
        PlakaOCR sınıfını başlatır.

        Args:
            languages (list|None): Tercih edilen diller (örn: ['tur', 'eng'])
            min_conf (int): Token güven eşiği (0-100, daha düşük = daha toleranslı)
        """
        self.languages = languages or ["tur", "eng"]
        self.min_conf = min_conf
        self.lang = self._resolve_lang(self.languages)
        self.province_codes = {f"{i:02d}" for i in range(1, 82)}
        
        # EasyOCR reader'ı başlat
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                
                if gpu_available:
                    # GPU'da çalıştır (RTX 2060 için optimize)
                    self.easyocr_reader = easyocr.Reader(
                        ['tr', 'en'], 
                        gpu=True, 
                        model_storage_directory='./.easyocr',
                        verbose=False  # Sesin kapat
                    )
                    print(f"✓ EasyOCR GPU'da yüklendi ({torch.cuda.get_device_name(0)})")
                else:
                    # CPU fallback
                    self.easyocr_reader = easyocr.Reader(
                        ['tr', 'en'], 
                        gpu=False, 
                        model_storage_directory='./.easyocr',
                        verbose=False
                    )
                    print("✓ EasyOCR CPU'da yüklendi")
                    
            except Exception as e:
                print(f"⚠ EasyOCR başarısız (Tesseract kullanılacak): {e}")
                self.easyocr_reader = None

    def _deskew(self, image):
        """Metin eğimini basitçe düzeltmeye çalışır."""
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=30, maxLineGap=10)
        if lines is None:
            return image

        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            if -30 <= angle <= 30:
                angles.append(angle)

        if not angles:
            return image

        median_angle = float(np.median(angles))
        if abs(median_angle) < 1.0:
            return image

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _tight_crop_text(self, gray):
        """Metin bölgesini sıkı kırpma (basit morfoloji ile)."""
        if gray is None or gray.size == 0:
            return gray

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, bw = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel2)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray

        best = None
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 0.4 * gray.shape[1] or h < 0.2 * gray.shape[0]:
                continue
            area = w * h
            if best is None or area > best[0]:
                best = (area, x, y, w, h)

        if best is None:
            return gray

        _, x, y, w, h = best
        pad_x = int(w * 0.03)
        pad_y = int(h * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(gray.shape[1], x + w + pad_x)
        y2 = min(gray.shape[0], y + h + pad_y)
        return gray[y1:y2, x1:x2]

    def _resolve_lang(self, languages):
        """Sistemde bulunan dilleri tespit edip uygun dili seçer."""
        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception:
            available = {"eng"}

        preferred = [lang for lang in languages if lang in available]
        if "tur" in preferred and "eng" in preferred:
            return "tur+eng"
        if preferred:
            return "+".join(preferred)
        if "tur" in available:
            return "tur"
        return "eng"

    def _preprocess_variants(self, plate_image):
        """OCR için birden fazla ön işleme varyantı üretir."""
        if plate_image is None or plate_image.size == 0:
            raise ValueError("Geçersiz plaka görüntüsü")

        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Büyütme (OCR için netlik)
        h, w = gray.shape[:2]
        scale = 3 if max(h, w) < 200 else (2 if max(h, w) < 300 else 1.5)  # Daha agresif büyütme
        gray_big = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Kontrast artırma (siyahları siyah, beyazları beyaz yapma yaklaşımı)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # Daha agresif
        clahe_img = clahe.apply(gray_big)

        deskewed = self._deskew(clahe_img)

        blur = cv2.bilateralFilter(deskewed, 5, 75, 75)

        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Yüksek kontrastlı siyah-beyaz versiyon
        high_contrast = cv2.normalize(deskewed, None, 0, 255, cv2.NORM_MINMAX)
        high_contrast = cv2.equalizeHist(high_contrast)

        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )

        # Erosion + Dilation (karakterleri belirginleştir)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        eroded = cv2.erode(blur, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        sharpen = cv2.addWeighted(blur, 1.5, cv2.GaussianBlur(blur, (0, 0), 2), -0.5, 0)
        tight = self._tight_crop_text(deskewed)
        
        # Histogram equalization da ekle
        hist_eq = cv2.equalizeHist(deskewed)

        return [
            ("gray", gray_big),
            ("clahe", clahe_img),
            ("deskew", deskewed),
            ("otsu", otsu),
            ("otsu_inv", otsu_inv),
            ("high_contrast", high_contrast),
            ("adaptive", adaptive),
            ("sharpen", sharpen),
            ("tight", tight),
            ("eroded_dilated", dilated),
            ("hist_eq", hist_eq),
        ]

    def _clean_plate_text(self, text):
        """OCR çıktısını temizler ve plaka formatına yaklaştırır."""
        if not text:
            return ""
        cleaned = re.sub(r"[^0-9A-Za-z]", "", text.upper())
        
        # Ortak karışıklıkları düzelt (Tesseract sıkça yapıyor)
        # 4 ve 1 karışıklığı
        cleaned = self._fix_common_confusions(cleaned)
        
        return self._normalize_turkish_plate(cleaned)

    def _fix_common_confusions(self, text):
        """Tesseract'ın yaygın karışıklıklarını düzeltmeye çalış."""
        if not text or len(text) < 7:
            return text
        
        # Son 4 karakter genelde rakam: XX YYY NNNN
        # Eğer hatalıysa düzelt
        
        # Hata patterni: 4→1, 7→1, 5→S, 0→O vb.
        # Kontekste göre düzeltme:
        
        fixed = list(text)
        
        # İlk 2 karakter digit olmalı - kontrol et
        for i in range(min(2, len(fixed))):
            if fixed[i] in 'OQDU':  # O-like characterler
                fixed[i] = '0'
            elif fixed[i] in 'ILJ':  # I-like characterler
                fixed[i] = '1'
        
        # Son 4 karakter digit olmalı (tipik format: 2 haneli)
        if len(fixed) >= 7:
            for i in range(len(fixed) - 4, len(fixed)):
                if fixed[i] in 'OQDU':
                    fixed[i] = '0'
                elif fixed[i] in 'ILJ':
                    fixed[i] = '1'
                elif fixed[i] == 'S' and i == len(fixed) - 1:
                    # Son karakter S ise 5 olabilir
                    fixed[i] = '5'
                elif fixed[i] == 'Z':
                    fixed[i] = '2'
        
        return "".join(fixed)

    def _map_chars(self, text, mapping):
        fixed = []
        changes = 0
        for ch in text:
            if ch in mapping:
                fixed.append(mapping[ch])
                changes += 1
            else:
                fixed.append(ch)
        return "".join(fixed), changes

    def _score_candidate(self, candidate, changes, mid_len):
        # Province penalty
        province = candidate.split(" ")[0]
        penalty = 0
        if province not in self.province_codes:
            penalty += 15
        # Prefer 2-3 letter plates over 1 letter (small bias)
        if mid_len == 1:
            penalty += 2
        return changes * 10 + penalty

    def _extract_province_hint(self, raw):
        """Ham OCR verisinden yüksek güvenli il kodu tahmini çıkar."""
        if not raw:
            return None

        best = None
        texts = []
        confs = []

        if isinstance(raw, dict) and "text" in raw and "conf" in raw:
            texts = raw["text"]
            confs = raw["conf"]
        elif isinstance(raw, list):
            # EasyOCR formatı: [bbox, text, conf]
            for item in raw:
                if len(item) >= 3:
                    texts.append(str(item[1]))
                    confs.append(float(item[2]) * 100.0)

        for txt, conf in zip(texts, confs):
            try:
                conf_val = float(conf)
            except Exception:
                continue
            if conf_val < 70:
                continue
            m = re.search(r"\b(\d{2})\b", str(txt))
            if not m:
                continue
            code = m.group(1)
            if code in self.province_codes:
                if best is None or conf_val > best[0]:
                    best = (conf_val, code)

        return best[1] if best else None

    def _normalize_turkish_plate(self, cleaned):
        """Türk plakası formatına göre düzeltme dener - MÜKEMMELİYETÇİ."""
        if not cleaned:
            return ""

        # 1. Doğrudan uygun ise - hassas eşleştirme
        direct = re.match(r"^(\d{2})([A-Z]{1,3})(\d{2,4})$", cleaned)
        if direct:
            return f"{direct.group(1)} {direct.group(2)} {direct.group(3)}"

        # 2. Çok kapsamlı karakter mapping
        digit_map = {
            # 0'a benzeyen
            "O": "0", "Q": "0", "D": "0", "U": "0", "ö": "0",
            # 1'e benzeyen
            "I": "1", "L": "1", "J": "1", "T": "1", "l": "1",
            # 2'ye benzeyen
            "Z": "2", "z": "2",
            # 3'e benzeyen
            "E": "3", "B": "3", "e": "3",
            # 4'e benzeyen
            "A": "4", "a": "4", "V": "4",
            # 5'e benzeyen
            "S": "5", "s": "5",
            # 6'ya benzeyen
            "G": "6", "C": "6", "g": "6", "c": "6",
            # 7'ye benzeyen
            "T": "7", "t": "7",
            # 8'e benzeyen
            "B": "8", "S": "8",
            # 9'a benzeyen
            "G": "9", "g": "9",
        }
        
        letter_map = {
            "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
            "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
        }

        best = None
        
        # 3. Her olasılığı dene (8-10 karakter, ilk 2 digit, ...)
        for start in range(max(0, len(cleaned) - 12), len(cleaned)):
            for total_len in range(8, 11):  # 8-10 karaktere izin ver
                end = start + total_len
                if end > len(cleaned):
                    break
                    
                sub = cleaned[start:end]
                
                # İlk 2 karakter digit olmalı
                for mid_len in (1, 2, 3, 4):  # 1-4 harf
                    tail_len = total_len - 2 - mid_len
                    if tail_len < 2 or tail_len > 4:
                        continue
                    
                    d1 = sub[:2]
                    mid = sub[2:2 + mid_len]
                    tail = sub[2 + mid_len:end]

                    # Mapping yap
                    d1_fixed, d1_changes = self._map_chars(d1, digit_map)
                    mid_fixed, mid_changes = self._map_chars(mid, letter_map)
                    tail_fixed, tail_changes = self._map_chars(tail, digit_map)

                    # Doğrulama
                    if not (d1_fixed.isdigit() and tail_fixed.isdigit() and mid_fixed.isalpha()):
                        continue

                    # İl kodu kontrolü (01-81)
                    try:
                        province_num = int(d1_fixed)
                        if not (1 <= province_num <= 81):
                            continue
                    except:
                        continue

                    changes = d1_changes + mid_changes + tail_changes
                    candidate = f"{d1_fixed} {mid_fixed} {tail_fixed}"
                    score = changes * 5  # Değişiklik sayısı * ağırlık
                    
                    # Hafif iyileştirme: daha az değişiklik daha iyi
                    if best is None or score < best[0]:
                        best = (score, candidate)

        if best:
            return best[1]

        # 4. Hiçbir şey uymazsa direkt döndür (formatsız)
        return cleaned

    def _run_tesseract(self, image, psm):
        """Tesseract OCR çalıştırır ve metin + güveni döner."""
        # Plaka-özel Tesseract konfigürasyonu
        config = (
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            "-c classify_bln_numeric_mode=1 "
            "-c preserve_interword_spaces=1 "
            "-c lstm_use_matrix=1 "
            "-c kmin=15 "
            "-c language_model_penalty_non_freq_dict_word=0.1 "
            "-c language_model_penalty_punc=0.1"
        )
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        tokens = []
        confidences = []
        for txt, conf in zip(data["text"], data["conf"]):
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = -1
            # Minimum güven çok düşük, hepsi al
            if txt and len(txt.strip()) > 0:
                tokens.append(txt)
                confidences.append(max(conf_val, 10))  # En düşük 10%

        if not tokens:
            raw_tokens = [t for t in data["text"] if t and len(t.strip()) > 0]
            if raw_tokens:
                tokens = raw_tokens
                confidences = [10] * len(raw_tokens)  # Minimal güven ver

        text = "".join(tokens).strip()  # Boşluklarsız birleştir
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        
        return text, mean_conf, data

    def _run_easyocr(self, image):
        """EasyOCR ile okuma yapar - GPU optimized."""
        if not self.easyocr_reader:
            return "", 0.0
        
        try:
            # GPU hızlı çalışması için parametreler
            results = self.easyocr_reader.readtext(
                image, 
                detail=1,
                paragraph=False,
                workers=1,  # GPU kullan
                batch_size=4  # Küçük batch (GPU memory sparing)
            )
            
            if not results:
                return "", 0.0
            
            texts = []
            confidences = []
            
            for result_item in results:
                if len(result_item) >= 3:
                    text = str(result_item[1]).strip().upper()
                    conf = float(result_item[2])
                else:
                    continue
                
                # Düşük eşik - her şeyi al
                if text and len(text) > 0 and conf > 0.1:
                    texts.append(text)
                    confidences.append(conf)
            
            if not texts:
                return "", 0.0
            
            text = "".join(texts).strip()  # Boşluksuz birleştir
            mean_conf = float(np.mean(confidences)) if confidences else 0.0
            
            return text, mean_conf * 100  # 0-100 arası
        except Exception as e:
            print(f"⚠ EasyOCR hatası: {e}")
            return "", 0.0

    def read_plate(self, plate_image):
        """
        Plaka görüntüsünden metin okur (Tesseract + EasyOCR + Validation).

        Returns:
            dict: OCR sonuçları
        """
        result = {
            "success": False,
            "text": "",
            "confidence": 0.0,
            "raw_results": None,
            "error": None,
        }

        try:
            variants = self._preprocess_variants(plate_image)
        except Exception as exc:
            result["error"] = f"OCR ön işleme hatası: {exc}"
            return result

        best = None
        candidates = []  # Tüm adayları sakla
        
        # PSM değerleri: 6=block, 7=line, 8=word (plakalar için 7,8 ideal)
        psm_modes = [8, 7, 6]  # Hızlı sıralama: word → line → block
        
        # TESSERACT ile dene (hızlı)
        for variant_name, img in variants[:8]:  # İlk 8 varyant
            for psm in psm_modes:
                text, conf, raw = self._run_tesseract(img, psm)
                cleaned = self._clean_plate_text(text)
                if not cleaned or len(cleaned) < 7:
                    continue

                # Doğrulama: format uyumluluğu
                is_valid_format = bool(re.match(r"^\d{2}\s?[A-Z]{1,3}\s?\d{2,4}$", cleaned))
                
                score = conf
                if is_valid_format:
                    score += 50  # Daha yüksek bonus

                candidate = {
                    'score': score,
                    'text': cleaned,
                    'conf': conf,
                    'raw': raw,
                    'method': f'Tesseract-{variant_name}-PSM{psm}',
                    'is_valid': is_valid_format
                }
                candidates.append(candidate)
                
                if best is None or candidate['score'] > best['score']:
                    best = candidate

        # EasyOCR ile dene (GPU hızlı)
        if self.easyocr_reader and (best is None or not best['is_valid']):
            for variant_name, img in variants[:3]:  # Sadece ilk 3 varyant (GPU hızlı)
                text, conf = self._run_easyocr(img)
                cleaned = self._clean_plate_text(text)
                if not cleaned or len(cleaned) < 7:
                    continue

                is_valid_format = bool(re.match(r"^\d{2}\s?[A-Z]{1,3}\s?\d{2,4}$", cleaned))
                score = conf + 60  # EasyOCR daha güvenilir
                if is_valid_format:
                    score += 30

                candidate = {
                    'score': score,
                    'text': cleaned,
                    'conf': conf,
                    'raw': None,
                    'method': f'EasyOCR-{variant_name}',
                    'is_valid': is_valid_format
                }
                candidates.append(candidate)
                
                if best is None or candidate['score'] > best['score']:
                    best = candidate
                
                # Valid format bulunursa kal
                if is_valid_format and conf > 60:
                    break

        if best is None:
            # Fallback: düşük güvenle de olsa sonuç döndür
            for variant_name, img in variants[:2]:
                for psm in [8, 7]:
                    config = (
                        f"--oem 3 --psm {psm} "
                        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    )
                    raw_text = pytesseract.image_to_string(img, lang=self.lang, config=config)
                    cleaned = self._clean_plate_text(raw_text)
                    if cleaned and len(cleaned) >= 7:
                        best = {
                            'score': 5.0,
                            'text': cleaned,
                            'conf': 25.0,
                            'raw': None,
                            'method': f'Fallback-{variant_name}-PSM{psm}',
                            'is_valid': False
                        }
                        break
                if best:
                    break

        if best is None:
            result["error"] = "Metin tespit edilemedi"
            return result

        result["success"] = True
        result["text"] = best['text']
        result["confidence"] = best['conf'] / 100.0  # 0-1 arası
        result["raw_results"] = best['raw']
        return result

    def read_plate_from_coordinates(self, full_image, coordinates):
        """Tam görüntüden koordinatlarla plakayı okur."""
        x1, y1, x2, y2 = coordinates
        h, w = full_image.shape[:2]

        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))

        if x2 <= x1 or y2 <= y1:
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "raw_results": None,
                "error": "Geçersiz plaka koordinatları",
            }

        pad_x = max(2, int((x2 - x1) * 0.05))
        pad_y = max(2, int((y2 - y1) * 0.1))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        plate_crop = full_image[y1:y2, x1:x2]
        if plate_crop is None or plate_crop.size == 0:
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "raw_results": None,
                "error": "Plaka bölgesi boş",
            }

        return self.read_plate(plate_crop)

    def batch_read_plates(self, full_image, coordinates_list):
        """Birden fazla plakayı okur."""
        results = []
        for coords in coordinates_list:
            ocr_result = self.read_plate_from_coordinates(full_image, coords)
            results.append({"coordinates": coords, "ocr": ocr_result})
        return results


if __name__ == "__main__":
    # Basit test
    test_image = "test_plate.jpg"
    if Path(test_image).exists():
        img = cv2.imread(test_image)
        ocr = PlakaOCR()
        out = ocr.read_plate(img)
        print(out)
    else:
        print("Test görüntüsü bulunamadı")
