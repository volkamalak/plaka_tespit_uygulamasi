"""
Character Detector Modülü
Character-level YOLO model kullanarak plakanın bireysel karakterlerini tespit eder
"""

import cv2
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class CharacterDetector:
    """Character-level YOLO modeli kullanarak karakter tespiti yapan sınıf"""

    def __init__(self, model_path='models/character_best.pt'):
        """
        CharacterDetector sınıfını başlatır

        Args:
            model_path (str): Character YOLO model dosyasının yolu
        """
        self.model_path = model_path
        self.model = None
        self.device = None
        self.names = None
        self.load_model()

    def load_model(self):
        """Character YOLO modelini yükler"""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                self.names = self.model.names
                # GPU kontrol et
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.model.to('cuda')
                        self.device = 0
                        print(f"✓ Character Model GPU'da yüklendi")
                    else:
                        self.device = 'cpu'
                        print("⚠ GPU bulunamadı, Character Model CPU'da çalışacak")
                except:
                    self.device = None
                print(f"✓ Character Model yüklendi: {self.model_path}")
            else:
                print(f"⚠ Character model bulunamadı: {self.model_path}")
        except Exception as e:
            print(f"❌ Character Model yükleme hatası: {e}")
            self.device = None

    @staticmethod
    def _pad_and_resize(plate_image, target=640):
        """Plakayı kareye yakın getirip modele uygun boyuta büyütür."""
        h, w = plate_image.shape[:2]
        pad_x = max(2, int(w * 0.12))
        pad_y = max(2, int(h * 0.18))
        padded = cv2.copyMakeBorder(
            plate_image, pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE
        )
        ph, pw = padded.shape[:2]
        scale = target / max(ph, pw)
        if scale != 1.0:
            padded = cv2.resize(
                padded,
                (int(pw * scale), int(ph * scale)),
                interpolation=cv2.INTER_CUBIC
            )
        return padded

    def detect_characters(self, plate_image, conf_threshold=0.12):
        """
        Plaka görüntüsündeki karakterleri tespit eder

        Args:
            plate_image (numpy.ndarray): Plaka görüntüsü (BGR)
            conf_threshold (float): Güven eşiği (0-1)

        Returns:
            dict: Tespit sonuçları
        """
        result = {
            'success': False,
            'characters': [],
            'text': '',
            'confidence': 0.0,
            'processing_time': 0.0,
            'error': None
        }

        if self.model is None:
            result['error'] = 'Character model yüklü değil'
            return result

        try:
            import time
            start_time = time.time()

            # Karakter modeli için pad + resize
            plate_image = self._pad_and_resize(plate_image, target=640)

            # YOLO ile karakterleri tespit et (device otomatik)
            results = self.model(
                plate_image,
                conf=conf_threshold,
                verbose=False,
                imgsz=640,
                device=self.device
            )

            # Hiç kutu yoksa daha düşük eşikle ikinci deneme
            if not any(len(r.boxes) for r in results):
                results = self.model(
                    plate_image,
                    conf=0.08,
                    verbose=False,
                    imgsz=640,
                    device=self.device
                )

            characters = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    char_label = self._class_to_char(cls)
                    if not char_label:
                        continue

                    character = {
                        'bbox': (x1, y1, x2, y2),
                        'center_x': (x1 + x2) / 2,
                        'confidence': conf,
                        'class': cls,
                        'char': char_label
                    }
                    characters.append(character)

            # Karakterleri soldan sağa sırala (x koordinatına göre)
            characters = sorted(characters, key=lambda c: c['center_x'])

            # Metni oluştur
            text = ''.join([c['char'] for c in characters])
            avg_confidence = np.mean([c['confidence'] for c in characters]) if characters else 0.0

            elapsed = time.time() - start_time

            if not characters:
                result['success'] = False
                result['error'] = 'Karakter bulunamadı'
            else:
                result['success'] = True
                result['characters'] = characters
                result['text'] = text
                result['confidence'] = avg_confidence
            result['processing_time'] = elapsed

        except Exception as e:
            result['error'] = f"Character tespit hatası: {str(e)}"
            import traceback
            traceback.print_exc()

        return result

    @staticmethod
    def _normalize_label(label):
        label = str(label).strip()
        # Patterns like "-0-" -> "0"
        if len(label) >= 3 and label[0] == "-" and label[-1] == "-":
            label = label[1:-1]
        if label.isdigit():
            return label
        if len(label) == 1 and label.isalpha():
            return label.upper()
        # Fallback: extract first alnum
        for ch in label:
            if ch.isdigit():
                return ch
            if ch.isalpha():
                return ch.upper()
        return ""

    def _class_to_char(self, class_id):
        """
        Class ID'sini karaktere çevirir

        Args:
            class_id (int): YOLO class ID

        Returns:
            str: Karakter
        """
        if isinstance(self.names, dict):
            label = self.names.get(class_id, "")
            return self._normalize_label(label)
        if isinstance(self.names, (list, tuple)) and 0 <= class_id < len(self.names):
            return self._normalize_label(self.names[class_id])
        # Fallback: eski varsayım (0-9, A-Z)
        if 0 <= class_id <= 9:
            return str(class_id)
        if 10 <= class_id <= 35:
            return chr(ord('A') + (class_id - 10))
        return ""

    def draw_detections(self, plate_image, characters):
        """
        Tespit edilen karakterleri resme çizer

        Args:
            plate_image (numpy.ndarray): Plaka görüntüsü
            characters (list): Tespit edilen karakterler

        Returns:
            numpy.ndarray: Çizilmiş resim
        """
        annotated = plate_image.copy()

        for i, char in enumerate(characters):
            x1, y1, x2, y2 = char['bbox']
            conf = char['confidence']
            text = char['char']

            # Bounding box çiz
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Karakter ve güven yazısı
            label = f"{text} {conf:.1%}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                annotated, label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1
            )

        return annotated


if __name__ == "__main__":
    # Test kodu
    detector = CharacterDetector()

    if detector.model:
        # Test plakası varsa çalıştır
        test_plate = "test_plate.jpg"
        if Path(test_plate).exists():
            plate_image = cv2.imread(test_plate)
            result = detector.detect_characters(plate_image)

            if result['success']:
                print(f"✓ Karakterler tespit edildi!")
                print(f"Okunan metni: {result['text']}")
                print(f"Ortalama güven: {result['confidence']:.1%}")
                print(f"İşlem süresi: {result['processing_time']:.3f}s")
                print(f"\nTespit edilen karakterler:")
                for i, char in enumerate(result['characters']):
                    print(f"  {i+1}. {char['char']} (Güven: {char['confidence']:.1%})")
            else:
                print(f"❌ Tespit başarısız: {result['error']}")
    else:
        print("❌ Model yüklenemedi")
