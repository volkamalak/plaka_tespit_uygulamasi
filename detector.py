"""
Plaka Tespit Modülü
YOLO kullanarak resimlerde plaka tespiti yapar
"""

import cv2
import time
from ultralytics import YOLO
from pathlib import Path


class PlakaDetector:
    """YOLO kullanarak plaka tespiti yapan sınıf"""

    def __init__(self, model_path='models/best.pt'):
        """
        PlakaDetector sınıfını başlatır

        Args:
            model_path (str): YOLO model dosyasının yolu
        """
        self.model_path = model_path
        self.model = None
        self.load_model()

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

    def detect_plate(self, image_path, conf_threshold=0.25):
        """
        Verilen resimdeki plakayı tespit eder

        Args:
            image_path (str): Resim dosyasının yolu
            conf_threshold (float): Güven eşiği (0-1 arası)

        Returns:
            dict: Tespit sonuçları
                - success (bool): Tespit başarılı mı?
                - image (numpy.ndarray): İşlenmiş resim
                - coordinates (list): Plaka koordinatları [(x1, y1, x2, y2), ...]
                - confidence (list): Güven skorları
                - processing_time (float): İşlem süresi (saniye)
                - error (str): Hata mesajı (varsa)
        """
        result = {
            'success': False,
            'image': None,
            'coordinates': [],
            'confidence': [],
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

                    # Çerçeve çiz
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Güven skorunu yaz
                    label = f'Plaka {conf:.2f}'
                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
    detector = PlakaDetector()

    # Test resmi varsa çalıştır
    test_image = "test.jpg"
    if Path(test_image).exists():
        result = detector.detect_plate(test_image)

        if result['success']:
            print(f"Tespit başarılı!")
            print(f"İşlem süresi: {result['processing_time']:.3f} saniye")
            print(f"Bulunan plaka sayısı: {len(result['coordinates'])}")

            for i, (coords, conf) in enumerate(zip(result['coordinates'], result['confidence'])):
                print(f"Plaka {i+1}: {coords}, Güven: {conf:.2f}")
        else:
            print(f"Hata: {result['error']}")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
