#!/usr/bin/env python3
"""
OCR Modülünün Test Scripti
"""

import cv2
import numpy as np
from plaka.ocr import PlakaOCR

def test_plate_recognition():
    """OCR testleri yapıştır."""
    
    print("=" * 60)
    print("OCR MODÜLÜNÜ TEST ETME")
    print("=" * 60)
    
    # OCR nesnesini oluştur
    ocr = PlakaOCR(languages=['tur', 'eng'], min_conf=20)
    print(f"\n✓ OCR Hazır (Tesseract dili: {ocr.lang})")
    print(f"✓ EasyOCR Yüklü: {ocr.easyocr_reader is not None}")
    
    # Sentetik test görüntüsü oluştur (33 ASA 608)
    print("\n" + "=" * 60)
    print("TEST: Sentetik Plaka Görüntüsü (33 ASA 608)")
    print("=" * 60)
    
    # Beyaz arka plan
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    
    # Mavi şerit (Türkiye plakaları için)
    img[10:40, 10:60] = [0, 51, 153]  # Koyu mavi
    
    # Sarı arka plan (plaka bölgesi)
    img[10:90, 60:290] = [0, 255, 255]  # Sarı
    
    # Siyah yazı
    cv2.putText(img, "33 ASA 608", (70, 60), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 0), 3)
    
    # Görüntüyü kaydet
    cv2.imwrite('/tmp/test_plate.png', img)
    print("Test görüntüsü oluşturuldu: /tmp/test_plate.png")
    
    # OCR ile oku
    result = ocr.read_plate(img)
    
    print(f"\nSonuç:")
    print(f"  Başarılı: {result['success']}")
    print(f"  Okunan Metin: {result['text']}")
    print(f"  Güven Skoru: {result['confidence']:.2%}")
    
    if result['success']:
        expected = "33 ASA 608"
        if result['text'].replace(' ', '') == expected.replace(' ', ''):
            print(f"\n✅ BAŞARILI! Doğru plaka: {expected}")
        else:
            print(f"\n⚠️  UYARI! Beklenen: {expected}, Okunan: {result['text']}")
    
    print("\n" + "=" * 60)
    print("TEST TAMAMLANDI")
    print("=" * 60)

if __name__ == "__main__":
    test_plate_recognition()
