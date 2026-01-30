#!/usr/bin/env python3
"""
Plaka Doğrulama Mekanizması - Testler
"""

import cv2
from pathlib import Path
from plaka.ocr import PlakaOCR

image_path = Path("/home/bugra/Desktop/foto/vlcsnap-2026-01-28-16h47m19s390.png")

if not image_path.exists():
    print(f"❌ Resim bulunamadı: {image_path}")
    exit(1)

img = cv2.imread(str(image_path))
print(f"✓ Resim yüklendi: {img.shape}")

ocr = PlakaOCR(languages=['tur', 'eng'], min_conf=10)
print(f"✓ OCR hazır")

# Detektör yükle
from plaka.detector import PlakaDetector
detector = PlakaDetector(use_ocr=False)

# Önce YOLO ile tespit
result = detector.detect_plate(image_path, read_text=False)

if result['success'] and result['coordinates']:
    print(f"\n✓ {len(result['coordinates'])} plaka tespit edildi")
    
    for i, (x1, y1, x2, y2) in enumerate(result['coordinates']):
        print(f"\n{'='*60}")
        print(f"Plaka {i+1}: ({x1}, {y1}) - ({x2}, {y2})")
        print(f"{'='*60}")
        
        # Plaka bölgesini çıkar
        plate_crop = img[y1:y2, x1:x2]
        
        # OCR ile oku
        ocr_result = ocr.read_plate(plate_crop)
        
        print(f"✓ OCR Sonucu: '{ocr_result['text']}'")
        print(f"  Güven: {ocr_result['confidence']:.1%}")
        
        # Karakter analizi
        if ocr_result['text']:
            text = ocr_result['text'].replace(' ', '')
            print(f"\n🔤 Karakterler: {' '.join(text)}")
            for j, ch in enumerate(text):
                print(f"   {j}: '{ch}'")
else:
    print(f"❌ Plaka tespit edilemedi")
    if result['error']:
        print(f"   Hata: {result['error']}")
