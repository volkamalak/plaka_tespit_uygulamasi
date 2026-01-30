#!/usr/bin/env python3
"""
Plaka Detektörü Test - Koordinatlarla OCR
"""

import cv2
from plaka.detector import PlakaDetector

image_path = "/home/bugra/Desktop/foto/vlcsnap-2026-01-28-16h47m19s390.png"

print("🔍 Plaka detektörü başlıyor...")
detector = PlakaDetector(use_ocr=True)

result = detector.detect_plate(image_path, read_text=True)

print("\n" + "="*70)
print("SONUÇLAR")
print("="*70)

if result['success']:
    print(f"✅ Başarılı!")
    print(f"📍 Tespit edilen plaka sayısı: {len(result['coordinates'])}")
    print(f"⏱️ İşlem süresi: {result['processing_time']:.2f} saniye")
    
    for i, (coords, conf, text, ocr_conf) in enumerate(zip(
        result['coordinates'],
        result['confidence'],
        result['plate_texts'],
        result['ocr_confidence']
    )):
        print(f"\n📌 Plaka {i+1}:")
        print(f"   Koordinatlar: {coords}")
        print(f"   YOLO Güveni: {conf:.1%}")
        print(f"   Plaka Numarası: {text}")
        print(f"   OCR Güveni: {ocr_conf:.1%}")
else:
    print(f"❌ Başarısız: {result['error']}")

print("\n" + "="*70)
