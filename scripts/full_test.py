#!/usr/bin/env python3
"""
Full Pipeline Test - YOLO + Model Okuma
"""

import cv2
from plaka.detector import PlakaDetector

image_path = "/home/bugra/Desktop/foto/vlcsnap-2026-01-28-16h47m19s390.png"

print("🔍 Detektör başlıyor...")
detector = PlakaDetector()

print(f"📸 Resim işleniyor: {image_path}")
result = detector.detect_plate(image_path, read_text=True, conf_threshold=0.3)

print("\n" + "="*70)
if result['success']:
    print(f"✅ {len(result['coordinates'])} plaka tespit edildi")
    print(f"⏱️  Süre: {result['processing_time']:.2f}s")
    
    for i, (coords, conf, text, text_conf) in enumerate(zip(
        result['coordinates'],
        result['confidence'],
        result['plate_texts'],
        result['text_confidence']
    )):
        print(f"\n📌 Plaka {i+1}:")
        print(f"   Konum: {coords}")
        print(f"   YOLO: {conf:.1%}")
        print(f"   Okunan: {text}")
        print(f"   Model Güveni: {text_conf:.1%}")
else:
    print(f"❌ Hata: {result['error']}")

print("="*70)
