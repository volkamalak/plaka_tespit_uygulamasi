#!/usr/bin/env python3
"""
Hızlı Test - Tesseract ile (GPU EasyOCR'ı atla)
"""

import cv2
import numpy as np
from pathlib import Path
from plaka.ocr import PlakaOCR

# Test 1: Sentetik plaka
print("="*60)
print("TEST 1: Sentetik Plaka")
print("="*60)

img = np.ones((85, 400, 3), dtype=np.uint8) * 255
img[:, :55] = (0, 51, 153)  # Mavi band
cv2.putText(img, '09 APF 547', (50, 55), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 3)
cv2.rectangle(img, (0, 0), (399, 84), (0, 0, 0), 2)

ocr = PlakaOCR(languages=['tur', 'eng'])
result = ocr.read_plate(img)

print(f"Beklenen: 09 APF 547")
print(f"Okunan:   {result['text']}")
print(f"Güven:    {result['confidence']:.1%}")

if result['text'].replace(' ', '') == '09APF547':
    print("✅ BAŞARILI!")
else:
    print("⚠️  HATA")

# Test 2: Gerçek resim (varsa)
test_image = Path("/home/bugra/Desktop/foto/vlcsnap-2026-01-28-16h47m19s390.png")
if test_image.exists():
    print("\n" + "="*60)
    print("TEST 2: Gerçek Resim")
    print("="*60)
    
    img = cv2.imread(str(test_image))
    result = ocr.read_plate(img)
    
    print(f"Beklenen: 09 APF 547")
    print(f"Okunan:   {result['text']}")
    print(f"Güven:    {result['confidence']:.1%}")
    
    if result['text'].replace(' ', '') == '09APF547':
        print("✅ BAŞARILI!")
    else:
        print("⚠️  HATA")
