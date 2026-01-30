#!/usr/bin/env python3
"""
Gerçekçi Türk Plakası Test Resimleri Oluşturucusu
"""

import cv2
import numpy as np
from pathlib import Path

def create_realistic_plate(plate_text, filename):
    """Gerçekçi Türk plakası resimsini oluştur."""
    
    # Plaka ölçüleri (gerçeğe yakın): 520x110 mm
    # Biz 400x85 kullanacağız
    width, height = 400, 85
    
    # Beyaz arka plan
    plate = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Mavi EU band (Türkiye plakaları - mavi, sarı yıldızlar, TR)
    eu_band_width = 55
    plate[:, :eu_band_width] = (0, 51, 153)  # Koyu mavi
    
    # EU text (TR)
    cv2.putText(plate, 'TR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Sarı arka plan (plaka numarası bölgesi) - türk plakalarında sarı band
    plate[65:85, :] = (0, 255, 255)
    
    # Siyah plaka numarası (büyük ve kalın)
    # Format: XX ABC 1234
    cv2.putText(plate, plate_text, (70, 55), 
                cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 3)
    
    # Kenarları belirginleştir (dışlı)
    cv2.rectangle(plate, (0, 0), (width-1, height-1), (0, 0, 0), 2)
    
    # Gürültü ekle (gerçekçi görünüm için)
    noise = np.random.normal(0, 5, plate.shape).astype(np.uint8)
    plate = cv2.add(plate, noise)
    
    # Kaydet
    cv2.imwrite(filename, plate)
    print(f"✓ Oluşturuldu: {filename}")
    return plate

if __name__ == "__main__":
    # Test klasörü oluştur
    Path("test_plates").mkdir(exist_ok=True)
    
    test_cases = [
        "33 ASA 608",  # Kullanıcının sorunu
        "34 ABC 1234",  # Standart
        "06 AZZ 99",    # Kısa
        "35 X 5678",    # 1 harf
        "01 TRM 2024",  # Ankara
    ]
    
    print("=" * 50)
    print("Türk Plakası Test Resimleri Oluşturuluyor")
    print("=" * 50 + "\n")
    
    for text in test_cases:
        filename = f"test_plates/{text.replace(' ', '_')}.jpg"
        create_realistic_plate(text, filename)
    
    print("\n✅ Tümü oluşturuldu!")
    print("\nÖrnek test komutu:")
    print("  python scripts/ocr_debug.py test_plates/33_ASA_608.jpg")
