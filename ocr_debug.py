#!/usr/bin/env python3
"""
OCR Debug Modu - Detaylı Tanısı
"""

import cv2
import numpy as np
from pathlib import Path
from ocr import PlakaOCR

def debug_ocr(image_path):
    """OCR'ı debug mode'da çalıştırır."""
    
    print("=" * 70)
    print("OCR DEBUG MODU")
    print("=" * 70)
    
    # Resmi yükle
    if not Path(image_path).exists():
        print(f"❌ Resim bulunamadı: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Resim açılamadı: {image_path}")
        return
    
    print(f"\n📷 Resim Bilgisi:")
    print(f"   Dosya: {image_path}")
    print(f"   Boyut: {img.shape[0]}x{img.shape[1]} px")
    print(f"   Format: {img.shape[2]} kanallı" if len(img.shape) > 2 else "Gri tonlu")
    
    # OCR
    ocr = PlakaOCR(languages=['tur', 'eng'], min_conf=10)  # Çok düşük threshold
    
    print(f"\n🔍 OCR Konfigürasyonu:")
    print(f"   Tesseract Dili: {ocr.lang}")
    print(f"   EasyOCR: {'✅ Aktif' if ocr.easyocr_reader else '❌ Pasif'}")
    print(f"   Min Güven: {ocr.min_conf}%")
    
    # Okumayı yap
    print(f"\n⏳ OCR İşlemi başlıyor...")
    result = ocr.read_plate(img)
    
    print(f"\n📊 Sonuçlar:")
    print(f"   Başarı: {'✅ BAŞARILI' if result['success'] else '❌ BAŞARISIZ'}")
    print(f"   Okunan Metin: '{result['text']}'")
    print(f"   Güven Skoru: {result['confidence']:.1%}")
    if result['error']:
        print(f"   Hata: {result['error']}")
    
    # Doğru plakayı sor
    print(f"\n❓ Lütfen doğru plaka numarasını gir (örn: 34 ABC 1234):")
    correct = input("➜ ").strip().upper()
    
    if correct:
        if result['text'].replace(' ', '') == correct.replace(' ', ''):
            print(f"✅ DOĞRU! Sistem doğru okudu!")
        else:
            print(f"\n❌ YANLIŞ OKUMA:")
            print(f"   Beklenen: {correct}")
            print(f"   Okunan:   {result['text']}")
            
            # Karakter karşılaştırması
            print(f"\n🔤 Karakter Analizi:")
            exp_clean = correct.replace(' ', '')
            read_clean = result['text'].replace(' ', '')
            max_len = max(len(exp_clean), len(read_clean))
            
            for i in range(max_len):
                exp_ch = exp_clean[i] if i < len(exp_clean) else '?'
                read_ch = read_clean[i] if i < len(read_clean) else '?'
                match = "✅" if exp_ch == read_ch else "❌"
                print(f"   Pos {i}: {match} '{exp_ch}' → '{read_ch}'")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Kullanım: python ocr_debug.py <resim_dosyasi>")
        print("\nÖrnek: python ocr_debug.py plaka.jpg")
        print("\nOr eğer images/ klasöründe resim varsa:")
        images = list(Path("images").glob("*.jpg")) + list(Path("images").glob("*.png"))
        if images:
            print(f"\nBulunan resimler:")
            for img in images[:5]:
                print(f"  - {img}")
            image_path = str(images[0])
            print(f"\nİlk resim kullanılıyor: {image_path}")
        else:
            print("\n❌ Hiç resim bulunamadı!")
            sys.exit(1)
    
    debug_ocr(image_path)
