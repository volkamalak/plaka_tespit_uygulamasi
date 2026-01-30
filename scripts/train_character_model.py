#!/usr/bin/env python3
"""
Karakter Seviyesi YOLO Modeli Eğitimi
Plaka çıkıntılarında karakter tespiti yapar
"""

from ultralytics import YOLO
from pathlib import Path

def train_character_model():
    """Karakter tespit modeli eğit"""
    
    print("=" * 70)
    print("KARAKTER SEVİYESİ YOLO MODELİ EĞİTİMİ")
    print("=" * 70)
    
    # Dataset yolu
    dataset_path = Path("datasets/characters_new/plaka.v1i.yolov8/data.yaml")
    
    if not dataset_path.exists():
        print(f"❌ Dataset bulunamadı: {dataset_path}")
        return
    
    print(f"\n✓ Dataset: {dataset_path}")
    print(f"  - 47 karakter sınıfı (0-9, A-Z, spesyal)")
    print(f"  - Train: datasets/characters/train/images")
    print(f"  - Val: datasets/characters/valid/images")
    print(f"  - Test: datasets/characters/test/images")
    
    # Yeni model oluştur
    print("\n🚀 YOLOv8 Nano modeli oluşturuluyor...")
    model = YOLO('yolov8n.pt')  # YOLOv8 Nano
    
    # Eğitim parametreleri
    print("\n⚙️  Eğitim parametreleri:")
    print("  - Epochs: 100")
    print("  - Batch Size: 16")
    print("  - Image Size: 640x640")
    print("  - Device: GPU (cuda:0)")
    print("  - Patience: 20")
    
    # Eğitim başlat
    print("\n⏳ Eğitim başlıyor... (bu uzun sürebilir)")
    results = model.train(
        data=str(dataset_path),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,  # GPU
        name='character_detector',
        save=True,
        verbose=True,
        plots=True,
        # Device-specific optimizations
        workers=8,
        cache='ram',  # RAM'de cache tut (hızlı)
    )
    
    print("\n" + "=" * 70)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    
    # Model konumu
    best_model = Path("runs/detect/character_detector/weights/best.pt")
    if best_model.exists():
        print(f"\n✓ En iyi model kaydedildi:")
        print(f"  {best_model}")
        
        # Kopyala
        import shutil
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        char_model_path = models_dir / "character_best.pt"
        shutil.copy(best_model, char_model_path)
        print(f"\n✓ Kopyalandı: {char_model_path}")
    
    print("\n📊 Sonuçlar:")
    print(f"  - mAP50: {results.results_dict.get('metrics/mAP50', 'N/A')}")
    print(f"  - Precision: {results.results_dict.get('metrics/precision', 'N/A')}")
    print(f"  - Recall: {results.results_dict.get('metrics/recall', 'N/A')}")
    
    print("\n🎉 Karakter modeli hazır! Şimdi OCR pipeline'ına entegre edebiliriz.")

if __name__ == "__main__":
    train_character_model()
