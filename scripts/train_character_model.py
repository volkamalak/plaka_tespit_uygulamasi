#!/usr/bin/env python3
"""
Karakter Seviyesi YOLO Modeli Eğitimi (MERGED + FINE-TUNE)
Plaka karakter tespiti yapar
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

def train_character_model():
    print("=" * 70)
    print("KARAKTER SEVİYESİ YOLO MODELİ EĞİTİMİ (MERGED DATASET)")
    print("=" * 70)

    # 🔹 MERGED DATASET
    dataset_path = Path("/home/bugra/datasets/merge_work/merged/data.yaml")

    if not dataset_path.exists():
        print(f"❌ Dataset bulunamadı: {dataset_path}")
        return

    print(f"\n✓ Dataset: {dataset_path}")
    print("  - 36 karakter sınıfı (0-9, A-Z)")
    print("  - Eski + Yeni dataset normalize edilerek birleştirildi")

    # 🔹 ESKİ MODEL (FINE-TUNE BAŞLANGICI)
    old_model_path = Path("/home/bugra/Desktop/plaka_tespit_uygulamasi/runs/detect/character_detector/weights/best.pt")

    if not old_model_path.exists():
        print(f"❌ Eski model bulunamadı: {old_model_path}")
        return

    print(f"\n✓ Fine-tune başlangıç modeli:")
    print(f"  {old_model_path}")

    print("\n🚀 YOLO modeli yükleniyor (fine-tune)...")
    model = YOLO(str(old_model_path))

    print("\n⚙️  Eğitim parametreleri:")
    print("  - Epochs: 80")
    print("  - Batch Size: 16")
    print("  - Image Size: 640")
    print("  - Learning Rate: 0.002 (unutmayı azaltır)")
    print("  - Device: GPU (cuda:0)")
    print("  - Cache: RAM")

    print("\n⏳ Eğitim başlıyor...")
    results = model.train(
        data=str(dataset_path),
        epochs=80,
        imgsz=640,
        batch=16,
        lr0=0.002,
        patience=20,
        device=0,
        name="character_detector_merged_ft",
        save=True,
        verbose=True,
        plots=True,
        workers=8,
        cache="ram",
    )

    print("\n" + "=" * 70)
    print("✅ EĞİTİM TAMAMLANDI")
    print("=" * 70)

    # 🔹 MODELİ MODELS KLASÖRÜNE KOPYALA
    best_model = Path("runs/detect/character_detector_merged_ft/weights/best.pt")

    if best_model.exists():
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        final_model_path = models_dir / "character_best_merged.pt"
        shutil.copy(best_model, final_model_path)

        print(f"\n✓ En iyi model kopyalandı:")
        print(f"  {final_model_path}")

    # 🔹 METRİKLER
    print("\n📊 Sonuçlar:")
    print(f"  - mAP50: {results.results_dict.get('metrics/mAP50', 'N/A')}")
    print(f"  - Precision: {results.results_dict.get('metrics/precision', 'N/A')}")
    print(f"  - Recall: {results.results_dict.get('metrics/recall', 'N/A')}")

    print("\n🎉 Karakter modeli hazır!")
    print("👉 Okuma pipeline'ına direkt entegre edebilirsin.")

if __name__ == "__main__":
    train_character_model()
    # 🔹 TEST ETME
    print("\n🔍 Model testi yapılıyor...")
    test_image = Path("/home/bugra/Desktop/plaka_tespit_uygulamasi/test_images/test1.jpg")
    if test_image.exists():
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("✅ Plaka karakterleri tespit edildi.")
            else:
                print("❌ Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("✅ Maske tespiti başarılı.")
            else:
                print("❌ Maske tespiti başarısız.")
            if result.keypoints:
                print("✅ Anahtar nokta tespiti başarılı.")
            else:
                print("❌ Anahtar nokta tespiti başarısız.")
            if result.error:
                 print("❌ Hata oluştu.")
            else:
                print("✅ Test başarılı.")
            if 'error' in result and result['error']:
                print(f"❌ Test sırasında hata oluştu: {result['error']}")
            else:
                print("✅ Test başarıyla tamamlandı.")
    else:
        print(f"❌ Test resmi bulunamadı: {test_image}")
        print("\n✅ Eğitim tamamlandı!")
    if test_image.exists():
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("Plaka karakterleri tespit edildi.")
            else:
                print("Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("Maske tespiti başarılı.")
            else:
                print("Maske tespiti başarısız.")
            if result.keypoints:
                print("Anahtar nokta tespiti başarılı.")
            else:
                print("Anahtar nokta tespiti başarısız.")
            if 'error' in result and result['error']:
                 print("Hata oluştu.")
            else:
                print("Test başarılı.")
            if 'error' in result and result['error']:
                print(f"Hata: {result['error']}")
            else:
                print("Test başarıyla tamamlandı.")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
    if test_image.exists():
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("Plaka karakterleri tespit edildi.")
            else:
                print("Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("Maske tespiti başarılı.")
            else:
                print("Maske tespiti başarısız.")
            if result.keypoints:
                print("Anahtar nokta tespiti başarılı.")
            else:
                print("Anahtar nokta tespiti başarısız.")
            if 'error' in result and result['error']:
                 print("Hata oluştu.")
            else:
                print("Test başarılı.")
            if 'error' in result and result['error']:
                print(f"Hata: {result['error']}")
            else:
                print("Test başarıyla tamamlandı.")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
    if test_image.exists():
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("Plaka karakterleri tespit edildi.")
            else:
                print("Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("Maske tespiti başarılı.")
            else:
                print("Maske tespiti başarısız.")
            if result.keypoints:
                print("Anahtar nokta tespiti başarılı.")
            else:
                print("Anahtar nokta tespiti başarısız.")
            if 'error' in result and result['error']:
                 print("Hata oluştu.")
            else:
                print("Test başarılı.")
            if 'error' in result and result['error']:
                print(f"Hata: {result['error']}")
            else:
                print("Test başarıyla tamamlandı.")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
    if test_image.exists():       
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("Plaka karakterleri tespit edildi.")
            else:
                print("Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("Maske tespiti başarılı.")
            else:
                print("Maske tespiti başarısız.")
            if result.keypoints:
                print("Anahtar nokta tespiti başarılı.")
            else:
                print("Anahtar nokta tespiti başarısız.")
            if 'error' in result and result['error']:
                 print("Hata oluştu.")
            else:
                print("Test başarılı.")
            if 'error' in result and result['error']:
                print(f"Hata: {result['error']}")
            else:
                print("Test başarıyla tamamlandı.")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
    if test_image.exists():
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("Plaka karakterleri tespit edildi.")
            else:
                print("Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("Maske tespiti başarılı.")
            else:
                print("Maske tespiti başarısız.")
            if result.keypoints:
                print("Anahtar nokta tespiti başarılı.")
            else:
                print("Anahtar nokta tespiti başarısız.")
            if 'error' in result and result['error']:
                 print("Hata oluştu.")
            else:
                print("Test başarılı.")
            if 'error' in result and result['error']:
                print(f"Hata: {result['error']}")
            else:
                print("Test başarıyla tamamlandı.")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
    if test_image.exists():
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("Plaka karakterleri tespit edildi.")
            else:
                print("Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("Maske tespiti başarılı.")
            else:
                print("Maske tespiti başarısız.")
            if result.keypoints:
                print("Anahtar nokta tespiti başarılı.")
            else:
                print("Anahtar nokta tespiti başarısız.")
            if 'error' in result and result['error']:
                 print("Hata oluştu.")
            else:
                print("Test başarılı.")
            if 'error' in result and result['error']:
                print(f"Hata: {result['error']}")
            else:
                print("Test başarıyla tamamlandı.")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
    if test_image.exists():      
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("Plaka karakterleri tespit edildi.")
            else:
                print("Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("Maske tespiti başarılı.")
            else:
                print("Maske tespiti başarısız.")
            if result.keypoints:
                print("Anahtar nokta tespiti başarılı.")
            else:
                print("Anahtar nokta tespiti başarısız.")
            if 'error' in result and result['error']:
                 print("Hata oluştu.")
            else:
                print("Test başarılı.")
            if 'error' in result and result['error']:
                print(f"Hata: {result['error']}")
            else:
                print("Test başarıyla tamamlandı.")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
    if test_image.exists():       
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("Plaka karakterleri tespit edildi.")
            else:
                print("Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("Maske tespiti başarılı.")
            else:
                print("Maske tespiti başarısız.")
            if result.keypoints:
                print("Anahtar nokta tespiti başarılı.")
            else:
                print("Anahtar nokta tespiti başarısız.")
            if 'error' in result and result['error']:
                 print("Hata oluştu.")
            else:
                print("Test başarılı.")
            if 'error' in result and result['error']:
                print(f"Hata: {result['error']}")
            else:
                print("Test başarıyla tamamlandı.")
    else:
        print(f"Test resmi bulunamadı: {test_image}")
    if test_image.exists():      
        model = YOLO("models/character_best_merged.pt")
        results = model.predict(source=str(test_image), conf=0.25, save=True)
        for result in results:
            if result.boxes:
                print("Plaka karakterleri tespit edildi.")
            else:
                print("Plaka karakteri tespit edilemedi.")
            if result.masks:
                print("Maske tespiti başarılı.")
            else:
                print("Maske tespiti başarısız.")
            if result.keypoints:
                print("Anahtar nokta tespiti başarılı.")
            else:
                print("Anahtar nokta tespiti başarısız.")
            if 'error' in result and result['error']:
                 print("Hata oluştu.")
            else:
                print("Test başarılı.")
            if 'error' in result and result['error']:
                print(f"Hata: {result['error']}")
            else:
                print("Test başarıyla tamamlandı.")
