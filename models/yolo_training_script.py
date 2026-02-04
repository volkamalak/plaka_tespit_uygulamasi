"""
SANKO Port - YOLO Plaka Tespit Modeli Eğitimi
License Plate Detection Training Script
"""

from ultralytics import YOLO
import torch
import yaml
import os
from pathlib import Path
import shutil

print("="*70)
print("SANKO PORT - YOLO PLAKA TESPİT MODELİ EĞİTİMİ")
print("="*70)

# GPU Kontrolü
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✓ Kullanılan cihaz: {device}")
if device == 'cuda':
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠ GPU bulunamadı, CPU ile eğitim yapılacak (daha yavaş)")

# data.yaml dosyasını kontrol et ve oku
yaml_path = 'data.yaml'
if not os.path.exists(yaml_path):
    print(f"\n❌ HATA: '{yaml_path}' dosyası bulunamadı!")
    exit(1)

with open(yaml_path, 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)

print(f"\n✓ Dataset konfigürasyonu yüklendi:")
print(f"  - Sınıf sayısı: {data_config.get('nc')}")
print(f"  - Sınıf: {data_config.get('names')}")
print(f"  - Train: {data_config.get('train')}")
print(f"  - Valid: {data_config.get('val')}")
print(f"  - Test: {data_config.get('test')}")

# ==================== EĞİTİM PARAMETRELERİ ====================

# Model boyutu seçimi (Plaka tespiti için öneriler)
# 'n' = nano    - Çok hızlı, mobil/embedded cihazlar için (mAP ~35-40%)
# 's' = small   - Hızlı, iyi performans dengesi (mAP ~40-45%) ✓ ÖNERİLEN
# 'm' = medium  - Daha yüksek doğruluk (mAP ~45-50%)
# 'l' = large   - Çok iyi doğruluk ama yavaş (mAP ~50-55%)
# 'x' = xlarge  - En iyi doğruluk ama çok yavaş (mAP ~55%+)

MODEL_SIZE = 's'  # Plaka tespiti için 's' optimal

# Eğitim hiperparametreleri
EPOCHS = 150              # Epoch sayısı (plaka için 100-200 yeterli)
IMG_SIZE = 640            # Görüntü boyutu (640 standart, 320/416/512/640/1280)
BATCH_SIZE = 16           # Batch size (GPU belleğinize göre 8/16/32)
PATIENCE = 50             # Early stopping (50 epoch iyileşme yoksa dur)
SAVE_PERIOD = 10          # Her 10 epoch'ta checkpoint kaydet

# Optimizer ayarları
LEARNING_RATE = 0.01      # Başlangıç learning rate
FINAL_LR = 0.001          # Final learning rate
MOMENTUM = 0.937          # SGD momentum
WEIGHT_DECAY = 0.0005     # L2 regularization

# Data augmentation (Plaka için optimize edilmiş)
AUGMENTATION = {
    'hsv_h': 0.015,       # Hue (renk tonu) değişimi
    'hsv_s': 0.7,         # Saturation (doygunluk) değişimi
    'hsv_v': 0.4,         # Value (parlaklık) değişimi
    'degrees': 5.0,       # Rotasyon (plakalar için düşük)
    'translate': 0.1,     # Kaydırma
    'scale': 0.5,         # Ölçekleme
    'shear': 0.0,         # Yamultma (plaka için 0)
    'perspective': 0.0,   # Perspektif (plaka için 0)
    'flipud': 0.0,        # Dikey çevirme (plaka için KAPALI)
    'fliplr': 0.5,        # Yatay çevirme (plaka için aktif)
    'mosaic': 1.0,        # Mosaic augmentation
    'mixup': 0.0,         # Mixup augmentation (plaka için kapalı)
    'copy_paste': 0.0,    # Copy-paste augmentation
}

print("\n" + "="*70)
print("EĞİTİM AYARLARI:")
print("="*70)
print(f"Model: YOLOv8{MODEL_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE} → {FINAL_LR}")
print(f"Device: {device}")
print(f"Patience (Early Stop): {PATIENCE} epochs")

# ==================== MODEL YÜKLEME ====================

print("\n" + "="*70)
print("MODEL YÜKLEME")
print("="*70)

# Pretrained YOLOv8 modelini yükle
model = YOLO(f'yolov8{MODEL_SIZE}.pt')
print(f"✓ YOLOv8{MODEL_SIZE} pretrained modeli yüklendi")

# ==================== MODEL EĞİTİMİ ====================

print("\n" + "="*70)
print("EĞİTİM BAŞLIYOR...")
print("="*70)

try:
    results = model.train(
        # Dataset
        data=yaml_path,
        
        # Temel parametreler
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        
        # Optimizer
        optimizer='auto',  # SGD, Adam, AdamW otomatik seçim
        lr0=LEARNING_RATE,
        lrf=FINAL_LR / LEARNING_RATE,  # Final lr çarpanı
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        
        # Data Augmentation
        **AUGMENTATION,
        
        # Kaydetme ayarları
        save=True,
        save_period=SAVE_PERIOD,
        project='runs/detect',
        name='plaka_detection',
        exist_ok=True,
        
        # Diğer
        pretrained=True,
        verbose=True,
        patience=PATIENCE,
        workers=8,
        seed=42,  # Tekrarlanabilirlik için
        
        # Val ayarları
        val=True,
        plots=True,  # Grafikleri kaydet
        
        # Multi-scale training (opsiyonel, daha yavaş ama daha robust)
        # rect=False,  # Rectangular training
        # close_mosaic=10,  # Son 10 epoch'ta mosaic kapat
    )
    
    print("\n" + "="*70)
    print("✓ EĞİTİM TAMAMLANDI!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ EĞİTİM HATASI: {e}")
    exit(1)

# ==================== VALİDASYON ====================

print("\n" + "="*70)
print("VALİDASYON")
print("="*70)

try:
    val_results = model.val()
    
    print(f"\nValidasyon Metrikleri:")
    print(f"  - mAP50: {val_results.box.map50:.4f}")
    print(f"  - mAP50-95: {val_results.box.map:.4f}")
    print(f"  - Precision: {val_results.box.mp:.4f}")
    print(f"  - Recall: {val_results.box.mr:.4f}")
    
except Exception as e:
    print(f"⚠ Validasyon hatası: {e}")

# ==================== TEST ====================

if data_config.get('test'):
    print("\n" + "="*70)
    print("TEST SETİ DEĞERLENDİRMESİ")
    print("="*70)
    
    try:
        # En iyi modeli yükle ve test et
        best_model = YOLO(model.trainer.best)
        test_results = best_model.val(data=yaml_path, split='test')
        
        print(f"\nTest Metrikleri:")
        print(f"  - mAP50: {test_results.box.map50:.4f}")
        print(f"  - mAP50-95: {test_results.box.map:.4f}")
        print(f"  - Precision: {test_results.box.mp:.4f}")
        print(f"  - Recall: {test_results.box.mr:.4f}")
        
    except Exception as e:
        print(f"⚠ Test hatası: {e}")

# ==================== MODEL KAYDETME ====================

print("\n" + "="*70)
print("MODEL KAYDETME")
print("="*70)

# En iyi modelin yolunu al
best_model_path = model.trainer.best
last_model_path = model.trainer.last

print(f"✓ En iyi model: {best_model_path}")
print(f"✓ Son model: {last_model_path}")

# Modeli kullanıcı dostu bir isimle kopyala
output_path = 'plaka_tespit_model.pt'
if os.path.exists(best_model_path):
    shutil.copy(best_model_path, output_path)
    print(f"\n✓ Model '{output_path}' olarak kaydedildi!")
    
    # Dosya boyutunu göster
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  - Dosya boyutu: {file_size:.2f} MB")
else:
    print(f"⚠ Model dosyası bulunamadı: {best_model_path}")

# ==================== SONUÇLAR VE KULLANIM ====================

print("\n" + "="*70)
print("EĞİTİM RAPORU")
print("="*70)

print(f"""
✓ Eğitim tamamlandı!
✓ Model dosyası: {output_path}
✓ Eğitim sonuçları: runs/detect/plaka_detection/

📊 Sonuç Dosyaları:
  - results.png: Eğitim grafikleri
  - confusion_matrix.png: Confusion matrix
  - val_batch0_pred.jpg: Örnek tahminler
  - weights/best.pt: En iyi model
  - weights/last.pt: Son model

🔍 Grafikler ve detaylı sonuçlar için:
  runs/detect/plaka_detection/ klasörüne bakın
""")

print("\n" + "="*70)
print("MODEL KULLANIMI")
print("="*70)

print(f"""
# 1. Modeli yükle
from ultralytics import YOLO
model = YOLO('{output_path}')

# 2. Tek görüntü üzerinde tahmin
results = model('test_image.jpg')
for r in results:
    boxes = r.boxes  # Bounding box bilgileri
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Koordinatlar
        conf = box.conf[0]  # Güven skoru
        cls = box.cls[0]  # Sınıf (0: License_Plate)
        print(f"Plaka bulundu! Güven: {{conf:.2f}}")

# 3. Sonuçları görselleştir ve kaydet
results[0].show()  # Göster
results[0].save('sonuc.jpg')  # Kaydet

# 4. Toplu tahmin (birden fazla görüntü)
results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# 5. Video üzerinde çalıştır
results = model('video.mp4', save=True, conf=0.5)

# 6. Webcam'den canlı tespit
results = model(source=0, show=True)  # 0 = webcam

# 7. Klasördeki tüm görselleri işle
results = model('path/to/images/', save=True)

# 8. Güven eşiği ayarlama
results = model('image.jpg', conf=0.5)  # Sadece %50+ güvenli tespitler

# 9. NMS (Non-Maximum Suppression) eşiği
results = model('image.jpg', iou=0.5)

# 10. Daha hızlı inference için
model.fuse()  # Conv + BN katmanlarını birleştir
results = model('image.jpg', half=True)  # FP16 inference
""")

print("\n" + "="*70)
print("İPUÇLARI")
print("="*70)

print("""
🎯 Model Performansını Artırmak İçin:

1. Daha fazla veri ekleyin (özellikle zor örnekler)
2. Epoch sayısını artırın (200-300)
3. Daha büyük model kullanın (yolov8m veya yolov8l)
4. Image size'ı artırın (1280)
5. Data augmentation'ı ayarlayın
6. Hyperparameter tuning yapın

📱 Mobil/Edge Cihazlar İçin:
- YOLOv8n kullanın (en hızlı)
- Image size'ı düşürün (320 veya 416)
- Export edin: model.export(format='onnx') veya 'tflite'

🚀 Üretim İçin:
- Model'i export edin (ONNX, TensorRT, CoreML)
- Inference pipeline optimize edin
- Batch processing kullanın
- GPU inference için TensorRT kullanın

⚠️ Sorun Giderme:
- GPU memory hatası: BATCH_SIZE'ı azaltın
- Overfitting: Weight decay artırın, data augmentation artırın
- Underfitting: Model boyutunu artırın, epoch artırın
- mAP düşük: Daha fazla veri, daha uzun eğitim
""")

print("\n" + "="*70)
print("✓ Tüm işlemler tamamlandı!")
print("="*70)
