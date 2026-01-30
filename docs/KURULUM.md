# Hızlı Başlangıç Kılavuzu

## 1. Adım: Python Kontrolü

Python'un yüklü olduğundan emin olun:

```bash
python --version
```

Python 3.8 veya üzeri gereklidir.

## 2. Adım: Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

### Kurulacak Kütüphaneler:
- `ultralytics` - YOLO modeli için
- `opencv-python` - Görüntü işleme için
- `pillow` - Resim gösterimi için
- `numpy` - Sayısal işlemler için

## 3. Adım: YOLO Modelini Hazırlayın

### Model Eğitimi (Roboflow + Ultralytics)

#### 3.1. Dataset Hazırlama (Roboflow)

1. [Roboflow](https://roboflow.com/) hesabı oluşturun
2. "New Project" ile yeni proje oluşturun
3. Plaka görsellerinizi yükleyin (en az 100-200 görsel önerilir)
4. Her görseldeki plakaları "Bounding Box" ile etiketleyin
5. Dataset'i bölün:
   - Train: %70
   - Valid: %20
   - Test: %10
6. "Generate" ile dataset'i oluşturun
7. "Export" ile YOLO formatında indirin

#### 3.2. Model Eğitimi

Eğitim scripti oluşturun (`train_model.py`):

```python
from ultralytics import YOLO

# Yeni bir model oluştur (YOLOv8 nano)
model = YOLO('yolov8n.pt')

# Eğitim parametreleri
results = model.train(
    data='path/to/data.yaml',  # Roboflow'dan indirilen data.yaml
    epochs=100,                 # Eğitim döngüsü sayısı
    imgsz=640,                  # Görüntü boyutu
    batch=16,                   # Batch boyutu
    name='plaka_detector',      # Proje adı
    patience=20,                # Erken durdurma sabır değeri
    save=True,                  # Model kaydetme
    device=0                    # GPU kullan (CPU için 'cpu')
)

print("Eğitim tamamlandı!")
print(f"En iyi model: runs/detect/plaka_detector/weights/best.pt")
```

Eğitimi çalıştırın:

```bash
python train_model.py
```

#### 3.3. Modeli Yerleştirin

Eğitim sonrası oluşan `best.pt` dosyasını kopyalayın:

```bash
# Linux/Mac
cp runs/detect/plaka_detector/weights/best.pt models/best.pt

# Windows
copy runs\detect\plaka_detector\weights\best.pt models\best.pt
```

## 4. Adım: Uygulamayı Çalıştırın

```bash
python main.py
```

## Alternatif: YOLOv11 Kullanımı

YOLOv11 daha yeni ve daha iyi performans sunabilir:

```python
from ultralytics import YOLO

# YOLOv11 nano modeli
model = YOLO('yolo11n.pt')

# Eğitim aynı şekilde
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## Test Etme

### Komut Satırından Test

```python
from plaka.detector import PlakaDetector

detector = PlakaDetector()
result = detector.detect_plate('test_resmi.jpg')

if result['success']:
    print(f"Tespit edilen plaka sayısı: {len(result['coordinates'])}")
    print(f"Koordinatlar: {result['coordinates']}")
    print(f"İşlem süresi: {result['processing_time']:.3f} saniye")
else:
    print(f"Hata: {result['error']}")
```

## Sorun Giderme

### "No module named 'ultralytics'" Hatası

```bash
pip install ultralytics
```

### CUDA/GPU Kullanımı

GPU kullanmak için CUDA kurulumu gereklidir:

1. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) indirin
2. PyTorch'u CUDA desteğiyle yükleyin:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Model Dosyası Bulunamıyor

`models/best.pt` dosyasının var olduğundan emin olun:

```bash
# Linux/Mac
ls -lh models/best.pt

# Windows
dir models\best.pt
```

## Proje Yapısı Kontrolü

Doğru yapıda olduğundan emin olun:

```
plaka_tespit_uygulamasi/
├── main.py
├── plaka/detector.py
├── requirements.txt
├── README.md
├── docs/KURULUM.md
├── models/
│   └── best.pt          ← Modeliniz burada olmalı
├── images/
│   └── .gitkeep
└── results/
    └── .gitkeep
```

## Başarıyla Kurulum Kontrolü

1. Model yüklendi mi? ✅
   - Uygulama açıldığında sağ üstte "✅ Model Yüklü" yazmalı

2. Resim yüklenebiliyor mu? ✅
   - "Resim Yükle" butonuna basarak test edin

3. Tespit çalışıyor mu? ✅
   - Test resmi yükleyip "Çalıştır" butonuna basın

## Yararlı Linkler

- [Ultralytics Dökümantasyon](https://docs.ultralytics.com/)
- [Roboflow](https://roboflow.com/)
- [YOLO GitHub](https://github.com/ultralytics/ultralytics)

---

Başarılı kullanımlar!
