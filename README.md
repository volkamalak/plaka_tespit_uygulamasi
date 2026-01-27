# Plaka Tespit Uygulaması

Python ve YOLO kullanarak resimlerde plaka tespiti yapan masaüstü uygulaması.

## Özellikler

- Resimden otomatik plaka tespiti
- Tespit edilen plakanın koordinatlarını gösterme
- İşlem süresini ölçme
- Görsel arayüz ile kolay kullanım
- İşlenmiş resimleri kaydetme

## Gereksinimler

- Python 3.8 veya üzeri
- YOLO modeli (best.pt)

## Kurulum

### 1. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2. YOLO Modelini Yerleştirin

Eğittiğiniz `best.pt` modelini `models/` klasörüne yerleştirin:

```
plaka_tespit_uygulamasi/
├── models/
│   └── best.pt          <- Modelinizi buraya koyun
├── images/              <- Test resimleri için
├── results/             <- Sonuçlar için
├── detector.py
├── main.py
└── requirements.txt
```

### 3. Model Eğitimi (Roboflow ile)

1. [Roboflow](https://roboflow.com/) hesabınıza giriş yapın
2. Yeni bir proje oluşturun ve plaka görsellerinizi yükleyin
3. Görselleri etiketleyin (bounding box ile)
4. Dataset'i dışa aktarın (YOLO formatında)
5. Ultralytics YOLO ile modeli eğitin:

```bash
from ultralytics import YOLO

# Model oluştur
model = YOLO('yolov8n.pt')  # veya 'yolov11n.pt'

# Eğitim
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# En iyi model 'runs/detect/train/weights/best.pt' konumunda olacak
```

6. `best.pt` dosyasını `models/` klasörüne kopyalayın

## Kullanım

### Uygulamayı Başlatma

```bash
python main.py
```

### Adım Adım Kullanım

1. **Resim Yükle**: "Resim Yükle" butonuna tıklayarak test edeceğiniz resmi seçin
2. **Çalıştır**: "Çalıştır" butonuna basarak plaka tespitini başlatın
3. **Sonuçları Görüntüle**:
   - Sol tarafta orijinal resim
   - Sağ tarafta tespit edilen plaka çerçeve içinde
   - Alt kısımda koordinatlar ve işlem süresi
4. **Kaydet**: İsterseniz işlenmiş resmi "Sonucu Kaydet" butonu ile kaydedin

## Proje Yapısı

```
plaka_tespit_uygulamasi/
│
├── main.py              # Ana uygulama (GUI)
├── detector.py          # YOLO plaka tespit modülü
├── requirements.txt     # Python kütüphaneleri
├── README.md           # Bu dosya
│
├── models/             # YOLO modelleri
│   └── best.pt        # Eğitilmiş model
│
├── images/            # Test resimleri
│   └── (test görselleri)
│
└── results/           # İşlenmiş resimler
    └── (sonuç görselleri)
```

## Teknik Detaylar

### Kullanılan Teknolojiler

- **Ultralytics YOLO**: Plaka tespiti için
- **OpenCV**: Görüntü işleme
- **Tkinter**: Grafik arayüz
- **Pillow**: Resim gösterimi

### detector.py

`PlakaDetector` sınıfı:
- `load_model()`: YOLO modelini yükler
- `detect_plate()`: Plaka tespiti yapar
- `save_result()`: İşlenmiş resmi kaydeder

### main.py

`PlakaTespitUygulamasi` sınıfı:
- İki bölmeli görsel arayüz
- Resim yükleme ve gösterme
- Tespit sonuçlarını görselleştirme
- Koordinat ve süre bilgilerini gösterme

## Özelleştirme

### Güven Eşiği Ayarlama

`detector.py` dosyasında:

```python
result = detector.detect_plate(image_path, conf_threshold=0.25)
```

`conf_threshold` parametresini değiştirerek güven eşiğini ayarlayabilirsiniz (0-1 arası).

### Çerçeve Rengi Değiştirme

`detector.py` dosyasında çizim kısmını düzenleyin:

```python
# Yeşil çerçeve (varsayılan)
cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Kırmızı çerçeve için
cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Mavi çerçeve için
cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
```

## Sorun Giderme

### "Model Bulunamadı" Hatası

- `models/best.pt` dosyasının var olduğundan emin olun
- Dosya yolunun doğru olduğunu kontrol edin

### Plaka Tespit Edilemiyor

- Model eğitiminin yeterli olup olmadığını kontrol edin
- Güven eşiğini düşürmeyi deneyin (`conf_threshold`)
- Test resminizin kalitesini kontrol edin

### Performans Sorunları

- GPU kullanımı için CUDA kurulumunu yapın
- Daha küçük bir YOLO modeli deneyin (yolov8n, yolov11n)
- Resim boyutunu küçültün

## Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## İletişim

Sorularınız için proje sahibi ile iletişime geçin.

---

**Not**: Uygulamayı kullanmadan önce `best.pt` modelini `models/` klasörüne yerleştirmeyi unutmayın!
