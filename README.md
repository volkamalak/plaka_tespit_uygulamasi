# SANKO Port

Python ve Ultralytics YOLO kullanarak görüntülerde plaka tespiti ve **model tabanlı** okuma yapan, Tkinter tabanlı masaüstü uygulaması.

## Özellikler

- 🚗 **YOLO tabanlı plaka tespiti** (eğitilmiş model ile)
- 🔤 **Character YOLO modeliyle okuma** (OCR yok)
- 🎥 **Video desteği**: Video yükleme, oynatma ve stabil okuma
- 📍 **Koordinat ve güven skorlarını gösterme**
- ⏱️ **İşlem süresi ölçümü**
- 💾 **Kırpılmış plaka görüntüsünü kaydetme**

## Gereksinimler

- Python 3.8+ (önerilen: 3.10+)
- (Opsiyonel) CUDA destekli GPU (daha hızlı tespit ve model okuma için)

Python bağımlılıkları `requirements.txt` içinde listelenir.

## Kurulum (Adım Adım - Tüm İşletim Sistemleri)

Kurulum adımları tüm sistemlerde aynıdır; sadece Python kurulumu ve paket yöneticisi farklıdır.

### 0) Projeyi açın

Terminali proje klasöründe açın:

```
SANKO Port/
```

### 1) Model dosyalarını yerleştirin

```
models/
├── best.pt             # Zorunlu: plaka tespit modeli
└── character_best.pt   # Opsiyonel: karakter (harf/rakam) okuma modeli
```

### 2) Windows (PowerShell / CMD)

1. **Python 3.8+ kurun**  
   - Kurulumda **“Add Python to PATH”** seçeneğini işaretleyin.
2. **Sanal ortam oluşturun ve etkinleştirin**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
```

3. **Bağımlılıkları yükleyin**

```powershell
pip install -r requirements.txt
```

### 3) macOS (Terminal)

1. **Python kurun** (Homebrew ile)

```bash
brew install python
```

2. **Sanal ortam oluşturun ve etkinleştirin**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Bağımlılıkları yükleyin**

```bash
pip install -r requirements.txt
```

### 4) Linux (Ubuntu / Debian)

1. **Sistem paketlerini kurun**

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-tk
```

2. **Sanal ortam oluşturun ve etkinleştirin**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Bağımlılıkları yükleyin**

```bash
pip install -r requirements.txt
```

### 5) Linux (Fedora)

1. **Sistem paketlerini kurun**

```bash
sudo dnf install -y python3 python3-pip python3-tkinter
```

2. **Sanal ortam oluşturun ve etkinleştirin**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Bağımlılıkları yükleyin**

```bash
pip install -r requirements.txt
```

### 6) Linux (Arch)

1. **Sistem paketlerini kurun**

```bash
sudo pacman -S --needed python python-pip tk
```

2. **Sanal ortam oluşturun ve etkinleştirin**

```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Bağımlılıkları yükleyin**

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main.py
```

Uygulama açıldığında:
- **Resim Yükle** ile görüntü seçin.
- **Plakayı Tespit Et** ile plaka bölgesini bulun.
- **Metni Oku** ile model okumasını çalıştırın.
- **Kaydet** ile kırpılmış plaka görüntüsünü diske yazın.

## Video Modu

1) **Video Yükle** ile bir video seçin.
2) **Oynat** ile akışı başlatın.
3) Uygulama belirli aralıklarla plaka tespiti yapar.
4) Stabil okuma yakalandığında sonuç paneli güncellenir.

> Video modunda okuma için "Model" anahtarı kullanılır.

## Model Dosyaları Hakkında

- `models/best.pt` **zorunludur**. Yoksa plaka tespiti yapılamaz.
- `models/character_best.pt` **opsiyoneldir**. Yoksa metin okuma yapılamaz.

Model yolunu değiştirmek için `plaka/detector.py` içindeki `model_path` ve `character_model_path` değerlerini güncelleyebilirsiniz.

## Kod ile Kullanım (Örnek)

```python
from plaka.detector import PlakaDetector

# Model ile tek resim
model = PlakaDetector(
    model_path="models/best.pt",
    use_character_model=True,
    character_model_path="models/character_best.pt",
)

result = model.detect_plate("/path/to/image.jpg", conf_threshold=0.25, read_text=True)
print(result["success"], result.get("plate_texts"))
```

## Yapılandırma İpuçları

- **Güven eşiği**: `detect_plate(..., conf_threshold=0.25)`
- **Model anahtarı**: GUI üzerindeki "Model" butonu ile aç/kapat
- **Video parametreleri**: `plaka/gui.py` içindeki `video_*` değişkenleri

## Çıktı ve Klasörler

- `results/` : Kaydedilen çıktılar ve debug dosyaları
- `results/debug_char/` : "Metni Oku" işlemi sırasında kaydedilen karakter debug görüntüleri

## Test ve Yardımcı Scriptler

`scripts/` klasörü altında hızlı test ve yardımcı komutlar bulunur.
Bazı scriptlerde sabit dosya yollarının güncellenmesi gerekir:
- `scripts/test_detector.py`
- `scripts/full_test.py`

## Sık Görülen Sorunlar

- **Model bulunamadı**: `models/best.pt` dosyasının var olduğundan emin olun.
- **Okuma yok**: `models/character_best.pt` yoksa metin okunamaz.

## Dokümantasyon

Daha ayrıntılı notlar için:
- `docs/KURULUM.md`
- `docs/GPU_OPTIMIZATION.md`
