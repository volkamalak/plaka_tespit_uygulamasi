# Plaka Tespit Uygulaması

Python ve Ultralytics YOLO kullanarak görüntülerde plaka tespiti ve okuma yapan, Tkinter tabanlı masaüstü uygulaması.

## Özellikler

- 🚗 **YOLO tabanlı plaka tespiti** (eğitilmiş model ile)
- 🔎 **İki okuma modu**: Character YOLO modeli + OCR (Tesseract + EasyOCR)
- 🎥 **Video desteği**: Video yükleme, oynatma ve stabil okuma
- 📍 **Koordinat ve güven skorlarını gösterme**
- ⏱️ **İşlem süresi ölçümü**
- 💾 **Kırpılmış plaka görüntüsünü kaydetme**
- 🇹🇷🇬🇧 **Türkçe ve İngilizce OCR desteği**

## Gereksinimler

- Python 3.8+ (önerilen: 3.10+)
- Tesseract OCR ikilisi (sistemde kurulu olmalı)
- (Opsiyonel) CUDA destekli GPU (daha hızlı tespit ve OCR için)

Python bağımlılıkları `requirements.txt` içinde listelenir.

## Kurulum (Adım Adım - Tüm İşletim Sistemleri)

Kurulum adımları tüm sistemlerde aynıdır; sadece Python/Tesseract kurulumu ve paket yöneticisi farklıdır.

### 0) Projeyi açın

Terminali proje klasöründe açın:

```
plaka_tespit_uygulamasi/
```

### 1) Model dosyalarını yerleştirin

```
models/
├── best.pt             # Zorunlu: plaka tespit modeli
└── character_best.pt   # Opsiyonel: karakter (harf/rakam) modeli
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

4. **Tesseract OCR kurun**
   - Kurulumdan sonra `C:\Program Files\Tesseract-OCR` klasörünü PATH'e ekleyin.
5. **Kurulumu doğrulayın**

```powershell
tesseract --version
```

### 3) macOS (Terminal)

1. **Python ve Tesseract kurun** (Homebrew ile)

```bash
brew install python tesseract
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

4. **Kurulumu doğrulayın**

```bash
tesseract --version
```

### 4) Linux (Ubuntu / Debian)

1. **Sistem paketlerini kurun**

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-tk tesseract-ocr
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

4. **Kurulumu doğrulayın**

```bash
tesseract --version
```

### 5) Linux (Fedora)

1. **Sistem paketlerini kurun**

```bash
sudo dnf install -y python3 python3-pip python3-tkinter tesseract
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

4. **Kurulumu doğrulayın**

```bash
tesseract --version
```

### 6) Linux (Arch)

1. **Sistem paketlerini kurun**

```bash
sudo pacman -S --needed python python-pip tk tesseract
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

4. **Kurulumu doğrulayın**

```bash
tesseract --version
```

> GPU ile hızlandırma için `docs/GPU_OPTIMIZATION.md` dosyasına göz atın.

## Çalıştırma

```bash
python main.py
```

Uygulama açıldığında:
- **Resim Yükle** ile görüntü seçin.
- **Plakayı Tespit Et** ile plaka bölgesini bulun.
- **Metni Oku** ile modeli ve/veya OCR'yi çalıştırın.
- **Kaydet** ile kırpılmış plaka görüntüsünü diske yazın.

## Video Modu

1) **Video Yükle** ile bir video seçin.
2) **Oynat** ile akışı başlatın.
3) Uygulama belirli aralıklarla plaka tespiti yapar.
4) Stabil okuma yakalandığında sonuç paneli güncellenir.

> Video modunda okuma için "Model" ve/veya "OCR" anahtarları kullanılır.

## Model Dosyaları Hakkında

- `models/best.pt` **zorunludur**. Yoksa plaka tespiti yapılamaz.
- `models/character_best.pt` **opsiyoneldir**. Yoksa karakter modeli devre dışı kalır; OCR yine çalışır.

Model yolunu değiştirmek için `plaka/detector.py` içindeki `model_path` ve `character_model_path` değerlerini güncelleyebilirsiniz.

## Kod ile Kullanım (Örnek)

```python
from plaka.detector import PlakaDetector

# Model + OCR ile tek resim
model = PlakaDetector(
    model_path="models/best.pt",
    use_ocr=True,
    ocr_languages=["tur", "eng"],
    use_character_model=True,
    character_model_path="models/character_best.pt",
)

result = model.detect_plate("/path/to/image.jpg", conf_threshold=0.25, read_text=True)
print(result["success"], result.get("plate_texts"))
```

## Yapılandırma İpuçları

- **Güven eşiği**: `detect_plate(..., conf_threshold=0.25)`
- **OCR dilleri**: `PlakaDetector(ocr_languages=["tur", "eng"])`
- **Model/OCR anahtarları**: GUI üzerindeki "Model" ve "OCR" butonları ile aç/kapat
- **Video parametreleri**: `plaka/gui.py` içindeki `video_*` değişkenleri

## Çıktı ve Klasörler

- `results/` : Kaydedilen çıktılar ve debug dosyaları
- `results/debug_char/` : "Metni Oku" işlemi sırasında kaydedilen karakter debug görüntüleri
- `./.easyocr/` : EasyOCR model cache klasörü (ilk çalıştırmada oluşur)

## Test ve Yardımcı Scriptler

`scripts/` klasörü altında hızlı test ve yardımcı komutlar bulunur.
Bazı scriptlerde sabit dosya yollarının güncellenmesi gerekir:
- `scripts/test_detector.py`
- `scripts/test_ocr.py`
- `scripts/quick_test.py`

## Sık Görülen Sorunlar

- **Model bulunamadı**: `models/best.pt` dosyasının var olduğundan emin olun.
- **Tesseract bulunamadı**: `tesseract --version` çalışmıyorsa Tesseract kurulumu eksiktir.
- **OCR yavaş**: GPU varsa PyTorch + CUDA ile hızlanır. GPU yoksa OCR'yi kapatabilirsiniz.

## Dokümantasyon

Daha ayrıntılı notlar için:
- `docs/KURULUM.md`
- `docs/OCR_IMPROVEMENTS.md`
- `docs/GPU_OPTIMIZATION.md`
