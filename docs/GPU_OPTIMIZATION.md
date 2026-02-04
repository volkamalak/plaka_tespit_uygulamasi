# ⚡ GPU Optimizasyon Notları

Bu proje OCR kullanmaz. Hızlandırma, YOLO tespit ve karakter modeli üzerinden yapılır.

## ✅ Öneriler

- **GPU aktif mi kontrol et**: PyTorch/CUDA kurulu olmalı.
- **Ultralytics YOLO** GPU’yu otomatik kullanır. İsterseniz `device=0` ile zorlayabilirsiniz.
- **Karakter modeli**: GPU varsa otomatik kullanır (Ultralytics/torch üzerinden).

## 🚀 Hızlı Başlangıç

```bash
# GPU ile çalıştır
.venv/bin/python main.py

# Test et
.venv/bin/python scripts/test_detector.py
```

## 🐛 Sık Sorunlar

- **Model bulunamadı**: `models/best.pt` yerleştirin.
- **Okuma yok**: `models/character_best.pt` yoksa metin okunamaz.
- **GPU kullanılmıyor**: PyTorch CUDA kurulumunu kontrol edin.
