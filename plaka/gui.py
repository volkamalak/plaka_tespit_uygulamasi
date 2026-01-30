"""
Plaka Tespit Uygulaması - Ana Dosya
Tkinter GUI ile plaka tespit ve OCR uygulaması
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from .detector import PlakaDetector
import time


class PlakaTespitUygulamasi:
    """Plaka tespit uygulaması GUI sınıfı"""

    def __init__(self, root):
        """Uygulamayı başlatır"""
        self.root = root
        self.root.title("Plaka Tespit ve Okuma Uygulaması")
        self.root.geometry("1600x900")

        # Tema
        self.colors = {
            "bg": "#f4f5f7",
            "surface": "#ffffff",
            "surface_alt": "#f0f2f5",
            "primary": "#1f2a44",
            "accent": "#2c7be5",
            "accent_alt": "#00a389",
            "danger": "#d64545",
            "text": "#1c1f23",
            "muted": "#6b7280",
            "border": "#e4e7ec",
        }
        self.fonts = {
            "title": ("Segoe UI", 20, "bold"),
            "subtitle": ("Segoe UI", 11, "bold"),
            "body": ("Segoe UI", 10),
            "mono": ("Consolas", 9),
        }

        self.root.configure(bg=self.colors["bg"])

        # Değişkenler
        self.ocr_languages = ['tur', 'eng']
        self.detector = PlakaDetector(use_ocr=True, ocr_languages=self.ocr_languages)
        self.current_image_path = None
        self.original_image = None
        self.cropped_plate = None
        self.last_coords = None
        self.images_are_bgr = True
        self.model_enabled = tk.BooleanVar(value=True)
        self.ocr_enabled = tk.BooleanVar(value=True)
        self.video_path = None
        self.video_cap = None
        self.video_running = False
        self.video_frame_index = 0
        self.video_every_n = 5
        self.video_delay_ms = 33
        self.video_stable_text = None
        self.video_stable_count = 0
        self.video_last_shown = None
        self.video_stable_required = 3
        self.video_stable_target_s = 1.0
        self.video_confidence_threshold = 0.75
        self.video_burst_frames = 10
        self.video_burst_detect_every_n = 2
        self.video_min_bbox_area_ratio = 0.002
        self.video_auto_burst = True
        self.video_burst_in_progress = False
        self.video_burst_cooldown_frames = 0

        # GUI oluştur
        self.create_gui()

    def create_gui(self):
        """GUI elemanlarını oluşturur"""
        # ===== ÜST BAR =====
        header = tk.Frame(self.root, bg=self.colors["primary"], height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        header_inner = tk.Frame(header, bg=self.colors["primary"])
        header_inner.pack(fill=tk.BOTH, expand=True, padx=20)

        title = tk.Label(
            header_inner,
            text="Plaka Tespit ve Okuma",
            font=self.fonts["title"],
            bg=self.colors["primary"],
            fg="white"
        )
        title.pack(side=tk.LEFT, pady=12)

        status_pill = tk.Frame(header_inner, bg=self.colors["primary"])
        status_pill.pack(side=tk.RIGHT, pady=12)

        self.model_status_label = tk.Label(
            status_pill,
            text="Model",
            font=self.fonts["subtitle"],
            bg=self.colors["primary"],
            fg="white"
        )
        self.model_status_label.pack(side=tk.LEFT, padx=8)

        self.ocr_status_label = tk.Label(
            status_pill,
            text="OCR",
            font=self.fonts["subtitle"],
            bg=self.colors["primary"],
            fg="white"
        )
        self.ocr_status_label.pack(side=tk.LEFT, padx=8)

        # ===== KONTROL PANELİ =====
        ctrl_frame = tk.Frame(
            self.root,
            bg=self.colors["surface"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        ctrl_frame.pack(fill=tk.X, padx=16, pady=(12, 8))

        btn_frame = tk.Frame(ctrl_frame, bg=self.colors["surface"])
        btn_frame.pack(fill=tk.X, padx=12, pady=10)

        self.load_btn = tk.Button(
            btn_frame,
            text="Resim Yükle",
            command=self.load_image,
            font=self.fonts["subtitle"],
            bg=self.colors["accent"],
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.show_btn = tk.Button(
            btn_frame,
            text="Plakayı Tespit Et",
            command=self.show_plate,
            font=self.fonts["subtitle"],
            bg="#5b6bfe",
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.show_btn.pack(side=tk.LEFT, padx=5)

        self.read_btn = tk.Button(
            btn_frame,
            text="Metni Oku",
            command=self.read_plate,
            font=self.fonts["subtitle"],
            bg=self.colors["danger"],
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.read_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(
            btn_frame,
            text="Kaydet",
            command=self.save_result,
            font=self.fonts["subtitle"],
            bg=self.colors["accent_alt"],
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.video_btn = tk.Button(
            btn_frame,
            text="Video Yükle",
            command=self.load_video,
            font=self.fonts["subtitle"],
            bg="#374151",
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.video_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = tk.Button(
            btn_frame,
            text="Oynat",
            command=self.toggle_video,
            font=self.fonts["subtitle"],
            bg="#111827",
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        # Switch alanı
        switch_frame = tk.Frame(btn_frame, bg=self.colors["surface"])
        switch_frame.pack(side=tk.LEFT, padx=15)

        self.model_switch = tk.Checkbutton(
            switch_frame,
            text="Model",
            variable=self.model_enabled,
            command=self.toggle_model,
            font=self.fonts["subtitle"],
            bg=self.colors["surface"],
            fg=self.colors["text"],
            selectcolor=self.colors["surface_alt"],
            indicatoron=False,
            width=10,
            relief=tk.GROOVE,
            cursor='hand2'
        )
        self.model_switch.pack(side=tk.LEFT, padx=5)

        self.ocr_switch = tk.Checkbutton(
            switch_frame,
            text="OCR",
            variable=self.ocr_enabled,
            command=self.toggle_ocr,
            font=self.fonts["subtitle"],
            bg=self.colors["surface"],
            fg=self.colors["text"],
            selectcolor=self.colors["surface_alt"],
            indicatoron=False,
            width=10,
            relief=tk.GROOVE,
            cursor='hand2'
        )
        self.ocr_switch.pack(side=tk.LEFT, padx=5)

        status_frame = tk.Frame(btn_frame, bg=self.colors["surface"])
        status_frame.pack(side=tk.RIGHT, padx=10)

        self.status_hint = tk.Label(
            status_frame,
            text="Hazır",
            font=self.fonts["body"],
            bg=self.colors["surface"],
            fg=self.colors["muted"]
        )
        self.status_hint.pack(side=tk.RIGHT)

        # ===== ANA İÇERİK =====
        main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        # SOL: Orijinal Resim
        left_frame = tk.Frame(
            main_frame,
            bg=self.colors["surface"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        left_title = tk.Label(
            left_frame,
            text="Orijinal Resim",
            font=self.fonts["subtitle"],
            bg=self.colors["surface"],
            fg=self.colors["text"],
            pady=10
        )
        left_title.pack(fill=tk.X)

        self.original_canvas = tk.Canvas(left_frame, bg=self.colors["surface_alt"], highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ORTADA: Tespit Edilen Plaka
        mid_frame = tk.Frame(
            main_frame,
            bg=self.colors["surface"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(6, 6))

        mid_title = tk.Label(
            mid_frame,
            text="Tespit Edilen Plaka",
            font=self.fonts["subtitle"],
            bg=self.colors["surface"],
            fg=self.colors["text"],
            pady=10,
            padx=10
        )
        mid_title.pack(fill=tk.X)

        plate_container = tk.Frame(mid_frame, bg=self.colors["surface_alt"])
        plate_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.plate_canvas = tk.Canvas(
            plate_container,
            bg='white',
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            width=220,
            height=120
        )
        self.plate_canvas.pack(expand=True)

        # SAĞ: Sonuçlar
        right_frame = tk.Frame(
            main_frame,
            bg=self.colors["surface"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        right_title = tk.Label(
            right_frame,
            text="Okunan Sonuçlar",
            font=self.fonts["subtitle"],
            bg=self.colors["surface"],
            fg=self.colors["text"],
            pady=10
        )
        right_title.pack(fill=tk.X)

        results_frame = tk.Frame(right_frame, bg=self.colors["surface"])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)

        # MODEL SONUÇLARI
        model_panel = tk.Frame(
            results_frame,
            bg=self.colors["surface_alt"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        model_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=0)

        model_title = tk.Label(
            model_panel,
            text="Model",
            font=self.fonts["subtitle"],
            bg=self.colors["surface_alt"],
            fg=self.colors["text"],
            pady=5
        )
        model_title.pack(fill=tk.X)

        model_scroll = tk.Frame(model_panel, bg=self.colors["surface_alt"])
        model_scroll.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        model_sb = tk.Scrollbar(model_scroll)
        model_sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.model_text = tk.Text(
            model_scroll,
            height=15,
            font=self.fonts["mono"],
            bg='white',
            fg=self.colors["text"],
            yscrollcommand=model_sb.set,
            wrap=tk.WORD
        )
        self.model_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        model_sb.config(command=self.model_text.yview)
        self.model_text.insert('1.0', 'Henüz işlem yapılmadı.')
        self.model_text.config(state=tk.DISABLED)

        # OCR SONUÇLARI
        ocr_panel = tk.Frame(
            results_frame,
            bg=self.colors["surface_alt"],
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        ocr_panel.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=0)

        ocr_title = tk.Label(
            ocr_panel,
            text="OCR",
            font=self.fonts["subtitle"],
            bg=self.colors["surface_alt"],
            fg=self.colors["text"],
            pady=5
        )
        ocr_title.pack(fill=tk.X)

        ocr_scroll = tk.Frame(ocr_panel, bg=self.colors["surface_alt"])
        ocr_scroll.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        ocr_sb = tk.Scrollbar(ocr_scroll)
        ocr_sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.ocr_text = tk.Text(
            ocr_scroll,
            height=15,
            font=self.fonts["mono"],
            bg='white',
            fg=self.colors["text"],
            yscrollcommand=ocr_sb.set,
            wrap=tk.WORD
        )
        self.ocr_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ocr_sb.config(command=self.ocr_text.yview)
        self.ocr_text.insert('1.0', 'Henüz işlem yapılmadı.')
        self.ocr_text.config(state=tk.DISABLED)

        # ===== ALT DURUM BARI =====
        footer = tk.Frame(self.root, bg=self.colors["primary"], height=40)
        footer.pack(fill=tk.X)
        footer.pack_propagate(False)

        self.status_label = tk.Label(
            footer,
            text="Hazır",
            font=self.fonts["body"],
            bg=self.colors["primary"],
            fg='white',
            anchor='w'
        )
        self.status_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        self.update_status_labels()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_status_labels(self):
        """Model/OCR durumlarını günceller"""
        model_ok = self.detector.model is not None
        ocr_ok = self.detector.ocr is not None

        model_status = "Model: Açık" if model_ok else "Model: Yok"
        if not self.model_enabled.get():
            model_status = "Model: Kapalı"
        self.model_status_label.config(text=model_status, fg='white')

        ocr_status = "OCR: Açık" if ocr_ok else "OCR: Yok"
        if not self.ocr_enabled.get():
            ocr_status = "OCR: Kapalı"
        self.ocr_status_label.config(text=ocr_status, fg='white')

    def toggle_model(self):
        """Character model aç/kapat"""
        self.detector.use_character_model = self.model_enabled.get()
        self.update_status_labels()

    def toggle_ocr(self):
        """OCR aç/kapat"""
        self.detector.use_ocr = self.ocr_enabled.get()
        if self.detector.use_ocr and self.detector.ocr is None:
            self.detector.load_ocr(self.ocr_languages)
        self.update_status_labels()

    def load_image(self):
        """Resim yükleme"""
        self.stop_video()
        file_path = filedialog.askopenfilename(
            title="Resim Seçin",
            filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp"), ("Tüm", "*.*")]
        )

        if file_path:
            try:
                self.current_image_path = file_path
                image = cv2.imread(file_path)
                self.original_image = image

                self.display_image(image, self.original_canvas, bgr_to_rgb=True)
                self.show_btn.config(state=tk.NORMAL)
                self.plate_canvas.delete("all")
                self.reset_results()
                self.status_label.config(text="Resim yüklendi")
                self.status_hint.config(text="Görüntü hazır")

            except Exception as e:
                messagebox.showerror("Hata", f"Resim yüklenemedi: {str(e)}")

    def load_video(self):
        """Video yükleme"""
        file_path = filedialog.askopenfilename(
            title="Video Seçin",
            filetypes=[("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv"), ("Tüm", "*.*")]
        )
        if not file_path:
            return

        # Önce mevcut video akışını durdur
        self.stop_video()

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Video açılamadı")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_delay_ms = int(1000 / fps) if fps and fps > 1 else 33
        # Optimum: stabil sonuç ~1 saniyede görünsün
        if fps and fps > 1:
            every_n = int(round((fps * self.video_stable_target_s) / self.video_stable_required))
            self.video_every_n = max(1, min(10, every_n))

        self.video_path = file_path
        self.video_cap = cap
        self.video_running = False
        self.video_frame_index = 0
        self.play_btn.config(state=tk.NORMAL, text="Oynat")

        ok, frame = cap.read()
        if ok:
            self.original_image = frame
            self.display_image(frame, self.original_canvas, bgr_to_rgb=True)
            self.status_label.config(text="Video yüklendi")
            self.status_hint.config(text="Oynatmak için başlat")
        else:
            messagebox.showerror("Hata", "Video ilk kare okunamadı")
            self.stop_video()

    def toggle_video(self):
        """Video oynat/durdur"""
        if not self.video_cap:
            return
        self.video_running = not self.video_running
        self.play_btn.config(text="Durdur" if self.video_running else "Oynat")
        if self.video_running:
            self.update_video_frame()

    def run_video_burst(self, auto=False):
        """Video/stream içinde burst analiz yap."""
        if not self.video_cap:
            if not auto:
                messagebox.showwarning("Uyarı", "Önce video yükleyin!")
            return

        if self.video_burst_in_progress:
            return

        self.video_burst_in_progress = True

        self.status_label.config(text="Burst analiz ediliyor...")
        self.status_hint.config(text="Lütfen bekleyin")
        self.root.update()

        read_text = bool(self.detector.use_character_model)

        try:
            result = self.detector.detect_plate_in_stream(
                self.video_cap,
                conf_threshold=0.25,
                burst_frames=self.video_burst_frames,
                read_text=read_text,
                min_bbox_area_ratio=self.video_min_bbox_area_ratio,
                detect_every_n=self.video_burst_detect_every_n,
            )
        except Exception as e:
            if not auto:
                messagebox.showerror("Hata", f"Burst analiz hatası: {str(e)}")
            self.status_label.config(text="Burst başarısız")
            self.video_burst_in_progress = False
            return

        if result.get("last_frame") is not None:
            self.original_image = result["last_frame"]
            self.display_image(self.original_image, self.original_canvas, bgr_to_rgb=True)

        if not result.get("success"):
            err = result.get("error") or "Burst sonucu alınamadı"
            self.status_label.config(text="Burst başarısız")
            self.status_hint.config(text=err)
            self.video_burst_in_progress = False
            return

        self.last_coords = result.get("best_bbox")
        best_crop = result.get("best_plate_crop")
        if best_crop is not None:
            self.cropped_plate = best_crop
            self.display_image(self.cropped_plate, self.plate_canvas, bgr_to_rgb=True)
            self.read_btn.config(state=tk.NORMAL)

        plate_text = result.get("best_plate_text") or ""
        plate_conf = float(result.get("best_plate_conf") or 0.0)
        det_conf = float(result.get("best_detector_conf") or 0.0)
        sharpness = float(result.get("best_sharpness") or 0.0)

        if plate_text:
            self.status_label.config(
                text=f"Burst: {plate_text} ({plate_conf:.1%})"
            )
        else:
            self.status_label.config(
                text=f"Burst: Plaka bulundu ({det_conf:.1%})"
            )
        self.status_hint.config(text="Burst tamamlandı")

        # Sonuç panellerini güncelle
        if read_text and plate_text:
            model_result = "Burst Sonucu\n" + ("─" * 25) + "\n"
            model_result += f"Plaka: {plate_text}\n"
            model_result += f"Güven: {plate_conf:.1%}\n"
            model_result += f"Det. Güven: {det_conf:.1%}\n"
            model_result += f"Keskinlik: {sharpness:.1f}"
        else:
            model_result = "Burst\nMetin okunmadı"

        ocr_result = "OCR bu modda çalıştırılmadı"

        self.model_text.config(state=tk.NORMAL)
        self.model_text.delete('1.0', tk.END)
        self.model_text.insert('1.0', model_result)
        self.model_text.config(state=tk.DISABLED)

        self.ocr_text.config(state=tk.NORMAL)
        self.ocr_text.delete('1.0', tk.END)
        self.ocr_text.insert('1.0', ocr_result)
        self.ocr_text.config(state=tk.DISABLED)

        self.video_burst_in_progress = False

    def update_video_frame(self):
        """Video karelerini güncelle"""
        if not self.video_running or not self.video_cap:
            return

        ok, frame = self.video_cap.read()
        if not ok:
            self.video_running = False
            self.play_btn.config(text="Oynat")
            self.status_label.config(text="Video bitti")
            return

        self.original_image = frame
        self.display_image(frame, self.original_canvas, bgr_to_rgb=True)

        if self.video_burst_cooldown_frames > 0:
            self.video_burst_cooldown_frames -= 1

        # Video akışında plaka tespitini her N karede yap
        if self.detector.model:
            if self.video_frame_index % self.video_every_n == 0:
                result = self.detector.detect_plate_in_image(frame, read_text=False)
                if result['success'] and result['coordinates']:
                    x1, y1, x2, y2 = result['coordinates'][0]
                    self.cropped_plate = frame[y1:y2, x1:x2]
                    self.last_coords = (x1, y1, x2, y2)
                    self.display_image(self.cropped_plate, self.plate_canvas, bgr_to_rgb=True)
                    self.read_btn.config(state=tk.NORMAL)
                    burst_ran = False
                    if self.video_auto_burst and not self.video_burst_in_progress:
                        if self.video_burst_cooldown_frames == 0:
                            self.run_video_burst(auto=True)
                            self.video_burst_cooldown_frames = max(
                                self.video_burst_frames, self.video_every_n
                            )
                            burst_ran = True
                    if not burst_ran:
                        if self.detector.use_character_model or self.detector.use_ocr:
                            self.process_video_read()

        self.video_frame_index += 1
        self.root.after(self.video_delay_ms, self.update_video_frame)

    def process_video_read(self):
        """Video akışında otomatik okuma (stabil veya yüksek güven ile göster)."""
        if self.cropped_plate is None:
            return

        _, _, model_data, ocr_data = self.detector.read_plate_text(
            self.cropped_plate,
            image_is_rgb=not self.images_are_bgr,
            full_image=self.original_image,
            coords=self.last_coords,
            use_full_image_for_char=False,
            debug_dir=None
        )

        candidate = ""
        candidate_conf = 0.0
        candidate_source = None
        if self.detector.use_character_model and model_data and model_data.get("success"):
            candidate = model_data.get("text", "")
            candidate_conf = float(model_data.get("confidence", 0.0) or 0.0)
            candidate_source = "model"
        if not candidate and self.detector.use_ocr and ocr_data:
            best = ocr_data.get("best") if isinstance(ocr_data, dict) else ocr_data
            if best and best.get("success"):
                candidate = best.get("text", "")
                candidate_conf = float(best.get("confidence", 0.0) or 0.0)
                candidate_source = "ocr"

        if not candidate:
            self.video_stable_text = None
            self.video_stable_count = 0
            self.status_hint.config(text="Okuma bekleniyor")
            return

        if candidate == self.video_stable_text:
            self.video_stable_count += 1
        else:
            self.video_stable_text = candidate
            self.video_stable_count = 1

        is_stable = self.video_stable_count >= self.video_stable_required
        is_high_conf = candidate_conf >= self.video_confidence_threshold
        if not is_stable and not is_high_conf:
            hint = f"Okuma stabil değil ({self.video_stable_count}/{self.video_stable_required})"
            hint += f" | Güven: {candidate_conf:.0%}"
            self.status_hint.config(text=hint)
            return

        if candidate == self.video_last_shown:
            return

        # Stable result: update UI
        self.video_last_shown = candidate
        if is_high_conf and not is_stable:
            source_label = "Model" if candidate_source == "model" else "OCR"
            self.status_hint.config(
                text=f"Yüksek güven ({source_label}): {candidate_conf:.0%}"
            )
        else:
            self.status_hint.config(text="Okuma sabitlenmiş")

        model_result = "Model kapalı"
        if self.detector.use_character_model and model_data and model_data.get("success"):
            model_result = "Model Sonucu\n" + ("─" * 25) + "\n"
            model_result += f"Plaka: {model_data['text']}\n"
            model_result += f"Güven: {model_data['confidence']:.1%}\n"
            model_result += f"Karakter: {len(model_data['characters'])}\n"
            model_result += f"Süre: {model_data['processing_time']:.3f}s"

        ocr_result = "OCR kapalı"
        if self.detector.use_ocr and ocr_data:
            best = ocr_data.get("best") if isinstance(ocr_data, dict) else ocr_data
            bgr = ocr_data.get("bgr") if isinstance(ocr_data, dict) else None
            gray = ocr_data.get("gray") if isinstance(ocr_data, dict) else None
            if best and best.get("success"):
                ocr_result = "OCR Sonucu\n" + ("─" * 25) + "\n"
                ocr_result += f"En İyi: {best['text']} ({best['confidence']:.1%})\n"
            else:
                ocr_result = "OCR\nBaşarısız"
            if bgr:
                ocr_result += f"BGR: {bgr.get('text', '')} ({bgr.get('confidence', 0.0):.1%})\n"
            if gray:
                ocr_result += f"GRAY: {gray.get('text', '')} ({gray.get('confidence', 0.0):.1%})"

        self.model_text.config(state=tk.NORMAL)
        self.model_text.delete('1.0', tk.END)
        self.model_text.insert('1.0', model_result)
        self.model_text.config(state=tk.DISABLED)

        self.ocr_text.config(state=tk.NORMAL)
        self.ocr_text.delete('1.0', tk.END)
        self.ocr_text.insert('1.0', ocr_result)
        self.ocr_text.config(state=tk.DISABLED)

    def stop_video(self):
        """Video kaynağını kapat"""
        if self.video_cap:
            try:
                self.video_cap.release()
            except Exception:
                pass
        self.video_cap = None
        self.video_running = False
        self.video_path = None
        self.video_stable_text = None
        self.video_stable_count = 0
        self.video_last_shown = None
        self.play_btn.config(state=tk.DISABLED, text="Oynat")
        self.video_burst_in_progress = False
        self.video_burst_cooldown_frames = 0

    def on_close(self):
        """Uygulama kapanışı"""
        self.stop_video()
        self.root.destroy()

    def show_plate(self):
        """Plakayı tespit et"""
        if not self.current_image_path:
            messagebox.showwarning("Uyarı", "Resim yükleyin!")
            return

        if not self.detector.model:
            messagebox.showerror("Hata", "Model yüklü değil!")
            return

        try:
            self.status_label.config(text="Plaka tespit ediliyor...")
            self.root.update()

            start = time.time()
            result = self.detector.detect_plate(self.current_image_path, read_text=False)

            if result['success'] and result['coordinates']:
                x1, y1, x2, y2 = result['coordinates'][0]
                conf = result['confidence'][0]

                self.cropped_plate = self.original_image[y1:y2, x1:x2]
                self.last_coords = (x1, y1, x2, y2)
                self.display_image(self.cropped_plate, self.plate_canvas, bgr_to_rgb=True)

                elapsed = time.time() - start
                self.status_label.config(
                    text=f"Plaka tespit edildi ({conf:.1%}) • {elapsed:.2f}s"
                )
                self.read_btn.config(state=tk.NORMAL)
                self.reset_results()

            else:
                messagebox.showwarning("Uyarı", "Plaka tespit edilemedi!")
                self.status_label.config(text="Plaka bulunamadı")

        except Exception as e:
            messagebox.showerror("Hata", f"Tespit hatası: {str(e)}")

    def read_plate(self):
        """OCR ile plakayı oku"""
        if self.cropped_plate is None:
            messagebox.showwarning("Uyarı", "Önce plakayı tespit edin!")
            return

        try:
            self.status_label.config(text="Plaka okunuyor...")
            self.root.update()

            start_total = time.time()

            # Model ve OCR Sonuçlarını al
            _, _, model_data, ocr_data = self.detector.read_plate_text(
                self.cropped_plate,
                image_is_rgb=not self.images_are_bgr,
                full_image=self.original_image,
                coords=self.last_coords,
                use_full_image_for_char=False,
                debug_dir="results/debug_char"
            )
            _ = time.time() - start_total

            # MODEL SONUCU
            model_result = "Model devre dışı"
            if not self.detector.use_character_model:
                model_result = "Model kapalı"
            elif model_data:
                if model_data['success']:
                    model_result = "Model Sonucu\n" + ("─" * 25) + "\n"
                    model_result += f"Plaka: {model_data['text']}\n"
                    model_result += f"Güven: {model_data['confidence']:.1%}\n"
                    model_result += f"Karakter: {len(model_data['characters'])}\n"
                    model_result += f"Süre: {model_data['processing_time']:.3f}s"
                else:
                    model_result = f"Model\n{model_data['error']}"
            else:
                model_result = "Model başlatılmadı"

            # OCR SONUCU (BGR + GRAY)
            ocr_result = "OCR devre dışı"
            if not self.detector.use_ocr:
                ocr_result = "OCR kapalı"
            elif ocr_data:
                best = ocr_data.get("best") if isinstance(ocr_data, dict) else ocr_data
                bgr = ocr_data.get("bgr") if isinstance(ocr_data, dict) else None
                gray = ocr_data.get("gray") if isinstance(ocr_data, dict) else None

                if best and best.get("success"):
                    ocr_result = "OCR Sonucu\n" + ("─" * 25) + "\n"
                    ocr_result += f"En İyi: {best['text']} ({best['confidence']:.1%})\n"
                else:
                    ocr_result = "OCR\nBaşarısız"

                if bgr:
                    ocr_result += f"BGR: {bgr.get('text', '')} ({bgr.get('confidence', 0.0):.1%})\n"
                if gray:
                    ocr_result += f"GRAY: {gray.get('text', '')} ({gray.get('confidence', 0.0):.1%})"
            else:
                ocr_result = "OCR başlatılmadı"

            # Sonuçları göster
            self.model_text.config(state=tk.NORMAL)
            self.model_text.delete('1.0', tk.END)
            self.model_text.insert('1.0', model_result)
            self.model_text.config(state=tk.DISABLED)

            self.ocr_text.config(state=tk.NORMAL)
            self.ocr_text.delete('1.0', tk.END)
            self.ocr_text.insert('1.0', ocr_result)
            self.ocr_text.config(state=tk.DISABLED)

            self.status_label.config(text="Okuma tamamlandı")

        except Exception as e:
            messagebox.showerror("Hata", f"Okuma hatası: {str(e)}")
            self.status_label.config(text="Okuma başarısız")

    def reset_results(self):
        """Sonuçları sıfırla"""
        self.model_text.config(state=tk.NORMAL)
        self.model_text.delete('1.0', tk.END)
        self.model_text.insert('1.0', 'Henüz işlem yapılmadı.')
        self.model_text.config(state=tk.DISABLED)

        self.ocr_text.config(state=tk.NORMAL)
        self.ocr_text.delete('1.0', tk.END)
        self.ocr_text.insert('1.0', 'Henüz işlem yapılmadı.')
        self.ocr_text.config(state=tk.DISABLED)

    def save_result(self):
        """Sonucu kaydet"""
        if self.cropped_plate is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek plaka yok!")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )

        if path:
            try:
                if self.images_are_bgr:
                    plate_bgr = self.cropped_plate
                else:
                    plate_bgr = cv2.cvtColor(self.cropped_plate, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, plate_bgr)
                messagebox.showinfo("Başarılı", f"Kaydedildi:\n{path}")
                self.status_label.config(text="Plaka kaydedildi")
            except Exception as e:
                messagebox.showerror("Hata", f"Kayıt hatası: {str(e)}")

    def display_image(self, image, canvas, bgr_to_rgb=False):
        """Resmi canvas'a göster"""
        if image is None:
            return
        if bgr_to_rgb and len(image.shape) == 3 and image.shape[2] == 3:
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

        canvas.update()
        w, h = canvas.winfo_width(), canvas.winfo_height()

        if w < 2 or h < 2:
            return

        pil_img = Image.fromarray(image)
        scale = min(w / pil_img.width, h / pil_img.height)
        new_w, new_h = int(pil_img.width * scale * 0.95), int(pil_img.height * scale * 0.95)

        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo


def main():
    root = tk.Tk()
    app = PlakaTespitUygulamasi(root)
    root.mainloop()


if __name__ == "__main__":
    main()
