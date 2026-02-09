"""
SANKO Port - Ana Dosya
Tkinter GUI ile plaka tespit ve model tabanlı okuma uygulaması
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from .detector import PlakaDetector
from .container_detector import ContainerNumberDetector
from .seal_detector import ContainerSealDetector
from .damage_detector import ContainerDamageDetector
import time


class PlakaTespitUygulamasi:
    """Plaka tespit uygulaması GUI sınıfı"""

    def __init__(self, root):
        """Uygulamayı başlatır"""
        self.root = root
        self.root.title("SANKO Port")
        self.root.geometry("1680x900")

        # Tema
        self.colors = {
            "bg": "#101418",
            "surface": "#1a2028",
            "surface_alt": "#222a34",
            "primary": "#0f141a",
            "accent": "#1e6fbf",
            "accent_alt": "#25b26b",
            "danger": "#d84a4a",
            "text": "#eef2f6",
            "muted": "#9aa4b2",
            "border": "#2e3642",
            "pill": "#1f2732",
            "success": "#25b26b",
            "warning": "#e6b84d",
            "info": "#2f87ff",
        }
        self.fonts = {
            "title": ("Segoe UI", 20, "bold"),
            "subtitle": ("Segoe UI", 11, "bold"),
            "body": ("Segoe UI", 10),
            "mono": ("Consolas", 9),
            "panel_title": ("Segoe UI", 12, "bold"),
            "big_value": ("Segoe UI", 22, "bold"),
            "big_value_alt": ("Segoe UI", 18, "bold"),
        }

        self.root.configure(bg=self.colors["bg"])

        # Değişkenler
        self.detector = PlakaDetector()
        self.container_detector = ContainerNumberDetector()
        self.seal_detector = ContainerSealDetector()
        self.damage_detector = ContainerDamageDetector(min_damage_conf=0.65)
        self.current_image_path = None
        self.original_image = None
        self.cropped_plate = None
        self.last_coords = None
        self.container_crop = None
        self.container_coords = None
        self.images_are_bgr = True
        self.model_enabled = tk.BooleanVar(value=True)
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
        self.video_confidence_threshold = 0.65
        self.overlay_boxes = []

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

        brand = tk.Frame(header_inner, bg=self.colors["primary"])
        brand.pack(side=tk.LEFT, pady=10)

        title_block = tk.Frame(brand, bg=self.colors["primary"])
        title_block.pack(side=tk.LEFT)

        title = tk.Label(
            title_block,
            text="SANKO Port",
            font=self.fonts["title"],
            bg=self.colors["primary"],
            fg=self.colors["text"]
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            title_block,
            text="AI-Powered Reading & Control System",
            font=("Segoe UI", 9),
            bg=self.colors["primary"],
            fg=self.colors["muted"]
        )
        subtitle.pack(anchor="w")

        header_right = tk.Frame(header_inner, bg=self.colors["primary"])
        header_right.pack(side=tk.RIGHT, pady=12)

        def pill(parent, text, dot_color):
            frame = tk.Frame(parent, bg=self.colors["pill"], highlightthickness=1, highlightbackground=self.colors["border"])
            label = tk.Label(frame, text=text, font=self.fonts["body"], bg=self.colors["pill"], fg=self.colors["text"])
            label.pack(side=tk.LEFT, padx=(10, 6), pady=6)
            dot = tk.Label(frame, text="●", font=self.fonts["body"], bg=self.colors["pill"], fg=dot_color)
            dot.pack(side=tk.LEFT, padx=(0, 10))
            return frame, label, dot

        self.system_pill, self.system_pill_label, self.system_pill_dot = pill(
            header_right, "System", self.colors["success"]
        )
        self.system_pill.pack(side=tk.LEFT, padx=6)

        self.status_pill, self.status_pill_label, self.status_pill_dot = pill(
            header_right, "Status: Stable", self.colors["success"]
        )
        self.status_pill.pack(side=tk.LEFT, padx=6)

        notif = tk.Label(header_right, text="🔔", font=("Segoe UI", 12), bg=self.colors["primary"], fg=self.colors["muted"])
        notif.pack(side=tk.LEFT, padx=10)

        # ===== ANA İÇERİK =====
        main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        left_col = tk.Frame(main_frame, bg=self.colors["bg"])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_col = tk.Frame(main_frame, bg=self.colors["bg"], width=440)
        right_col.pack(side=tk.LEFT, fill=tk.Y)
        right_col.pack_propagate(False)

        # ===== VIDEO PANEL =====
        video_panel = tk.Frame(left_col, bg=self.colors["surface"], highlightthickness=1, highlightbackground=self.colors["border"])
        video_panel.pack(fill=tk.BOTH, expand=True)

        video_header = tk.Frame(video_panel, bg=self.colors["surface"])
        video_header.pack(fill=tk.X, padx=12, pady=(10, 0))

        video_title = tk.Label(
            video_header,
            text="VIDEO",
            font=self.fonts["panel_title"],
            bg=self.colors["surface"],
            fg=self.colors["text"]
        )
        video_title.pack(side=tk.LEFT)

        video_ctrl = tk.Frame(video_header, bg=self.colors["surface"])
        video_ctrl.pack(side=tk.RIGHT)

        self.play_btn = tk.Button(
            video_ctrl,
            text="Oynat",
            command=self.toggle_video,
            font=self.fonts["body"],
            bg="#2d3642",
            fg="white",
            padx=10,
            pady=4,
            cursor="hand2",
            relief=tk.FLAT,
            activebackground="#2d3642",
            activeforeground="white",
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.RIGHT, padx=(8, 0))

        video_ctrl_label = tk.Label(video_ctrl, text="—   ✕", font=("Segoe UI", 10), bg=self.colors["surface"], fg=self.colors["muted"])
        video_ctrl_label.pack(side=tk.RIGHT)

        video_body = tk.Frame(video_panel, bg=self.colors["surface"])
        video_body.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.original_canvas = tk.Canvas(video_body, bg="#0f141a", highlightthickness=1, highlightbackground=self.colors["border"])
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # ===== LOG PANEL =====
        log_panel = tk.Frame(left_col, bg=self.colors["surface"], highlightthickness=1, highlightbackground=self.colors["border"])
        log_panel.pack(fill=tk.X, pady=(12, 0))

        log_header = tk.Frame(log_panel, bg=self.colors["surface"])
        log_header.pack(fill=tk.X, padx=12, pady=(10, 0))

        log_title = tk.Label(log_header, text="LOG", font=self.fonts["panel_title"], bg=self.colors["surface"], fg=self.colors["text"])
        log_title.pack(side=tk.LEFT)

        log_body = tk.Frame(log_panel, bg=self.colors["surface"])
        log_body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        log_scroll = tk.Scrollbar(log_body)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_body,
            height=6,
            font=self.fonts["mono"],
            bg="#11161d",
            fg=self.colors["muted"],
            insertbackground=self.colors["text"],
            yscrollcommand=log_scroll.set,
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        self.log_text.insert("1.0", "System ready...\n")
        self.log_text.config(state=tk.DISABLED)

        # ===== RIGHT COLUMN CARDS =====
        def card(parent, title_text):
            frame = tk.Frame(parent, bg=self.colors["surface"], highlightthickness=1, highlightbackground=self.colors["border"])
            title = tk.Label(frame, text=title_text, font=self.fonts["panel_title"], bg=self.colors["surface"], fg=self.colors["text"])
            title.pack(anchor="w", padx=12, pady=(10, 6))
            body = tk.Frame(frame, bg=self.colors["surface"])
            body.pack(fill=tk.X, padx=12, pady=(0, 12))
            return frame, body

        # PLAKA ISLEMLERI
        plate_ops, plate_ops_body = card(right_col, "PLAKA İŞLEMLERİ")
        plate_ops.pack(fill=tk.X, pady=(0, 12))

        btn_row = tk.Frame(plate_ops_body, bg=self.colors["surface"])
        btn_row.pack(fill=tk.X)

        def styled_button(parent, text, command, bg, state=tk.NORMAL):
            return tk.Button(
                parent,
                text=text,
                command=command,
                font=self.fonts["body"],
                bg=bg,
                fg="white",
                padx=12,
                pady=8,
                cursor="hand2",
                relief=tk.FLAT,
                activebackground=bg,
                activeforeground="white",
                state=state
            )

        self.load_btn = styled_button(btn_row, "Görüntü Yükle", self.load_image, self.colors["accent"])
        self.load_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.video_btn = styled_button(btn_row, "Kamera Başlat", self.load_video, "#3a4454")
        self.video_btn.pack(side=tk.LEFT, padx=8)

        self.detect_read_btn = styled_button(btn_row, "Plaka Tespit & Oku", self.detect_and_read_plate, self.colors["info"], state=tk.DISABLED)
        self.detect_read_btn.pack(side=tk.LEFT, padx=(8, 0))

        # play button moved to video header for visibility

        # KONTEYNER ISLEMLERI
        cont_ops, cont_ops_body = card(right_col, "KONTEYNER İŞLEMLERİ")
        cont_ops.pack(fill=tk.X, pady=(0, 12))

        cont_row = tk.Frame(cont_ops_body, bg=self.colors["surface"])
        cont_row.pack(fill=tk.X)

        self.container_find_btn = styled_button(cont_row, "Konteyner ISO Bul ve Oku", self.find_container_iso, "#1f6b63", state=tk.DISABLED)
        self.container_find_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.container_read_btn = styled_button(cont_row, "Konteyner ISO Oku", self.read_container_iso, "#2454b5", state=tk.DISABLED)
        self.container_read_btn.pack(side=tk.LEFT, padx=8)

        cont_row2 = tk.Frame(cont_ops_body, bg=self.colors["surface"])
        cont_row2.pack(fill=tk.X, pady=(8, 0))

        self.seal_btn = styled_button(cont_row2, "Mühür Kontrol Et", self.check_container_seal, "#7c2d12", state=tk.DISABLED)
        self.seal_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.damage_btn = styled_button(cont_row2, "Hasar Kontrol Et", self.check_container_damage, "#8f1f2f", state=tk.DISABLED)
        self.damage_btn.pack(side=tk.LEFT, padx=8)

        # PLAKA SONUCU
        plate_result, plate_result_body = card(right_col, "PLAKA SONUCU")
        plate_result.pack(fill=tk.X, pady=(0, 12))

        plate_value_row = tk.Frame(plate_result_body, bg=self.colors["surface"])
        plate_value_row.pack(fill=tk.X)

        self.plate_result_value = tk.Label(
            plate_value_row,
            text="--",
            font=self.fonts["big_value"],
            bg=self.colors["surface"],
            fg=self.colors["text"]
        )
        self.plate_result_value.pack(side=tk.LEFT)

        self.plate_result_status = tk.Label(
            plate_value_row,
            text="○",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["muted"]
        )
        self.plate_result_status.pack(side=tk.RIGHT, padx=6)

        # KONTEYNER ISO SONUCU
        cont_result, cont_result_body = card(right_col, "KONTEYNER ISO SONUCU")
        cont_result.pack(fill=tk.X, pady=(0, 12))

        self.container_result_value = tk.Label(
            cont_result_body,
            text="--",
            font=self.fonts["big_value_alt"],
            bg=self.colors["surface"],
            fg=self.colors["text"]
        )
        self.container_result_value.pack(anchor="w")

        self.container_result_sub = tk.Label(
            cont_result_body,
            text="",
            font=self.fonts["body"],
            bg=self.colors["surface"],
            fg=self.colors["muted"]
        )
        self.container_result_sub.pack(anchor="w", pady=(4, 0))

        self.container_result_status = tk.Label(
            cont_result_body,
            text="○",
            font=("Segoe UI", 16, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["muted"]
        )
        self.container_result_status.pack(anchor="e")

        # MÜHÜR KONTROLÜ
        seal_result, seal_result_body = card(right_col, "MÜHÜR KONTROLÜ")
        seal_result.pack(fill=tk.X)

        self.seal_result_value = tk.Label(
            seal_result_body,
            text="--",
            font=self.fonts["big_value_alt"],
            bg=self.colors["surface"],
            fg=self.colors["muted"]
        )
        self.seal_result_value.pack(anchor="w")

        # KONTEYNER HASAR KONTROLÜ
        damage_result, damage_result_body = card(right_col, "KONTEYNER HASAR KONTROLÜ")
        damage_result.pack(fill=tk.X, pady=(12, 0))

        self.damage_result_value = tk.Label(
            damage_result_body,
            text="--",
            font=self.fonts["big_value_alt"],
            bg=self.colors["surface"],
            fg=self.colors["muted"]
        )
        self.damage_result_value.pack(anchor="w")

        self.damage_result_sub = tk.Label(
            damage_result_body,
            text="",
            font=self.fonts["body"],
            bg=self.colors["surface"],
            fg=self.colors["muted"]
        )
        self.damage_result_sub.pack(anchor="w", pady=(4, 0))

        # Hidden plate canvas for preview (kept for existing flow)
        self.plate_canvas = tk.Canvas(right_col, width=1, height=1, highlightthickness=0, bg=self.colors["bg"])
        self.plate_canvas.place_forget()

        # Status labels
        self.status_hint = tk.Label(
            header_right,
            text="Hazır",
            font=self.fonts["body"],
            bg=self.colors["primary"],
            fg=self.colors["muted"]
        )
        self.status_hint.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(
            header_right,
            text="",
            font=self.fonts["body"],
            bg=self.colors["primary"],
            fg=self.colors["muted"]
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.update_status_labels()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_status_labels(self):
        """Model durumunu günceller"""
        model_ok = self.detector.model is not None and self.model_enabled.get()
        if model_ok:
            self.status_pill_label.config(text="Status: Stable")
            self.status_pill_dot.config(fg=self.colors["success"])
        else:
            self.status_pill_label.config(text="Status: Model Off")
            self.status_pill_dot.config(fg=self.colors["warning"])

    def log_message(self, message):
        """Log paneline mesaj ekler."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert("1.0", f"{ts} - {message}\n")
        self.log_text.config(state=tk.DISABLED)

    def _set_overlays(self, overlays):
        self.overlay_boxes = overlays or []

    def _draw_overlays(self, image):
        if image is None or not self.overlay_boxes:
            return image
        overlay = image.copy()
        for item in self.overlay_boxes:
            x1, y1, x2, y2 = item.get("bbox", (0, 0, 0, 0))
            color = item.get("color", (0, 200, 255))
            label = item.get("label", "")
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                cv2.putText(overlay, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return overlay

    def toggle_model(self):
        """Character model aç/kapat"""
        self.detector.use_character_model = self.model_enabled.get()
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
                self.detect_read_btn.config(state=tk.NORMAL)
                self.container_find_btn.config(state=tk.NORMAL)
                self.container_read_btn.config(state=tk.DISABLED)
                self.seal_btn.config(state=tk.NORMAL)
                self.damage_btn.config(state=tk.NORMAL)
                self.plate_canvas.delete("all")
                self.reset_results()
                self.reset_container_results()
                self.container_crop = None
                self.container_coords = None
                self._set_overlays([])
                self.status_label.config(text="Resim yüklendi")
                self.status_hint.config(text="Görüntü hazır")
                self.log_message("Görüntü yüklendi")

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
        self.video_running = True
        self.video_frame_index = 0
        self.play_btn.config(state=tk.NORMAL, text="Durdur")

        ok, frame = cap.read()
        if ok:
            self.original_image = frame
            self.display_image(frame, self.original_canvas, bgr_to_rgb=True)
            self.status_label.config(text="Video yüklendi")
            self.status_hint.config(text="Video akışı başladı")
            self.container_find_btn.config(state=tk.NORMAL)
            self.container_read_btn.config(state=tk.DISABLED)
            self.seal_btn.config(state=tk.NORMAL)
            self.damage_btn.config(state=tk.NORMAL)
            self.update_video_frame()
            self.log_message("Video yüklendi ve akış başladı")
        else:
            messagebox.showerror("Hata", "Video ilk kare okunamadı")
            self.stop_video()

    def toggle_video(self):
        """Video oynat/durdur"""
        if not self.video_cap:
            return
        self.video_running = not self.video_running
        self.play_btn.config(text="Durdur" if self.video_running else "Oynat")
        self.status_label.config(text="Video oynatılıyor" if self.video_running else "Video durduruldu")
        self.log_message("Video oynatıldı" if self.video_running else "Video durduruldu")
        if self.video_running:
            self.update_video_frame()

    def update_video_frame(self):
        """Video karelerini güncelle"""
        if not self.video_running or not self.video_cap:
            return

        ok, frame = self.video_cap.read()
        if not ok:
            self.video_running = False
            self.play_btn.config(text="Oynat")
            self.status_label.config(text="Video bitti")
            self.log_message("Video bitti")
            return

        self.original_image = frame
        self.display_image(frame, self.original_canvas, bgr_to_rgb=True)

        # Video akışında plaka tespitini her N karede yap
        if self.detector.model:
            if self.video_frame_index % self.video_every_n == 0:
                result = self.detector.detect_plate_in_image(frame, read_text=False)
                if result['success'] and result['coordinates']:
                    x1, y1, x2, y2 = result['coordinates'][0]
                    self.cropped_plate = frame[y1:y2, x1:x2]
                    self.last_coords = (x1, y1, x2, y2)
                    self._set_overlays([{
                        "bbox": (x1, y1, x2, y2),
                        "label": "License Plate",
                        "color": (0, 200, 90),
                    }])
                    self.display_image(self.cropped_plate, self.plate_canvas, bgr_to_rgb=True)
                    self.detect_read_btn.config(state=tk.NORMAL)
                    if self.detector.use_character_model:
                        self.process_video_read()

        self.video_frame_index += 1
        self.root.after(self.video_delay_ms, self.update_video_frame)

    def process_video_read(self):
        """Video akışında otomatik okuma (stabil veya yüksek güven ile göster)."""
        if self.cropped_plate is None:
            return

        _, _, model_data = self.detector.read_plate_text(
            self.cropped_plate,
            image_is_rgb=not self.images_are_bgr,
            full_image=self.original_image,
            coords=self.last_coords,
            use_full_image_for_char=False,
            debug_dir=None
        )

        candidate = ""
        candidate_conf = 0.0
        if self.detector.use_character_model and model_data and model_data.get("success"):
            candidate = model_data.get("text", "")
            candidate_conf = float(model_data.get("confidence", 0.0) or 0.0)

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
            self.status_hint.config(text=f"Yüksek güven (Model): {candidate_conf:.0%}")
        else:
            self.status_hint.config(text="Okuma sabitlenmiş")

        if self.detector.use_character_model and model_data and model_data.get("success"):
            self.plate_result_value.config(text=model_data.get("text", "--"))
            self.plate_result_status.config(text="✔", fg=self.colors["success"])
            self.log_message(f"Plaka okundu: {model_data.get('text', '')}")

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
        self.status_label.config(text="Video durduruldu")
        self.play_btn.config(state=tk.DISABLED, text="Oynat")

    def on_close(self):
        """Uygulama kapanışı"""
        self.stop_video()
        self.root.destroy()

    def show_plate(self):
        """Plakayı tespit et"""
        if not self.detector.model:
            # Lazy reload in case model was added after GUI start
            try:
                self.detector.load_model()
            except Exception:
                pass
        if not self.detector.model:
            messagebox.showerror(
                "Hata",
                f"Model yüklü değil! Beklenen dosya: {self.detector.model_path}"
            )
            return

        try:
            self.status_label.config(text="Plaka tespit ediliyor...")
            self.root.update()

            start = time.time()
            if self.current_image_path:
                result = self.detector.detect_plate(self.current_image_path, read_text=False)
            elif self.original_image is not None:
                result = self.detector.detect_plate_in_image(self.original_image, read_text=False)
            else:
                messagebox.showwarning("Uyarı", "Önce görüntü veya video yükleyin!")
                return

            if result['success'] and result['coordinates']:
                areas = []
                for idx, (x1, y1, x2, y2) in enumerate(result['coordinates']):
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    areas.append((area, idx))
                _, best_idx = min(areas, key=lambda x: x[0])
                x1, y1, x2, y2 = result['coordinates'][best_idx]
                conf = result['confidence'][best_idx]

                self.cropped_plate = self.original_image[y1:y2, x1:x2]
                self.last_coords = (x1, y1, x2, y2)
                self._set_overlays([{
                    "bbox": (x1, y1, x2, y2),
                    "label": "License Plate",
                    "color": (0, 200, 90),
                }])
                self.display_image(self.original_image, self.original_canvas, bgr_to_rgb=True)
                self.display_image(self.cropped_plate, self.plate_canvas, bgr_to_rgb=True)

                elapsed = time.time() - start
                self.status_label.config(
                    text=f"Plaka tespit edildi ({conf:.1%}) • {elapsed:.2f}s"
                )
                self.detect_read_btn.config(state=tk.NORMAL)
                self.reset_results()
                self.log_message("Plaka tespit edildi")

            else:
                messagebox.showwarning("Uyarı", "Plaka tespit edilemedi!")
                self.status_label.config(text="Plaka bulunamadı")

        except Exception as e:
            messagebox.showerror("Hata", f"Tespit hatası: {str(e)}")

    def detect_and_read_plate(self):
        """Plaka tespit + okuma akışı"""
        self.show_plate()
        if self.cropped_plate is not None:
            self.read_plate()

    def read_plate(self):
        """Model ile plakayı oku"""
        if self.cropped_plate is None:
            messagebox.showwarning("Uyarı", "Önce plakayı tespit edin!")
            return

        try:
            self.status_label.config(text="Plaka okunuyor...")
            self.root.update()

            start_total = time.time()

            # Model sonuçlarını al
            _, _, model_data = self.detector.read_plate_text(
                self.cropped_plate,
                image_is_rgb=not self.images_are_bgr,
                full_image=self.original_image,
                coords=self.last_coords,
                use_full_image_for_char=False,
                debug_dir="results/debug_char"
            )
            _ = time.time() - start_total

            if not self.detector.use_character_model:
                self.plate_result_value.config(text="MODEL KAPALI", fg=self.colors["muted"])
                self.plate_result_status.config(text="○", fg=self.colors["muted"])
            elif model_data and model_data.get("success"):
                self.plate_result_value.config(text=model_data.get("text", "--"), fg=self.colors["text"])
                self.plate_result_status.config(text="✔", fg=self.colors["success"])
            else:
                self.plate_result_value.config(text="OKUMA BAŞARISIZ", fg=self.colors["warning"])
                self.plate_result_status.config(text="!", fg=self.colors["warning"])

            self.status_label.config(text="Okuma tamamlandı")

        except Exception as e:
            messagebox.showerror("Hata", f"Okuma hatası: {str(e)}")
            self.status_label.config(text="Okuma başarısız")

    def reset_results(self):
        """Sonuçları sıfırla"""
        self.plate_result_value.config(text="--", fg=self.colors["text"])
        self.plate_result_status.config(text="○", fg=self.colors["muted"])

    def reset_container_results(self):
        """Konteyner sonuçlarını sıfırla"""
        self.container_result_value.config(text="--", fg=self.colors["text"])
        self.container_result_sub.config(text="", fg=self.colors["muted"])
        self.container_result_status.config(text="○", fg=self.colors["muted"])
        self.seal_result_value.config(text="--", fg=self.colors["muted"])
        self.damage_result_value.config(text="--", fg=self.colors["muted"])
        self.damage_result_sub.config(text="", fg=self.colors["muted"])
        self._set_overlays([])

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

    def find_container_iso(self):
        """Konteyner ISO alanını tespit eder"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce resim veya video yükleyin!")
            return

        if not self.container_detector.model:
            messagebox.showerror("Hata", "Konteyner modeli yüklü değil!")
            return

        try:
            self.status_label.config(text="Konteyner ISO tespit ediliyor...")
            self.root.update()

            result = self.container_detector.detect_container_number_in_image(
                self.original_image, read_text=False
            )

            if result['success'] and result['coordinates']:
                coords = result.get("coordinates") or []
                confs = result.get("confidence") or []
                area_ratios = result.get("area_ratio") or []

                candidates = []
                for idx, bbox in enumerate(coords):
                    conf = confs[idx] if idx < len(confs) else 0.0
                    area_ratio = area_ratios[idx] if idx < len(area_ratios) else None
                    if area_ratio is None:
                        size_score = 0.0
                    else:
                        size_score = min(1.0, float(area_ratio) / 0.02)
                    det_score = 0.7 * float(conf) + 0.3 * size_score
                    candidates.append({
                        "bbox": bbox,
                        "conf": conf,
                        "area_ratio": area_ratio,
                        "det_score": det_score,
                    })

                candidates.sort(key=lambda c: c["det_score"], reverse=True)

                best = None
                best_tier = -1
                best_score = -1.0
                best_text = ""
                best_text_conf = 0.0
                best_model_data = None

                max_try = min(4, len(candidates))
                for cand in candidates[:max_try]:
                    x1, y1, x2, y2 = cand["bbox"]
                    crop = self.original_image[y1:y2, x1:x2]
                    text, text_conf, model_data = self.container_detector.read_number_from_crop(
                        crop,
                        image_is_rgb=not self.images_are_bgr,
                        full_image=self.original_image,
                        coords=(x1, y1, x2, y2),
                    )
                    check_ok = model_data.get("check_digit_ok") if model_data else None

                    if text and check_ok is True:
                        tier = 2
                        score = float(text_conf)
                    elif text:
                        tier = 1
                        score = 0.6 * float(text_conf) + 0.4 * float(cand["det_score"])
                    else:
                        tier = 0
                        score = float(cand["det_score"])

                    if tier > best_tier or (tier == best_tier and score > best_score):
                        best = cand
                        best_tier = tier
                        best_score = score
                        best_text = text
                        best_text_conf = text_conf
                        best_model_data = model_data

                if best is None:
                    best = candidates[0]

                x1, y1, x2, y2 = best["bbox"]
                conf = best["conf"]

                self.container_crop = self.original_image[y1:y2, x1:x2]
                self.container_coords = (x1, y1, x2, y2)
                self.container_read_btn.config(state=tk.NORMAL)

                if best_text:
                    display_text = best_model_data.get("formatted") if best_model_data else None
                    if not display_text:
                        display_text = best_text
                    self.container_result_value.config(text=display_text, fg=self.colors["text"])
                    check_ok = best_model_data.get("check_digit_ok") if best_model_data else None
                    if check_ok is True:
                        self.container_result_sub.config(text="✔ Valid Check Digit", fg=self.colors["success"])
                        self.container_result_status.config(text="✔", fg=self.colors["success"])
                    elif check_ok is False:
                        self.container_result_sub.config(text="✖ Check Digit Hatalı", fg=self.colors["warning"])
                        self.container_result_status.config(text="!", fg=self.colors["warning"])
                    else:
                        self.container_result_sub.config(
                            text=f"Güven: {max(conf, best_text_conf):.1%}", fg=self.colors["muted"]
                        )
                        self.container_result_status.config(text="✔", fg=self.colors["success"])
                    self.container_result_value.config(text=display_text, fg=self.colors["text"])
                else:
                    self.container_result_value.config(text="OKUMA BAŞARISIZ", fg=self.colors["warning"])
                    self.container_result_sub.config(text=f"Güven: {conf:.1%}", fg=self.colors["muted"])
                    self.container_result_status.config(text="!", fg=self.colors["warning"])

                self._set_overlays([{
                    "bbox": (x1, y1, x2, y2),
                    "label": "Container ISO",
                    "color": (255, 140, 40),
                }])
                self.display_image(self.original_image, self.original_canvas, bgr_to_rgb=True)
                self.status_label.config(text="Konteyner ISO alanı bulundu ve okundu")
            else:
                messagebox.showwarning("Uyarı", "Konteyner ISO alanı bulunamadı!")
                self.status_label.config(text="Konteyner ISO alanı yok")

        except Exception as e:
            messagebox.showerror("Hata", f"Konteyner tespit hatası: {str(e)}")

    def read_container_iso(self):
        """Konteyner ISO numarasını okur"""
        if self.container_crop is None:
            messagebox.showwarning("Uyarı", "Önce konteyner ISO alanını tespit edin!")
            return

        try:
            self.status_label.config(text="Konteyner ISO okunuyor...")
            self.root.update()

            text, conf, model_data = self.container_detector.read_number_from_crop(
                self.container_crop,
                image_is_rgb=not self.images_are_bgr,
                full_image=self.original_image,
                coords=self.container_coords,
            )

            if text:
                display_text = model_data.get("formatted") if model_data else None
                if not display_text:
                    display_text = text
                self.container_result_value.config(text=display_text, fg=self.colors["text"])
                check_ok = model_data.get("check_digit_ok") if model_data else None
                if check_ok is True:
                    self.container_result_sub.config(text="✔ Valid Check Digit", fg=self.colors["success"])
                    self.container_result_status.config(text="✔", fg=self.colors["success"])
                elif check_ok is False:
                    self.container_result_sub.config(text="✖ Check Digit Hatalı", fg=self.colors["warning"])
                    self.container_result_status.config(text="!", fg=self.colors["warning"])
                else:
                    self.container_result_sub.config(text=f"Güven: {conf:.1%}", fg=self.colors["muted"])
                    self.container_result_status.config(text="✔", fg=self.colors["success"])
            else:
                self.container_result_value.config(text="OKUMA BAŞARISIZ", fg=self.colors["warning"])
                self.container_result_sub.config(text="", fg=self.colors["muted"])
                self.container_result_status.config(text="!", fg=self.colors["warning"])

            self.status_label.config(text="Konteyner ISO okuma tamamlandı")

        except Exception as e:
            messagebox.showerror("Hata", f"Konteyner okuma hatası: {str(e)}")
            self.status_label.config(text="Konteyner ISO okuma başarısız")

    def check_container_seal(self):
        """Konteyner mühür var/yok kontrolü"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce resim veya video yükleyin!")
            return

        if not self.seal_detector.model:
            messagebox.showerror("Hata", "Mühür modeli yüklü değil!")
            return

        try:
            self.status_label.config(text="Mühür kontrol ediliyor...")
            self.root.update()

            result = self.seal_detector.detect_seal(self.original_image)
            if not result.get("success"):
                err = result.get("error") or "Bilinmeyen hata"
                self.seal_result_value.config(text=f"YOK ({err})", fg=self.colors["warning"])
                self.status_label.config(text=f"Mühür kontrol başarısız: {err}")
                self.log_message(f"Mühür kontrol başarısız: {err}")
                return

            present = result.get("present")
            conf = result.get("confidence", 0.0)
            low_conf = result.get("low_confidence", False)

            if low_conf or present is None:
                self.seal_result_value.config(text=f"DÜŞÜK GÜVEN ({conf:.0%})", fg=self.colors["warning"])
            elif present:
                self.seal_result_value.config(text="MEVCUT (OK)", fg=self.colors["success"])
            else:
                self.seal_result_value.config(text="YOK", fg=self.colors["warning"])

            boxes = result.get("boxes") or []
            if boxes:
                best = max(boxes, key=lambda b: b.get("confidence", 0.0))
                bbox = best.get("bbox")
                if bbox:
                    self._set_overlays([{
                        "bbox": bbox,
                        "label": "Seal",
                        "color": (0, 215, 255) if present else (40, 120, 255),
                    }])
                    self.display_image(self.original_image, self.original_canvas, bgr_to_rgb=True)

            self.status_label.config(text=f"Mühür kontrol tamamlandı ({conf:.0%})")
            if low_conf or present is None:
                self.log_message("Mühür kontrol: DÜŞÜK GÜVEN")
            else:
                self.log_message(f"Mühür kontrol: {'MEVCUT' if present else 'YOK'}")

        except Exception as e:
            messagebox.showerror("Hata", f"Mühür kontrol hatası: {str(e)}")
            self.status_label.config(text="Mühür kontrol başarısız")

    def check_container_damage(self):
        """Konteyner hasar kontrolü ve hasar cinsi raporu."""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce resim veya video yükleyin!")
            return

        if not self.damage_detector.model:
            messagebox.showerror("Hata", "Hasar modeli yüklü değil!")
            return

        def _fmt_damage_type(name):
            text = str(name or "").strip()
            if not text:
                return ""
            key = text.replace("-", "_").replace(" ", "_").lower()
            tr_names = {
                "dent": "EZIK",
                "scratch": "CIZIK",
                "crack": "CATLAK",
                "corrosion": "KOROZYON",
                "deformation": "DEFORMASYON",
                "breakage": "KIRILMA",
                "broken": "KIRIK",
                "hole": "DELIK",
                "tear": "YIRTIK",
                "paint_damage": "BOYA HASARI",
                "damage": "HASAR",
                "no_damage": "HASAR YOK",
            }
            if key in tr_names:
                return tr_names[key]
            return key.replace("_", " ").upper()

        try:
            self.status_label.config(text="Hasar kontrol ediliyor...")
            self.root.update()

            result = self.damage_detector.detect_damage(self.original_image)
            if not result.get("success"):
                err = result.get("error") or "Bilinmeyen hata"
                self.damage_result_value.config(text=f"HATA ({err})", fg=self.colors["warning"])
                self.damage_result_sub.config(text="", fg=self.colors["muted"])
                self.status_label.config(text=f"Hasar kontrol başarısız: {err}")
                self.log_message(f"Hasar kontrol başarısız: {err}")
                return

            has_damage = result.get("has_damage")
            conf = float(result.get("confidence", 0.0) or 0.0)
            low_conf = bool(result.get("low_confidence", False))
            threshold_pct = int(round(float(self.damage_detector.min_damage_conf) * 100))

            damage_types = result.get("damage_types") or []
            primary_damage = result.get("damage_type")
            if not damage_types and primary_damage:
                damage_types = [primary_damage]
            formatted_types = [_fmt_damage_type(t) for t in damage_types if t]

            if low_conf or has_damage is None:
                self.damage_result_value.config(text=f"DÜŞÜK GÜVEN ({conf:.0%})", fg=self.colors["warning"])
                self.damage_result_sub.config(
                    text=f"Eşik: %{threshold_pct} altı, sonuç bastırıldı",
                    fg=self.colors["muted"],
                )
            elif has_damage:
                type_text = ", ".join(formatted_types[:3]) if formatted_types else "HASAR"
                self.damage_result_value.config(text=f"HASAR VAR ({type_text})", fg=self.colors["danger"])
                self.damage_result_sub.config(text=f"Güven: {conf:.0%}", fg=self.colors["warning"])
            else:
                self.damage_result_value.config(text="HASAR YOK", fg=self.colors["success"])
                self.damage_result_sub.config(text=f"Güven: {conf:.0%}", fg=self.colors["muted"])

            self._set_overlays([])
            boxes = result.get("boxes") or []
            if boxes and not (low_conf or has_damage is None):
                overlays = []
                sorted_boxes = sorted(boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)
                for box in sorted_boxes:
                    bbox = box.get("bbox")
                    if not bbox:
                        continue

                    d_type = str(box.get("damage_type") or box.get("class_name") or "").lower()
                    if has_damage is True and d_type == "no_damage":
                        continue
                    if has_damage is False and d_type != "no_damage":
                        continue

                    box_conf = float(box.get("confidence", 0.0) or 0.0)
                    label_type = _fmt_damage_type(d_type) or "DAMAGE"
                    overlays.append({
                        "bbox": bbox,
                        "label": f"{label_type} {box_conf:.2f}",
                        "color": (35, 90, 235) if d_type != "no_damage" else (35, 180, 85),
                    })
                    if len(overlays) >= 4:
                        break

                if overlays:
                    self._set_overlays(overlays)
                    self.display_image(self.original_image, self.original_canvas, bgr_to_rgb=True)

            self.status_label.config(text=f"Hasar kontrol tamamlandı ({conf:.0%})")
            if low_conf or has_damage is None:
                self.log_message(f"Hasar kontrol: DÜŞÜK GÜVEN (Eşik %{threshold_pct})")
            elif has_damage:
                shown = ", ".join(formatted_types[:3]) if formatted_types else "HASAR"
                self.log_message(f"Hasar kontrol: HASAR VAR ({shown})")
            else:
                self.log_message("Hasar kontrol: HASAR YOK")

        except Exception as e:
            messagebox.showerror("Hata", f"Hasar kontrol hatası: {str(e)}")
            self.status_label.config(text="Hasar kontrol başarısız")

    def display_image(self, image, canvas, bgr_to_rgb=False):
        """Resmi canvas'a göster"""
        if image is None:
            return
        if canvas == self.original_canvas:
            image = self._draw_overlays(image)
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

    def _resize_to_canvas(self, image, canvas, bgr_to_rgb=False):
        """Canvas boyutuna ölçeklenmiş PIL image döner."""
        if image is None:
            return None
        if bgr_to_rgb and len(image.shape) == 3 and image.shape[2] == 3:
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception:
                pass
        canvas.update()
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 2 or h < 2:
            return None
        pil_img = Image.fromarray(image)
        scale = min(w / pil_img.width, h / pil_img.height)
        new_w, new_h = int(pil_img.width * scale * 0.95), int(pil_img.height * scale * 0.95)
        return pil_img.resize((new_w, new_h), Image.LANCZOS)


def main():
    root = tk.Tk()
    app = PlakaTespitUygulamasi(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    
