"""
SANKO Port - Ana Dosya
Tkinter GUI ile plaka tespit ve model tabanlı okuma uygulaması
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
from urllib.parse import unquote, urlparse
from PIL import Image, ImageTk
import cv2
from .detector import PlakaDetector
from .container_detector import ContainerNumberDetector
from .seal_detector import ContainerSealDetector
from .damage_detector import ContainerDamageDetector
import time

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = None
    TkinterDnD = None


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
        self.damage_detector = ContainerDamageDetector(min_damage_conf=0.52, decision_margin=0.08)
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
        self.image_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
        }
        self.video_extensions = {
            ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"
        }
        self.dragdrop_enabled = False

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

        self.video_btn = styled_button(btn_row, "Video Yükle", self.load_video, "#3a4454")
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

        self._setup_drag_drop()
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

    def _center_popup(self, window, width=1120, height=720):
        """Pencereyi ekran merkezine konumlandır."""
        window.update_idletasks()
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")

    @staticmethod
    def _format_file_size(size_bytes):
        """Byte değerini okunabilir boyuta çevir."""
        size = float(max(0, int(size_bytes)))
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return "0 B"

    def _reset_media_state(self):
        """Yeni medya yüklenirken geçici tespit durumlarını sıfırlar."""
        self.cropped_plate = None
        self.last_coords = None
        self.container_crop = None
        self.container_coords = None
        self.video_stable_text = None
        self.video_stable_count = 0
        self.video_last_shown = None
        self.plate_canvas.delete("all")
        self.reset_results()
        self.reset_container_results()
        self._set_overlays([])

    def _detect_media_type(self, file_path):
        """Dosya uzantısından medya tipini belirler."""
        ext = os.path.splitext(str(file_path))[1].lower()
        if ext in self.image_extensions:
            return "image"
        if ext in self.video_extensions:
            return "video"
        return None

    def _parse_dropped_paths(self, raw_data):
        """Sürükle-bırak event verisini dosya yol listesine çevirir."""
        if not raw_data:
            return []

        try:
            items = list(self.root.tk.splitlist(raw_data))
        except Exception:
            items = [raw_data]

        paths = []
        for item in items:
            candidate = str(item).strip()
            if not candidate:
                continue
            if candidate.startswith("{") and candidate.endswith("}"):
                candidate = candidate[1:-1].strip()
            if candidate.startswith("file://"):
                parsed = urlparse(candidate)
                candidate = unquote(parsed.path or "")
                if os.name == "nt" and candidate.startswith("/") and len(candidate) > 2 and candidate[2] == ":":
                    candidate = candidate[1:]
            candidate = os.path.normpath(os.path.expanduser(candidate))
            if candidate:
                paths.append(candidate)
        return paths

    def _on_drop_files(self, event):
        """Sürüklenen dosyayı algılar ve uygun akışı başlatır."""
        dropped = self._parse_dropped_paths(getattr(event, "data", ""))
        if not dropped:
            return

        for path in dropped:
            if not os.path.isfile(path):
                continue
            media_type = self._detect_media_type(path)
            if media_type == "image":
                self._load_image_from_path(path)
                return
            if media_type == "video":
                self._load_video_from_path(path)
                return

        messagebox.showwarning(
            "Uyarı",
            "Bırakılan dosyalar arasında desteklenen görüntü/video bulunamadı."
        )

    def _setup_drag_drop(self):
        """Destek varsa sürükle-bırak olaylarını bağlar."""
        self.dragdrop_enabled = False
        if DND_FILES is None:
            self.log_message("Sürükle-bırak pasif (tkinterdnd2 kurulu değil)")
            return

        targets = [self.root, self.original_canvas]
        for widget in targets:
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop_files)
                self.dragdrop_enabled = True
            except Exception:
                continue

        if self.dragdrop_enabled:
            self.log_message("Sürükle-bırak aktif")
            self.status_hint.config(text="Dosyayı sürükleyip bırakabilirsiniz")
        else:
            self.log_message("Sürükle-bırak başlatılamadı")

    def _load_image_from_path(self, file_path):
        """Verilen yoldan görüntüyü yükler."""
        if not file_path:
            return False
        try:
            if self.video_cap:
                self.stop_video()

            image = cv2.imread(file_path)
            if image is None or getattr(image, "size", 0) == 0:
                raise ValueError("Seçilen dosya geçerli bir görsel değil")

            self._reset_media_state()
            self.current_image_path = file_path
            self.video_path = None
            self.original_image = image

            self.display_image(image, self.original_canvas, bgr_to_rgb=True)
            self.detect_read_btn.config(state=tk.NORMAL)
            self.container_find_btn.config(state=tk.NORMAL)
            self.container_read_btn.config(state=tk.DISABLED)
            self.seal_btn.config(state=tk.NORMAL)
            self.damage_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Resim yüklendi")
            self.status_hint.config(text="Görüntü hazır")
            self.log_message("Görüntü yüklendi")
            return True
        except Exception as err:
            messagebox.showerror("Hata", f"Resim yüklenemedi: {str(err)}")
            return False

    def _load_video_from_path(self, file_path):
        """Verilen yoldan videoyu yükler."""
        if not file_path:
            return False
        if self.video_cap:
            self.stop_video()

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Video açılamadı")
            return False

        ok, frame = cap.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            try:
                cap.release()
            except Exception:
                pass
            messagebox.showerror("Hata", "Video ilk kare okunamadı")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_delay_ms = int(1000 / fps) if fps and fps > 1 else 33
        if fps and fps > 1:
            every_n = int(round((fps * self.video_stable_target_s) / self.video_stable_required))
            self.video_every_n = max(1, min(10, every_n))

        self._reset_media_state()
        self.current_image_path = None
        self.video_path = file_path
        self.video_cap = cap
        self.video_running = True
        self.video_frame_index = 0
        self.play_btn.config(state=tk.NORMAL, text="Durdur")

        self.original_image = frame
        self.display_image(frame, self.original_canvas, bgr_to_rgb=True)
        self.status_label.config(text="Video yüklendi")
        self.status_hint.config(text="Video akışı başladı")
        self.detect_read_btn.config(state=tk.NORMAL)
        self.container_find_btn.config(state=tk.NORMAL)
        self.container_read_btn.config(state=tk.DISABLED)
        self.seal_btn.config(state=tk.NORMAL)
        self.damage_btn.config(state=tk.NORMAL)
        self.update_video_frame()
        self.log_message("Video yüklendi ve akış başladı")
        return True

    def _open_media_picker(self, media_type="image"):
        """Daha büyük ve önizlemeli özel medya seçici açar."""
        is_image = str(media_type).lower() == "image"
        allowed_ext = self.image_extensions if is_image else self.video_extensions
        picker_title = "Görüntü Seç" if is_image else "Video Seç"
        picker_subtitle = (
            "Dosya listesi solda, önizleme ve detaylar sağda."
            if is_image
            else "Video dosyanızı seçin, ilk kare önizlemesi sağda gösterilir."
        )

        base_path = self.current_image_path if is_image else self.video_path
        initial_dir = os.path.dirname(base_path) if base_path else os.getcwd()
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()

        dialog = tk.Toplevel(self.root)
        dialog.title(picker_title)
        dialog.configure(bg=self.colors["bg"])
        dialog.minsize(980, 620)
        self._center_popup(dialog, width=1120, height=720)
        dialog.transient(self.root)
        dialog.grab_set()

        result = {"path": None}
        state = {
            "current_dir": initial_dir,
            "items": [],
            "selected_path": None,
            "selected_is_dir": False,
            "preview_photo": None,
        }

        header = tk.Frame(dialog, bg=self.colors["primary"], height=76)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        header_inner = tk.Frame(header, bg=self.colors["primary"])
        header_inner.pack(fill=tk.BOTH, expand=True, padx=18, pady=10)

        tk.Label(
            header_inner,
            text=picker_title,
            font=self.fonts["title"],
            bg=self.colors["primary"],
            fg=self.colors["text"],
        ).pack(anchor="w")
        tk.Label(
            header_inner,
            text=picker_subtitle,
            font=self.fonts["body"],
            bg=self.colors["primary"],
            fg=self.colors["muted"],
        ).pack(anchor="w")

        body = tk.Frame(dialog, bg=self.colors["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        left = tk.Frame(body, bg=self.colors["bg"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right = tk.Frame(
            body,
            bg=self.colors["surface"],
            width=360,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        right.pack(side=tk.LEFT, fill=tk.BOTH)
        right.pack_propagate(False)

        nav_card = tk.Frame(
            left,
            bg=self.colors["surface"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        nav_card.pack(fill=tk.X, pady=(0, 10))

        nav_row = tk.Frame(nav_card, bg=self.colors["surface"])
        nav_row.pack(fill=tk.X, padx=12, pady=(10, 6))

        path_var = tk.StringVar(value=state["current_dir"])
        path_entry = tk.Entry(
            nav_row,
            textvariable=path_var,
            font=self.fonts["body"],
            bg="#11161d",
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief=tk.FLAT,
            disabledbackground="#11161d",
            disabledforeground=self.colors["muted"],
        )
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        path_entry.config(state=tk.DISABLED)

        search_row = tk.Frame(nav_card, bg=self.colors["surface"])
        search_row.pack(fill=tk.X, padx=12, pady=(0, 10))

        tk.Label(
            search_row,
            text="Filtre:",
            font=self.fonts["body"],
            bg=self.colors["surface"],
            fg=self.colors["muted"],
        ).pack(side=tk.LEFT, padx=(0, 8))

        search_var = tk.StringVar()
        search_entry = tk.Entry(
            search_row,
            textvariable=search_var,
            font=self.fonts["body"],
            bg="#11161d",
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief=tk.FLAT,
        )
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        list_card = tk.Frame(
            left,
            bg=self.colors["surface"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        list_card.pack(fill=tk.BOTH, expand=True)

        list_header = tk.Frame(list_card, bg=self.colors["surface"])
        list_header.pack(fill=tk.X, padx=12, pady=(10, 0))
        tk.Label(
            list_header,
            text="Dosyalar",
            font=self.fonts["subtitle"],
            bg=self.colors["surface"],
            fg=self.colors["text"],
        ).pack(side=tk.LEFT)
        count_label = tk.Label(
            list_header,
            text="",
            font=self.fonts["body"],
            bg=self.colors["surface"],
            fg=self.colors["muted"],
        )
        count_label.pack(side=tk.RIGHT)

        list_body = tk.Frame(list_card, bg=self.colors["surface"])
        list_body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 12))

        list_scroll = tk.Scrollbar(list_body)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        file_list = tk.Listbox(
            list_body,
            font=self.fonts["body"],
            bg="#11161d",
            fg=self.colors["text"],
            selectbackground=self.colors["accent"],
            selectforeground="white",
            activestyle="none",
            relief=tk.FLAT,
            yscrollcommand=list_scroll.set,
        )
        file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=file_list.yview)

        preview_header = tk.Frame(right, bg=self.colors["surface"])
        preview_header.pack(fill=tk.X, padx=12, pady=(10, 0))
        tk.Label(
            preview_header,
            text="Önizleme",
            font=self.fonts["subtitle"],
            bg=self.colors["surface"],
            fg=self.colors["text"],
        ).pack(anchor="w")

        preview_canvas = tk.Canvas(
            right,
            bg="#0f141a",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        preview_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 8))

        preview_name = tk.Label(
            right,
            text="Dosya seçilmedi",
            font=self.fonts["body"],
            bg=self.colors["surface"],
            fg=self.colors["text"],
            anchor="w",
            justify=tk.LEFT,
            wraplength=320,
        )
        preview_name.pack(fill=tk.X, padx=12)

        preview_meta = tk.Label(
            right,
            text="",
            font=self.fonts["mono"],
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            anchor="w",
            justify=tk.LEFT,
            wraplength=320,
        )
        preview_meta.pack(fill=tk.X, padx=12, pady=(6, 12))

        footer = tk.Frame(dialog, bg=self.colors["primary"], height=68)
        footer.pack(fill=tk.X)
        footer.pack_propagate(False)

        footer_inner = tk.Frame(footer, bg=self.colors["primary"])
        footer_inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        selected_var = tk.StringVar(value="Seçili dosya: yok")
        selected_label = tk.Label(
            footer_inner,
            textvariable=selected_var,
            font=self.fonts["body"],
            bg=self.colors["primary"],
            fg=self.colors["muted"],
            anchor="w",
        )
        selected_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def close_dialog():
            try:
                dialog.grab_release()
            except Exception:
                pass
            dialog.destroy()

        def choose_file():
            if state["selected_path"] and not state["selected_is_dir"]:
                result["path"] = state["selected_path"]
                close_dialog()
            else:
                messagebox.showwarning("Uyarı", "Lütfen bir dosya seçin.", parent=dialog)

        select_btn = tk.Button(
            footer_inner,
            text="Seç",
            command=choose_file,
            font=self.fonts["body"],
            bg=self.colors["accent"],
            fg="white",
            padx=16,
            pady=8,
            relief=tk.FLAT,
            activebackground=self.colors["accent"],
            activeforeground="white",
            state=tk.DISABLED,
            cursor="hand2",
        )
        select_btn.pack(side=tk.RIGHT, padx=(8, 0))

        tk.Button(
            footer_inner,
            text="İptal",
            command=close_dialog,
            font=self.fonts["body"],
            bg="#3a4454",
            fg="white",
            padx=16,
            pady=8,
            relief=tk.FLAT,
            activebackground="#3a4454",
            activeforeground="white",
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        def draw_placeholder(title_text, sub_text):
            preview_canvas.update_idletasks()
            cw = max(10, preview_canvas.winfo_width())
            ch = max(10, preview_canvas.winfo_height())
            preview_canvas.delete("all")
            preview_canvas.create_text(
                cw // 2,
                ch // 2 - 10,
                text=title_text,
                fill=self.colors["muted"],
                font=self.fonts["subtitle"],
            )
            preview_canvas.create_text(
                cw // 2,
                ch // 2 + 14,
                text=sub_text,
                fill=self.colors["muted"],
                font=self.fonts["body"],
            )
            state["preview_photo"] = None
            preview_canvas.image = None

        def draw_pil_image(pil_image):
            preview_canvas.update_idletasks()
            cw = max(60, preview_canvas.winfo_width())
            ch = max(60, preview_canvas.winfo_height())
            max_w = max(40, cw - 24)
            max_h = max(40, ch - 24)
            scale = min(max_w / pil_image.width, max_h / pil_image.height, 1.0)
            new_w = max(1, int(pil_image.width * scale))
            new_h = max(1, int(pil_image.height * scale))
            resized = pil_image.resize((new_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            preview_canvas.delete("all")
            preview_canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
            preview_canvas.image = photo
            state["preview_photo"] = photo

        def update_preview(path=None, is_dir=False):
            if not path:
                preview_name.config(text="Dosya seçilmedi")
                preview_meta.config(text="")
                draw_placeholder(
                    "Önizleme hazır",
                    "Soldan bir dosya seçin",
                )
                return

            base_name = os.path.basename(path) or path
            preview_name.config(text=base_name)

            if is_dir:
                preview_meta.config(text="Klasör • Çift tıklayarak açabilirsiniz")
                draw_placeholder("Klasör", "İçeriğe girmek için çift tıklayın")
                return

            try:
                size_text = self._format_file_size(os.path.getsize(path))
            except OSError:
                size_text = "Boyut bilinmiyor"

            ext = os.path.splitext(path)[1].lower()
            if is_image:
                try:
                    with Image.open(path) as img:
                        rgb_img = img.convert("RGB")
                        w, h = rgb_img.size
                        preview_meta.config(text=f"{w} x {h} px • {size_text}")
                        draw_pil_image(rgb_img)
                except Exception:
                    preview_meta.config(text=f"Görsel açılamadı • {size_text}")
                    draw_placeholder("Önizleme yok", "Bu dosya gösterilemiyor")
            else:
                cap = cv2.VideoCapture(path)
                ok = False
                frame = None
                fps = cap.get(cv2.CAP_PROP_FPS) if cap else 0.0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap else 0.0
                try:
                    if cap and cap.isOpened():
                        ok, frame = cap.read()
                finally:
                    if cap:
                        cap.release()

                duration_text = ""
                if fps and fps > 0 and frame_count and frame_count > 0:
                    duration_s = frame_count / fps
                    minutes = int(duration_s // 60)
                    seconds = int(duration_s % 60)
                    duration_text = f" • {minutes:02d}:{seconds:02d}"

                if ok and frame is not None:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb)
                        w, h = pil_img.size
                        preview_meta.config(
                            text=f"{w} x {h} px • {size_text}{duration_text} • {ext or 'video'}"
                        )
                        draw_pil_image(pil_img)
                    except Exception:
                        preview_meta.config(text=f"Video açılamadı • {size_text}")
                        draw_placeholder("Önizleme yok", "Video kareleri okunamıyor")
                else:
                    preview_meta.config(text=f"Video açılamadı • {size_text}")
                    draw_placeholder("Önizleme yok", "İlk kare okunamadı")

        def refresh_file_list():
            current_dir = state["current_dir"]
            path_var.set(current_dir)
            query = search_var.get().strip().lower()
            file_list.delete(0, tk.END)
            state["items"] = []
            state["selected_path"] = None
            state["selected_is_dir"] = False
            select_btn.config(state=tk.DISABLED)
            selected_var.set("Seçili dosya: yok")

            dirs = []
            files = []
            try:
                with os.scandir(current_dir) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir():
                                dirs.append(entry.path)
                            elif entry.is_file():
                                ext = os.path.splitext(entry.name)[1].lower()
                                if ext in allowed_ext:
                                    files.append(entry.path)
                        except OSError:
                            continue
            except OSError as err:
                messagebox.showerror("Hata", f"Klasör okunamadı:\n{err}", parent=dialog)
                return

            dirs.sort(key=lambda p: os.path.basename(p).lower())
            files.sort(key=lambda p: os.path.basename(p).lower())

            visible_dir_count = 0
            visible_file_count = 0
            for d in dirs:
                name = os.path.basename(d)
                if query and query not in name.lower():
                    continue
                file_list.insert(tk.END, f"[Klasör] {name}")
                state["items"].append({"path": d, "is_dir": True})
                visible_dir_count += 1

            for fpath in files:
                name = os.path.basename(fpath)
                if query and query not in name.lower():
                    continue
                file_list.insert(tk.END, name)
                state["items"].append({"path": fpath, "is_dir": False})
                visible_file_count += 1

            count_label.config(text=f"{visible_file_count} dosya • {visible_dir_count} klasör")

            if state["items"]:
                default_idx = 0
                for idx, item in enumerate(state["items"]):
                    if not item["is_dir"]:
                        default_idx = idx
                        break
                file_list.selection_set(default_idx)
                file_list.see(default_idx)
                file_list.event_generate("<<ListboxSelect>>")
            else:
                update_preview(None)
                preview_meta.config(text="Uygun dosya bulunamadı")
                draw_placeholder("Dosya yok", "Klasör değiştirin veya filtreyi temizleyin")

        def go_up():
            parent_dir = os.path.dirname(state["current_dir"])
            if parent_dir and parent_dir != state["current_dir"]:
                state["current_dir"] = parent_dir
                refresh_file_list()

        def open_folder():
            chosen = filedialog.askdirectory(
                parent=dialog,
                initialdir=state["current_dir"],
                title="Klasör Seç",
            )
            if chosen:
                state["current_dir"] = chosen
                refresh_file_list()

        nav_buttons = tk.Frame(nav_row, bg=self.colors["surface"])
        nav_buttons.pack(side=tk.RIGHT, padx=(10, 0))

        tk.Button(
            nav_buttons,
            text="Yukarı",
            command=go_up,
            font=self.fonts["body"],
            bg="#2d3642",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            pady=5,
            cursor="hand2",
            activebackground="#2d3642",
            activeforeground="white",
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            nav_buttons,
            text="Klasör Aç",
            command=open_folder,
            font=self.fonts["body"],
            bg="#2d3642",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            pady=5,
            cursor="hand2",
            activebackground="#2d3642",
            activeforeground="white",
        ).pack(side=tk.LEFT)

        def on_select(_event=None):
            selection = file_list.curselection()
            if not selection:
                state["selected_path"] = None
                state["selected_is_dir"] = False
                select_btn.config(state=tk.DISABLED)
                selected_var.set("Seçili dosya: yok")
                update_preview(None)
                return

            idx = int(selection[0])
            if idx < 0 or idx >= len(state["items"]):
                return
            selected = state["items"][idx]
            state["selected_path"] = selected["path"]
            state["selected_is_dir"] = bool(selected["is_dir"])

            if state["selected_is_dir"]:
                selected_var.set(f"Seçili klasör: {selected['path']}")
                select_btn.config(state=tk.DISABLED)
            else:
                selected_var.set(f"Seçili dosya: {selected['path']}")
                select_btn.config(state=tk.NORMAL)
            update_preview(selected["path"], is_dir=state["selected_is_dir"])

        def on_double_click(_event=None):
            selection = file_list.curselection()
            if not selection:
                return
            idx = int(selection[0])
            if idx < 0 or idx >= len(state["items"]):
                return

            selected = state["items"][idx]
            if selected["is_dir"]:
                state["current_dir"] = selected["path"]
                refresh_file_list()
            else:
                result["path"] = selected["path"]
                close_dialog()

        file_list.bind("<<ListboxSelect>>", on_select)
        file_list.bind("<Double-Button-1>", on_double_click)
        file_list.bind("<Return>", on_double_click)
        search_var.trace_add("write", lambda *_: refresh_file_list())

        dialog.bind("<Escape>", lambda _e: close_dialog())
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

        update_preview(None)
        refresh_file_list()
        search_entry.focus_set()

        self.root.wait_window(dialog)
        return result["path"]

    def load_image(self):
        """Resim yükleme"""
        file_path = self._open_media_picker(media_type="image")
        if file_path:
            self._load_image_from_path(file_path)

    def load_video(self):
        """Video yükleme"""
        file_path = self._open_media_picker(media_type="video")
        if not file_path:
            return
        self._load_video_from_path(file_path)

    def toggle_video(self):
        """Video oynat/durdur"""
        if not self.video_cap:
            return
        self.video_running = not self.video_running
        self.play_btn.config(text="Durdur" if self.video_running else "Oynat")
        self.status_label.config(text="Video oynatılıyor" if self.video_running else "Video durduruldu")
        self.log_message("Video oynatıldı" if self.video_running else "Video durduruldu")
        if self.video_running:
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            current_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if total_frames > 0 and current_pos >= max(0, total_frames - 1):
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.video_frame_index = 0
                self.video_stable_text = None
                self.video_stable_count = 0
                self.video_last_shown = None
                self.status_hint.config(text="Video başa sarıldı")
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
        self._set_overlays([])
        self.display_image(frame, self.original_canvas, bgr_to_rgb=True)

        # Video akışında plaka tespitini her N karede yap
        if self.detector.model:
            if self.video_frame_index % self.video_every_n == 0:
                result = self.detector.detect_plate_in_image(frame, read_text=False)
                if result['success'] and result['coordinates']:
                    coords = result.get("coordinates") or []
                    confs = result.get("confidence") or []
                    best_idx = max(
                        range(len(coords)),
                        key=lambda idx: (
                            float(confs[idx]) if idx < len(confs) else 0.0,
                            max(0, coords[idx][2] - coords[idx][0]) * max(0, coords[idx][3] - coords[idx][1]),
                        ),
                    )
                    x1, y1, x2, y2 = coords[best_idx]
                    self.cropped_plate = frame[y1:y2, x1:x2]
                    self.last_coords = (x1, y1, x2, y2)
                    self._set_overlays([{
                        "bbox": (x1, y1, x2, y2),
                        "label": "License Plate",
                        "color": (0, 200, 90),
                    }])
                    self.display_image(frame, self.original_canvas, bgr_to_rgb=True)
                    self.display_image(self.cropped_plate, self.plate_canvas, bgr_to_rgb=True)
                    self.detect_read_btn.config(state=tk.NORMAL)
                    if self.detector.use_character_model:
                        self.process_video_read()
                else:
                    self.cropped_plate = None
                    self.last_coords = None
                    self.video_stable_text = None
                    self.video_stable_count = 0
                    self.status_hint.config(text="Plaka aranıyor")

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
        self.cropped_plate = None
        self.last_coords = None

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
                coords = result.get("coordinates") or []
                confs = result.get("confidence") or []
                best_idx = max(
                    range(len(coords)),
                    key=lambda idx: (
                        float(confs[idx]) if idx < len(confs) else 0.0,
                        max(0, coords[idx][2] - coords[idx][0]) * max(0, coords[idx][3] - coords[idx][1]),
                    ),
                )
                x1, y1, x2, y2 = coords[best_idx]
                conf = float(confs[best_idx]) if best_idx < len(confs) else 0.0

                crop = self.original_image[y1:y2, x1:x2]
                if crop is None or getattr(crop, "size", 0) == 0:
                    messagebox.showwarning("Uyarı", "Plaka bölgesi çıkarılamadı.")
                    self._set_overlays([])
                    self.display_image(self.original_image, self.original_canvas, bgr_to_rgb=True)
                    self.status_label.config(text="Plaka bölgesi çıkarılamadı")
                    return

                self.cropped_plate = crop
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
                self._set_overlays([])
                if self.original_image is not None:
                    self.display_image(self.original_image, self.original_canvas, bgr_to_rgb=True)
                self.reset_results()
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
                self.container_crop = None
                self.container_coords = None
                self.container_read_btn.config(state=tk.DISABLED)
                self._set_overlays([])
                self.display_image(self.original_image, self.original_canvas, bgr_to_rgb=True)
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
            self._set_overlays([])

            result = self.seal_detector.detect_seal(self.original_image)
            if not result.get("success"):
                err = result.get("error") or "Bilinmeyen hata"
                self.seal_result_value.config(text=f"YOK ({err})", fg=self.colors["warning"])
                self.status_label.config(text=f"Mühür kontrol başarısız: {err}")
                self.display_image(self.original_image, self.original_canvas, bgr_to_rgb=True)
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

        def _clip_bbox(bbox, width, height):
            if not bbox:
                return None
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
            except Exception:
                return None
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)

        def _bbox_iou(a, b):
            if not a or not b:
                return 0.0
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = float(iw * ih)
            if inter <= 0:
                return 0.0
            area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
            area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
            union = area_a + area_b - inter
            if union <= 0:
                return 0.0
            return inter / union

        def _bbox_area_ratio(bbox, width, height):
            if not bbox:
                return 0.0
            x1, y1, x2, y2 = bbox
            return ((x2 - x1) * (y2 - y1)) / float(max(1, width * height))

        def _estimate_contour_container_roi(seed_bbox=None):
            if self.original_image is None:
                return None

            img_h, img_w = self.original_image.shape[:2]
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 45, 140)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            merged = cv2.dilate(merged, kernel, iterations=1)
            contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            seed_cx = seed_cy = None
            if seed_bbox:
                sx1, sy1, sx2, sy2 = seed_bbox
                seed_cx = (sx1 + sx2) / 2.0
                seed_cy = (sy1 + sy2) / 2.0

            img_cx = img_w / 2.0
            img_cy = img_h / 2.0
            img_diag = max(1.0, (img_w ** 2 + img_h ** 2) ** 0.5)
            best_bbox = None
            best_score = -1e9

            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                bbox = _clip_bbox((x, y, x + bw, y + bh), img_w, img_h)
                if not bbox:
                    continue

                area_ratio = _bbox_area_ratio(bbox, img_w, img_h)
                if area_ratio < 0.12 or area_ratio > 0.95:
                    continue

                aspect = max(bw, bh) / float(max(1, min(bw, bh)))
                if aspect < 1.15:
                    continue

                border_touch = (
                    (x <= 2)
                    + (y <= 2)
                    + (x + bw >= img_w - 2)
                    + (y + bh >= img_h - 2)
                )
                if border_touch >= 4:
                    continue

                fill_ratio = float(cv2.contourArea(cnt)) / float(max(1, bw * bh))
                if fill_ratio < 0.04:
                    continue

                score = area_ratio + (0.12 * aspect) - (0.05 * border_touch)

                if seed_bbox and seed_cx is not None and seed_cy is not None:
                    contains_seed = (x <= seed_cx <= x + bw) and (y <= seed_cy <= y + bh)
                    iou_seed = _bbox_iou(bbox, seed_bbox)
                    if not contains_seed and iou_seed < 0.02:
                        continue
                    score += 0.20 if contains_seed else 0.0
                    score += min(0.12, iou_seed * 0.5)
                else:
                    cx = x + (bw / 2.0)
                    cy = y + (bh / 2.0)
                    center_dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
                    center_bonus = max(0.0, 1.0 - (center_dist / img_diag))
                    score += 0.15 * center_bonus

                if score > best_score:
                    best_score = score
                    best_bbox = bbox

            return best_bbox

        def _estimate_damage_roi():
            if self.original_image is None:
                return None

            img_h, img_w = self.original_image.shape[:2]
            seed_bbox = _clip_bbox(self.container_coords, img_w, img_h)

            if seed_bbox is None and self.container_detector:
                try:
                    quick = self.container_detector.detect_container_number_in_image(
                        self.original_image,
                        conf_threshold=0.18,
                        low_conf_threshold=0.10,
                        read_text=False,
                    )
                    coords = quick.get("coordinates") or []
                    if coords:
                        scores = quick.get("scores") or []
                        confs = quick.get("confidence") or []

                        def _pick_score(idx):
                            score = float(scores[idx]) if idx < len(scores) else 0.0
                            conf = float(confs[idx]) if idx < len(confs) else 0.0
                            return (score, conf)

                        best_idx = max(range(len(coords)), key=_pick_score)
                        seed_bbox = _clip_bbox(coords[best_idx], img_w, img_h)
                        if seed_bbox:
                            self.container_coords = seed_bbox
                except Exception:
                    seed_bbox = None

            expanded_roi = None
            if seed_bbox is not None:
                x1, y1, x2, y2 = seed_bbox
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)

                # Conservative expansion around ISO region; previous version was too wide.
                pad_x = max(int(bw * 3.2), int(img_w * 0.06))
                pad_top = max(int(bh * 4.8), int(img_h * 0.08))
                pad_bottom = max(int(bh * 6.5), int(img_h * 0.16))
                if y2 < int(img_h * 0.40):
                    pad_bottom = max(pad_bottom, int(img_h * 0.52))
                expanded_roi = _clip_bbox(
                    (x1 - pad_x, y1 - pad_top, x2 + pad_x, y2 + pad_bottom),
                    img_w,
                    img_h,
                )

            contour_roi = _estimate_contour_container_roi(seed_bbox=seed_bbox)

            roi = None
            if expanded_roi and contour_roi:
                if _bbox_iou(expanded_roi, contour_roi) >= 0.03:
                    rx1 = max(expanded_roi[0], contour_roi[0])
                    ry1 = max(expanded_roi[1], contour_roi[1])
                    rx2 = min(expanded_roi[2], contour_roi[2])
                    ry2 = min(expanded_roi[3], contour_roi[3])
                    merged = _clip_bbox((rx1, ry1, rx2, ry2), img_w, img_h)
                    roi = merged if merged is not None else contour_roi
                else:
                    roi = expanded_roi if seed_bbox is not None else contour_roi
            elif expanded_roi:
                roi = expanded_roi
            elif contour_roi:
                roi = contour_roi

            if roi is None:
                return None

            roi_area_ratio = _bbox_area_ratio(roi, img_w, img_h)
            if roi_area_ratio < 0.10 or roi_area_ratio > 0.92:
                return None
            if seed_bbox is None:
                rx1, ry1, rx2, ry2 = roi
                width_ratio = (rx2 - rx1) / float(max(1, img_w))
                if roi_area_ratio < 0.25 and width_ratio < 0.40:
                    return None
            return roi

        try:
            self.status_label.config(text="Hasar kontrol ediliyor...")
            self.root.update()

            roi_bbox = _estimate_damage_roi()
            detect_image = self.original_image
            roi_offset = (0, 0)
            used_roi = False

            if roi_bbox is not None:
                rx1, ry1, rx2, ry2 = roi_bbox
                crop = self.original_image[ry1:ry2, rx1:rx2]
                if getattr(crop, "size", 0) > 0:
                    detect_image = crop
                    roi_offset = (rx1, ry1)
                    used_roi = True

            detect_conf = 0.25 if used_roi else 0.32
            result = self.damage_detector.detect_damage(detect_image, conf_threshold=detect_conf)
            if not result.get("success"):
                err = result.get("error") or "Bilinmeyen hata"
                self.damage_result_value.config(text=f"HATA ({err})", fg=self.colors["warning"])
                self.damage_result_sub.config(text="", fg=self.colors["muted"])
                self.status_label.config(text=f"Hasar kontrol başarısız: {err}")
                self.log_message(f"Hasar kontrol başarısız: {err}")
                return

            if used_roi:
                ox, oy = roi_offset
                img_h, img_w = self.original_image.shape[:2]
                shifted_boxes = []
                rx1, ry1, rx2, ry2 = roi_bbox
                roi_w = max(1, rx2 - rx1)
                roi_h = max(1, ry2 - ry1)
                core_margin_x = max(2, int(roi_w * 0.02))
                core_margin_y = max(2, int(roi_h * 0.02))
                core_roi = (
                    rx1 + core_margin_x,
                    ry1 + core_margin_y,
                    rx2 - core_margin_x,
                    ry2 - core_margin_y,
                )
                for box in (result.get("boxes") or []):
                    bbox = box.get("bbox")
                    if not bbox:
                        continue
                    clipped = _clip_bbox(
                        (bbox[0] + ox, bbox[1] + oy, bbox[2] + ox, bbox[3] + oy),
                        img_w,
                        img_h,
                    )
                    if not clipped:
                        continue

                    d_type = str(box.get("damage_type") or "").strip().lower()
                    is_generic = d_type in {"", "damage"}
                    bx1, by1, bx2, by2 = clipped
                    cx = (bx1 + bx2) // 2
                    cy = (by1 + by2) // 2
                    area_ratio = ((bx2 - bx1) * (by2 - by1)) / float(max(1, img_w * img_h))
                    near_roi_border = (
                        bx1 <= (rx1 + 1)
                        or by1 <= (ry1 + 1)
                        or bx2 >= (rx2 - 1)
                        or by2 >= (ry2 - 1)
                    )
                    if is_generic and near_roi_border and area_ratio < 0.02:
                        continue
                    if is_generic and area_ratio < 0.03:
                        cx1, cy1, cx2, cy2 = core_roi
                        if not (cx1 <= cx <= cx2 and cy1 <= cy <= cy2):
                            continue

                    item = dict(box)
                    item["bbox"] = clipped
                    shifted_boxes.append(item)
                result["boxes"] = shifted_boxes

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

            mode_text = "ROI" if used_roi else "FULL"
            self.status_label.config(text=f"Hasar kontrol tamamlandı ({conf:.0%}, {mode_text})")
            if low_conf or has_damage is None:
                self.log_message(f"Hasar kontrol: DÜŞÜK GÜVEN (Eşik %{threshold_pct}, {mode_text})")
            elif has_damage:
                shown = ", ".join(formatted_types[:3]) if formatted_types else "HASAR"
                self.log_message(f"Hasar kontrol: HASAR VAR ({shown}, {mode_text})")
            else:
                self.log_message(f"Hasar kontrol: HASAR YOK ({mode_text})")

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
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    app = PlakaTespitUygulamasi(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    
