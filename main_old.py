"""
Plaka Tespit Uygulaması - Ana Dosya
Tkinter GUI ile plaka tespit ve OCR uygulaması
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from pathlib import Path
from detector import PlakaDetector


class PlakaTespitUygulamasi:
    """Plaka tespit uygulaması GUI sınıfı"""

    def __init__(self, root):
        """
        Uygulamayı başlatır

        Args:
            root: Tkinter root penceresi
        """
        self.root = root
        self.root.title("Plaka Tespit ve OCR Uygulaması")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Değişkenler
        self.detector = PlakaDetector(use_ocr=True, ocr_languages=['tur', 'eng'])
        self.current_image_path = None
        self.original_image = None
        self.processed_image = None
        self.cropped_plate = None
        self.plate_coordinates = None
        self.ocr_enabled = tk.BooleanVar(value=True)

        # GUI oluştur
        self.create_gui()

    def create_gui(self):
        """GUI elemanlarını oluşturur"""

        # Başlık
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="🚗 PLAKA TESPİT ve OCR UYGULAMASI",
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=10)

        # Buton ve ayarlar paneli
        button_frame = tk.Frame(self.root, bg='#f0f0f0', height=100)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        button_frame.pack_propagate(False)

        # Üst satır - Butonlar
        top_row = tk.Frame(button_frame, bg='#f0f0f0')
        top_row.pack(fill=tk.X, pady=(0, 5))

        # Resim yükle butonu
        self.load_btn = tk.Button(
            top_row,
            text="📁 Resim Yükle",
            command=self.load_image,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        self.load_btn.pack(side=tk.LEFT, padx=10)

        # Plakayı Göster butonu
        self.show_plate_btn = tk.Button(
            top_row,
            text="🔍 Plakayı Göster",
            command=self.show_plate,
            font=('Arial', 12, 'bold'),
            bg='#9b59b6',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED
        )
        self.show_plate_btn.pack(side=tk.LEFT, padx=10)

        # Oku butonu
        self.read_btn = tk.Button(
            top_row,
            text="📖 Oku (Model + OCR)",
            command=self.read_plate,
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED
        )
        self.read_btn.pack(side=tk.LEFT, padx=10)

        # Kaydet butonu
        self.save_btn = tk.Button(
            top_row,
            text="💾 Sonucu Kaydet",
            command=self.save_result,
            font=('Arial', 12, 'bold'),
            bg='#e67e22',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=10)

        # Model ve OCR durumu (sağ taraf)
        status_frame = tk.Frame(top_row, bg='#f0f0f0')
        status_frame.pack(side=tk.RIGHT, padx=10)

        model_status = "✅ Model Yüklü" if self.detector.model else "⚠️ Model Bulunamadı"
        model_color = "#27ae60" if self.detector.model else "#e74c3c"

        self.model_label = tk.Label(
            status_frame,
            text=model_status,
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            fg=model_color
        )
        self.model_label.pack()

        ocr_status = "✅ OCR Hazır" if self.detector.ocr else "⚠️ OCR Yüklenemedi"
        ocr_color = "#27ae60" if self.detector.ocr else "#e74c3c"

        self.ocr_label = tk.Label(
            status_frame,
            text=ocr_status,
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            fg=ocr_color
        )
        self.ocr_label.pack()

        # Alt satır - OCR ayarları
        bottom_row = tk.Frame(button_frame, bg='#f0f0f0')
        bottom_row.pack(fill=tk.X)

        # OCR açık/kapalı
        ocr_check = tk.Checkbutton(
            bottom_row,
            text="📝 Plaka Numarası Oku (OCR)",
            variable=self.ocr_enabled,
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50',
            selectcolor='#ecf0f1',
            activebackground='#f0f0f0',
            cursor='hand2'
        )
        ocr_check.pack(side=tk.LEFT, padx=10)

        # Dil bilgisi
        lang_label = tk.Label(
            bottom_row,
            text="🌍 Desteklenen Diller: Türkçe, İngilizce",
            font=('Arial', 9, 'italic'),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        lang_label.pack(side=tk.LEFT, padx=20)

        # Resim görüntüleme paneli
        image_frame = tk.Frame(self.root, bg='#f0f0f0', height=200)
        image_frame.pack(fill=tk.X, expand=False, padx=20, pady=10)
        image_frame.pack_propagate(False)

        # Sol panel - Orijinal resim
        left_panel = tk.Frame(image_frame, bg='#ecf0f1', relief=tk.SUNKEN, bd=2, width=450, height=190)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left_panel.pack_propagate(False)

        left_title = tk.Label(
            left_panel,
            text="Orijinal Resim",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white',
            pady=5
        )
        left_title.pack(fill=tk.X)

        self.original_canvas = tk.Canvas(left_panel, bg='white', highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sağ panel - Plaka görüntüsü
        right_panel = tk.Frame(image_frame, bg='#ecf0f1', relief=tk.SUNKEN, bd=2, width=180, height=90)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))
        right_panel.pack_propagate(False)

        right_title = tk.Label(
            right_panel,
            text="Tespit Edilen Plaka",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white',
            pady=5
        )
        right_title.pack(fill=tk.X)

        self.processed_canvas = tk.Canvas(right_panel, bg='white', highlightthickness=0)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bilgi paneli
        info_frame = tk.Frame(self.root, bg='#34495e', relief=tk.RAISED, bd=2)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        # Tespit bilgisi
        detect_label = tk.Label(
            info_frame,
            text="🎯 Tespit:",
            font=('Arial', 11, 'bold'),
            bg='#34495e',
            fg='white'
        )
        detect_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.detect_text = tk.Label(
            info_frame,
            text="Henüz tespit yapılmadı",
            font=('Arial', 10),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.detect_text.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)

        # İşlem süresi
        self.time_label = tk.Label(
            info_frame,
            text="⏱️ Süre: -",
            font=('Arial', 11, 'bold'),
            bg='#34495e',
            fg='white'
        )
        self.time_label.pack(side=tk.RIGHT, padx=20, pady=10)

        # OCR sonuçları paneli (İKİ SÜTUN)
        ocr_frame = tk.Frame(self.root, bg='#ecf0f1', relief=tk.SUNKEN, bd=2)
        ocr_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        ocr_title = tk.Label(
            ocr_frame,
            text="📋 Okunan Plaka Numaraları",
            font=('Arial', 12, 'bold'),
            bg='#16a085',
            fg='white',
            pady=5
        )
        ocr_title.pack(fill=tk.X)

        # İki sütun için frame
        results_row = tk.Frame(ocr_frame, bg='white')
        results_row.pack(fill=tk.X, padx=5, pady=5)

        # SOL - MODEL SONUÇLARI
        model_frame = tk.Frame(results_row, bg='white', relief=tk.RIDGE, bd=1)
        model_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        model_title = tk.Label(
            model_frame,
            text="🤖 Model Sonuçları",
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            pady=3
        )
        model_title.pack(fill=tk.X)

        model_scroll_frame = tk.Frame(model_frame, bg='white')
        model_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        model_scrollbar = tk.Scrollbar(model_scroll_frame)
        model_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.model_text = tk.Text(
            model_scroll_frame,
            height=6,
            font=('Courier', 10),
            bg='white',
            fg='#2c3e50',
            yscrollcommand=model_scrollbar.set,
            wrap=tk.WORD
        )
        self.model_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        model_scrollbar.config(command=self.model_text.yview)
        self.model_text.insert('1.0', 'Henüz model çalıştırılmadı.')
        self.model_text.config(state=tk.DISABLED)

        # SAĞ - OCR SONUÇLARI
        ocr_result_frame = tk.Frame(results_row, bg='white', relief=tk.RIDGE, bd=1)
        ocr_result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ocr_result_title = tk.Label(
            ocr_result_frame,
            text="📝 OCR Sonuçları",
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            pady=3
        )
        ocr_result_title.pack(fill=tk.X)

        ocr_result_scroll_frame = tk.Frame(ocr_result_frame, bg='white')
        ocr_result_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        ocr_result_scrollbar = tk.Scrollbar(ocr_result_scroll_frame)
        ocr_result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.ocr_text = tk.Text(
            ocr_result_scroll_frame,
            height=6,
            font=('Courier', 10),
            bg='white',
            fg='#2c3e50',
            yscrollcommand=ocr_result_scrollbar.set,
            wrap=tk.WORD
        )
        self.ocr_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ocr_result_scrollbar.config(command=self.ocr_text.yview)
        self.ocr_text.insert('1.0', 'Henüz OCR çalıştırılmadı.')
        self.ocr_text.config(state=tk.DISABLED)

    def load_image(self):
        """Resim yükleme fonksiyonu"""
        file_path = filedialog.askopenfilename(
            title="Resim Seçin",
            filetypes=[
                ("Tüm Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Tüm Dosyalar", "*.*")
            ]
        )

        if file_path:
            try:
                self.current_image_path = file_path

                # Resmi yükle
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.original_image = image

                # Canvas'a göster
                self.display_image(image, self.original_canvas)

                # Plakayı Göster butonunu aktif et
                self.show_plate_btn.config(state=tk.NORMAL)
                self.read_btn.config(state=tk.DISABLED)

                # Bilgi alanlarını sıfırla
                self.detect_text.config(text="Henüz tespit yapılmadı")
                self.time_label.config(text="⏱️ Süre: -")
                
                self.model_text.config(state=tk.NORMAL)
                self.model_text.delete('1.0', tk.END)
                self.model_text.insert('1.0', 'Henüz model çalıştırılmadı.')
                self.model_text.config(state=tk.DISABLED)

                self.ocr_text.config(state=tk.NORMAL)
                self.ocr_text.delete('1.0', tk.END)
                self.ocr_text.insert('1.0', 'Henüz OCR çalıştırılmadı.')
                self.ocr_text.config(state=tk.DISABLED)

                # İşlenmiş resmi temizle
                self.processed_canvas.delete("all")
                self.processed_image = None
                self.cropped_plate = None
                self.plate_coordinates = None
                self.save_btn.config(state=tk.DISABLED)

            except Exception as e:
                messagebox.showerror("Hata", f"Resim yüklenemedi:\n{str(e)}")

    def show_plate(self):
        """Plakayı tespit edip göster"""
        if not self.current_image_path:
            messagebox.showwarning("Uyarı", "Lütfen önce bir resim yükleyin!")
            return

        if not self.detector.model:
            messagebox.showerror(
                "Model Hatası",
                "YOLO modeli yüklü değil!\n\n"
                "Lütfen 'best.pt' modelini 'models/' klasörüne yerleştirin."
            )
            return

        try:
            import time
            start_time = time.time()

            # YOLO ile plaka tespit et
            result = self.detector.detect_plate(self.current_image_path, read_text=False)
            
            if result['success'] and result['coordinates']:
                # İlk plakayı al
                x1, y1, x2, y2 = result['coordinates'][0]
                conf = result['confidence'][0]
                
                # Plakayı crop et
                plate_image = self.original_image[y1:y2, x1:x2]
                self.cropped_plate = plate_image
                self.plate_coordinates = (x1, y1, x2, y2)
                
                # Canvas'a göster
                self.display_image(plate_image, self.processed_canvas)
                
                # Bilgi güncelle
                self.detect_text.config(text=f"Plaka tespit edildi (Güven: {conf:.2%})")
                
                elapsed_time = time.time() - start_time
                self.time_label.config(text=f"⏱️ Tespit Süresi: {elapsed_time:.3f} saniye")
                
                # Oku butonunu aktif et
                self.read_btn.config(state=tk.NORMAL)
                
                # Model ve OCR sonuçlarını sıfırla
                self.model_text.config(state=tk.NORMAL)
                self.model_text.delete('1.0', tk.END)
                self.model_text.insert('1.0', 'Oku butonuna basarak başlayın.')
                self.model_text.config(state=tk.DISABLED)

                self.ocr_text.config(state=tk.NORMAL)
                self.ocr_text.delete('1.0', tk.END)
                self.ocr_text.insert('1.0', 'Oku butonuna basarak başlayın.')
                self.ocr_text.config(state=tk.DISABLED)
                
                messagebox.showinfo("Başarılı", f"Plaka tespit edildi!")
            else:
                messagebox.showwarning("Uyarı", "Resimde plaka tespit edilemedi.")
                self.read_btn.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu:\n{str(e)}")

    def read_plate(self):
        """Croplanmış plakayı hem Model hem OCR ile oku"""
        if self.cropped_plate is None:
            messagebox.showwarning("Uyarı", "Önce 'Plakayı Göster' butonuna basın!")
            return

        try:
            # Model ve OCR Sonuçlarını al
            plate_text, plate_conf, model_data, ocr_data = self.detector.read_plate_text(
                self.cropped_plate, image_is_rgb=True
            )

            # MODEL SONUCU
            model_result = "❌ Model yüklü değil"
            if model_data:
                if model_data['success']:
                    model_result = f"✓ Character Model\n"
                    model_result += f"─" * 25 + "\n"
                    model_result += f"Plaka: {model_data['text']}\n"
                    model_result += f"Güven: {model_data['confidence']:.1%}\n"
                    model_result += f"Chars: {len(model_data['characters'])}\n"
                    model_result += f"Süre: {model_data['processing_time']:.3f}s"
                else:
                    model_result = f"⚠ Model:\n{model_data['error']}"
            else:
                model_result = "⚠ Model başlatılmadı"

            # OCR SONUCU
            ocr_result = "❌ OCR yüklü değil"
            if ocr_data:
                if ocr_data['success']:
                    ocr_result = f"✓ Tesseract/EasyOCR\n"
                    ocr_result += f"─" * 25 + "\n"
                    ocr_result += f"Plaka: {ocr_data['text']}\n"
                    ocr_result += f"Güven: {ocr_data['confidence']:.1%}"
                else:
                    ocr_result = f"⚠ OCR:\n{ocr_data['error']}"
            else:
                ocr_result = "⚠ OCR başlatılmadı"

            # Sonuçları göster
            self.model_text.config(state=tk.NORMAL)
            self.model_text.delete('1.0', tk.END)
            self.model_text.insert('1.0', model_result)
            self.model_text.config(state=tk.DISABLED)

            self.ocr_text.config(state=tk.NORMAL)
            self.ocr_text.delete('1.0', tk.END)
            self.ocr_text.insert('1.0', ocr_result)
            self.ocr_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Hata", f"Okuma hatası:\n{str(e)}")

    def save_result(self):
        """Sonucu kaydetme fonksiyonu"""
        if self.processed_image is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek işlenmiş resim yok!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Sonucu Kaydet",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("Tüm Dosyalar", "*.*")
            ]
        )

        if file_path:
            try:
                success = self.detector.save_result(self.processed_image, file_path)
                if success:
                    messagebox.showinfo("Başarılı", f"Resim kaydedildi:\n{file_path}")
                else:
                    messagebox.showerror("Hata", "Resim kaydedilemedi!")
            except Exception as e:
                messagebox.showerror("Hata", f"Kaydetme hatası:\n{str(e)}")

    def display_image(self, image, canvas):
        """
        Resmi canvas üzerinde gösterir

        Args:
            image: Gösterilecek resim (numpy array, RGB)
            canvas: Tkinter Canvas nesnesi
        """
        # Canvas boyutlarını al
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # Resmi PIL formatına çevir
        pil_image = Image.fromarray(image)

        # Resmi canvas boyutuna sığdır (aspect ratio koruyarak)
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale * 0.95)
        new_height = int(img_height * scale * 0.95)

        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # PhotoImage'e çevir
        photo = ImageTk.PhotoImage(pil_image)

        # Canvas'a ekle
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER
        )

        # Referansı sakla (garbage collection'dan korumak için)
        canvas.image = photo


def main():
    """Ana fonksiyon"""
    root = tk.Tk()
    app = PlakaTespitUygulamasi(root)
    root.mainloop()


if __name__ == "__main__":
    main()
