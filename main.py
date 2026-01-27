"""
Plaka Tespit Uygulaması - Ana Dosya
Tkinter GUI ile plaka tespit uygulaması
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
        self.root.title("Plaka Tespit Uygulaması")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')

        # Değişkenler
        self.detector = PlakaDetector()
        self.current_image_path = None
        self.original_image = None
        self.processed_image = None

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
            text="🚗 PLAKA TESPİT UYGULAMASI",
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=10)

        # Buton paneli
        button_frame = tk.Frame(self.root, bg='#f0f0f0', height=80)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        button_frame.pack_propagate(False)

        # Resim yükle butonu
        self.load_btn = tk.Button(
            button_frame,
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

        # Çalıştır butonu
        self.run_btn = tk.Button(
            button_frame,
            text="▶ Çalıştır",
            command=self.run_detection,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED
        )
        self.run_btn.pack(side=tk.LEFT, padx=10)

        # Kaydet butonu
        self.save_btn = tk.Button(
            button_frame,
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

        # Model durumu
        model_status = "✅ Model Yüklü" if self.detector.model else "⚠️ Model Bulunamadı"
        model_color = "#27ae60" if self.detector.model else "#e74c3c"

        self.model_label = tk.Label(
            button_frame,
            text=model_status,
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            fg=model_color
        )
        self.model_label.pack(side=tk.RIGHT, padx=10)

        # Resim görüntüleme paneli
        image_frame = tk.Frame(self.root, bg='#f0f0f0')
        image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Sol panel - Orijinal resim
        left_panel = tk.Frame(image_frame, bg='#ecf0f1', relief=tk.SUNKEN, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

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

        # Sağ panel - İşlenmiş resim
        right_panel = tk.Frame(image_frame, bg='#ecf0f1', relief=tk.SUNKEN, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        right_title = tk.Label(
            right_panel,
            text="İşlenmiş Resim",
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

        # Koordinat bilgisi
        coord_label = tk.Label(
            info_frame,
            text="📍 Koordinatlar:",
            font=('Arial', 11, 'bold'),
            bg='#34495e',
            fg='white'
        )
        coord_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.coord_text = tk.Label(
            info_frame,
            text="Henüz tespit yapılmadı",
            font=('Arial', 10),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.coord_text.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)

        # İşlem süresi
        self.time_label = tk.Label(
            info_frame,
            text="⏱️ Süre: -",
            font=('Arial', 11, 'bold'),
            bg='#34495e',
            fg='white'
        )
        self.time_label.pack(side=tk.RIGHT, padx=20, pady=10)

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

                # Çalıştır butonunu aktif et
                self.run_btn.config(state=tk.NORMAL)

                # Bilgi alanlarını sıfırla
                self.coord_text.config(text="Henüz tespit yapılmadı")
                self.time_label.config(text="⏱️ Süre: -")

                # İşlenmiş resmi temizle
                self.processed_canvas.delete("all")
                self.processed_image = None
                self.save_btn.config(state=tk.DISABLED)

            except Exception as e:
                messagebox.showerror("Hata", f"Resim yüklenemedi:\n{str(e)}")

    def run_detection(self):
        """Plaka tespiti çalıştırma fonksiyonu"""
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
            # Tespit yap
            result = self.detector.detect_plate(self.current_image_path)

            if result['success']:
                # İşlenmiş resmi göster
                processed_img = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
                self.processed_image = result['image']
                self.display_image(processed_img, self.processed_canvas)

                # Koordinat bilgilerini göster
                if result['coordinates']:
                    coord_info = []
                    for i, (coords, conf) in enumerate(zip(result['coordinates'], result['confidence'])):
                        x1, y1, x2, y2 = coords
                        coord_info.append(
                            f"Plaka {i+1}: ({x1}, {y1}) - ({x2}, {y2}) | Güven: {conf:.2%}"
                        )
                    self.coord_text.config(text=" | ".join(coord_info))
                else:
                    self.coord_text.config(text=result['error'] or "Plaka tespit edilemedi")

                # İşlem süresini göster
                self.time_label.config(text=f"⏱️ Süre: {result['processing_time']:.3f} saniye")

                # Kaydet butonunu aktif et
                if result['coordinates']:
                    self.save_btn.config(state=tk.NORMAL)

                # Başarı mesajı
                if result['coordinates']:
                    messagebox.showinfo(
                        "Başarılı",
                        f"{len(result['coordinates'])} plaka tespit edildi!\n"
                        f"İşlem süresi: {result['processing_time']:.3f} saniye"
                    )
                else:
                    messagebox.showwarning("Uyarı", "Resimde plaka tespit edilemedi.")

            else:
                messagebox.showerror("Hata", f"Tespit başarısız:\n{result['error']}")

        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu:\n{str(e)}")

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
        new_width = int(img_width * scale * 0.95)  # %95 boyut
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
