#!/usr/bin/env python3
"""
Joint Load Extraction Force Relationship Tool - Tkinter GUI

Dosya seçim diyalogları, progress bar ve log çıktısı ile
kullanıcı dostu masaüstü arayüzü.
"""

import logging
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path


class QueueHandler(logging.Handler):
    """Log mesajlarını queue'ya yönlendirir (thread-safe GUI güncellemesi için)."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class FileSelector(ttk.Frame):
    """Dosya seçimi için label + entry + browse butonu bileşeni."""

    def __init__(
        self,
        parent,
        label_text: str,
        filetypes: list,
        is_save: bool = False,
        default_ext: str = "",
    ):
        super().__init__(parent)
        self.filetypes = filetypes
        self.is_save = is_save
        self.default_ext = default_ext

        self.columnconfigure(1, weight=1)

        self.label = ttk.Label(self, text=label_text, width=14, anchor="w")
        self.label.grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.var)
        self.entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))

        self.btn = ttk.Button(self, text="Gozat...", width=9, command=self._browse)
        self.btn.grid(row=0, column=2)

    def _browse(self):
        if self.is_save:
            path = filedialog.asksaveasfilename(
                defaultextension=self.default_ext,
                filetypes=self.filetypes,
            )
        else:
            path = filedialog.askopenfilename(filetypes=self.filetypes)
        if path:
            self.var.set(path)

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str):
        self.var.set(value)


class Application(tk.Tk):
    """Ana uygulama penceresi."""

    WINDOW_TITLE = "Joint Load Extraction - Force Relationship Tool"
    MIN_WIDTH = 750
    MIN_HEIGHT = 620

    def __init__(self):
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.geometry("800x680")

        # Queue ve thread
        self.log_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread = None
        self._is_running = False

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self):
        """Arayüz bileşenlerini oluştur."""
        # --- Ana container ---
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)

        # === Başlık ===
        title_lbl = ttk.Label(
            main_frame,
            text="Joint Load Extraction - Force Relationship",
            font=("Segoe UI", 14, "bold"),
        )
        title_lbl.grid(row=0, column=0, pady=(0, 8), sticky="w")

        # === Dosya Seçimi Bölümü ===
        file_frame = ttk.LabelFrame(main_frame, text=" Dosya Secimi ", padding=10)
        file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(0, weight=1)

        self.bdf_selector = FileSelector(
            file_frame,
            "BDF Dosyasi:",
            [("Nastran BDF", "*.bdf *.dat *.nas *.BDF *.DAT"), ("Tum Dosyalar", "*.*")],
        )
        self.bdf_selector.grid(row=0, column=0, sticky="ew", pady=2)

        self.op2_selector = FileSelector(
            file_frame,
            "OP2 Dosyasi:",
            [("Nastran OP2", "*.op2 *.OP2"), ("Tum Dosyalar", "*.*")],
        )
        self.op2_selector.grid(row=1, column=0, sticky="ew", pady=2)

        self.h5_selector = FileSelector(
            file_frame,
            "H5 Dosyasi:",
            [("HDF5", "*.h5 *.hdf5 *.H5 *.HDF5"), ("Tum Dosyalar", "*.*")],
        )
        self.h5_selector.grid(row=2, column=0, sticky="ew", pady=2)

        self.excel_selector = FileSelector(
            file_frame,
            "Excel Dosyasi:",
            [("Excel", "*.xlsx *.xls *.XLSX"), ("Tum Dosyalar", "*.*")],
        )
        self.excel_selector.grid(row=3, column=0, sticky="ew", pady=2)

        self.output_selector = FileSelector(
            file_frame,
            "Cikti Dosyasi:",
            [("Excel", "*.xlsx")],
            is_save=True,
            default_ext=".xlsx",
        )
        self.output_selector.grid(row=4, column=0, sticky="ew", pady=2)
        self.output_selector.set("joint_load_correlation_output.xlsx")

        # === Ayarlar Bölümü ===
        settings_frame = ttk.LabelFrame(main_frame, text=" Ayarlar ", padding=10)
        settings_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)

        # Subcase
        ttk.Label(settings_frame, text="Subcase ID:", anchor="w").grid(
            row=0, column=0, padx=(0, 5), sticky="w"
        )
        self.subcase_var = tk.StringVar(value="")
        self.subcase_entry = ttk.Entry(
            settings_frame, textvariable=self.subcase_var, width=12
        )
        self.subcase_entry.grid(row=0, column=1, sticky="w", padx=(0, 20))

        ttk.Label(
            settings_frame, text="(Bos = tum subcaseler)", foreground="gray"
        ).grid(row=0, column=2, sticky="w")

        # H5 Group Path
        ttk.Label(settings_frame, text="H5 Grup Yolu:", anchor="w").grid(
            row=1, column=0, padx=(0, 5), pady=(5, 0), sticky="w"
        )
        self.h5_group_var = tk.StringVar(value="")
        self.h5_group_entry = ttk.Entry(
            settings_frame, textvariable=self.h5_group_var
        )
        self.h5_group_entry.grid(
            row=1, column=1, columnspan=3, sticky="ew", pady=(5, 0)
        )

        ttk.Label(
            settings_frame,
            text='(Bos = otomatik ara, ornek: "JOINT_LOADS_CAP/table")',
            foreground="gray",
        ).grid(row=2, column=1, columnspan=3, sticky="w")

        # Verbose
        self.verbose_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            settings_frame, text="Detayli cikti (verbose)", variable=self.verbose_var
        ).grid(row=3, column=0, columnspan=2, pady=(5, 0), sticky="w")

        # === Butonlar ===
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        btn_frame.columnconfigure(0, weight=1)

        self.run_btn = ttk.Button(
            btn_frame,
            text="  Calistir  ",
            command=self._on_run,
            style="Accent.TButton",
        )
        self.run_btn.grid(row=0, column=0, pady=2)

        # === Progress Bar ===
        self.progress = ttk.Progressbar(
            main_frame, mode="indeterminate", length=400
        )
        self.progress.grid(row=4, column=0, sticky="ew", pady=(0, 4))

        # Durum etiketi
        self.status_var = tk.StringVar(value="Hazir.")
        self.status_label = ttk.Label(
            main_frame, textvariable=self.status_var, foreground="gray"
        )
        self.status_label.grid(row=5, column=0, sticky="w", pady=(0, 4))

        # === Log Çıktısı ===
        log_frame = ttk.LabelFrame(main_frame, text=" Log ", padding=4)
        log_frame.grid(row=6, column=0, sticky="nsew", pady=(0, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Log renk etiketleri
        self.log_text.tag_config("INFO", foreground="#4fc1ff")
        self.log_text.tag_config("WARNING", foreground="#cca700")
        self.log_text.tag_config("ERROR", foreground="#f44747")
        self.log_text.tag_config("SUCCESS", foreground="#89d185")

        # === Alt bar: Clear Log ===
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=7, column=0, sticky="ew", pady=(4, 0))

        ttk.Button(
            bottom_frame, text="Logu Temizle", command=self._clear_log
        ).pack(side="right")

    def _validate_inputs(self) -> bool:
        """Input dosyalarını kontrol et."""
        checks = [
            (self.bdf_selector.get(), "BDF dosyasi"),
            (self.op2_selector.get(), "OP2 dosyasi"),
            (self.h5_selector.get(), "H5 dosyasi"),
            (self.excel_selector.get(), "Excel dosyasi"),
            (self.output_selector.get(), "Cikti dosyasi"),
        ]

        for path, name in checks:
            if not path:
                messagebox.showwarning("Eksik Alan", f"{name} secilmedi!")
                return False

        # Dosya varlık kontrolü (çıktı hariç)
        for path, name in checks[:-1]:
            if not Path(path).exists():
                messagebox.showerror("Dosya Bulunamadi", f"{name} bulunamadi:\n{path}")
                return False

        return True

    def _on_run(self):
        """Çalıştır butonuna tıklandığında."""
        if self._is_running:
            messagebox.showinfo("Bilgi", "Islem devam ediyor, lutfen bekleyin.")
            return

        if not self._validate_inputs():
            return

        self._is_running = True
        self.run_btn.configure(state="disabled")
        self.progress.start(15)
        self.status_var.set("Islem baslatiliyor...")
        self._clear_log()

        # Worker thread başlat
        self.worker_thread = threading.Thread(target=self._run_analysis, daemon=True)
        self.worker_thread.start()

    def _run_analysis(self):
        """Analizi ayrı thread'de çalıştır."""
        # Logging kur
        log_level = logging.DEBUG if self.verbose_var.get() else logging.INFO
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        # Mevcut handlerları temizle (duplicate önleme)
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        root_logger.addHandler(queue_handler)

        logger = logging.getLogger(__name__)

        try:
            # Import burada yapılır (ilk çalıştırmada modül yüklenmesi)
            from joint_load_extractor.bdf_parser import BDFParser
            from joint_load_extractor.op2_reader import OP2Reader
            from joint_load_extractor.h5_reader import H5Reader
            from joint_load_extractor.correlation import CorrelationEngine
            from joint_load_extractor.reporter import ReportGenerator
            from main import (
                read_bar_element_set,
                build_bar_forces_dataframe,
                build_shell_forces_dataframe,
                run_correlation_analysis,
            )

            bdf_path = self.bdf_selector.get()
            op2_path = self.op2_selector.get()
            h5_path = self.h5_selector.get()
            excel_path = self.excel_selector.get()
            output_path = self.output_selector.get()
            subcase_str = self.subcase_var.get().strip()
            subcase_id = int(subcase_str) if subcase_str else None
            h5_group = self.h5_group_var.get().strip() or None

            # ADIM 1
            self._update_status("Adim 1/6: Excel okunuyor...")
            logger.info("=" * 50)
            logger.info("ADIM 1: Bar element listesi okunuyor...")
            bar_element_ids = read_bar_element_set(excel_path)
            logger.info("  %d bar element okundu", len(bar_element_ids))

            # ADIM 2
            self._update_status("Adim 2/6: BDF parse ediliyor...")
            logger.info("ADIM 2: BDF parse ediliyor...")
            bdf_parser = BDFParser(bdf_path)
            bdf_parser.parse()
            connectivity = bdf_parser.get_bar_connectivity(bar_element_ids)
            logger.info("  %d bar element baglantisi bulundu", len(connectivity))

            all_shell_eids = set()
            for info in connectivity.values():
                all_shell_eids.update(info.connected_quads.keys())
                all_shell_eids.update(info.connected_trias.keys())
            logger.info("  %d bagli shell element", len(all_shell_eids))

            # ADIM 3
            self._update_status("Adim 3/6: OP2 okunuyor...")
            logger.info("ADIM 3: OP2 okunuyor...")
            op2_reader = OP2Reader(op2_path)
            op2_reader.read()

            bar_forces = op2_reader.get_bar_forces(
                list(connectivity.keys()), subcase_id
            )
            shell_forces = op2_reader.get_shell_forces(
                list(all_shell_eids), subcase_id
            )
            logger.info("  %d bar, %d shell kuvvet verisi", len(bar_forces), len(shell_forces))

            # ADIM 4
            self._update_status("Adim 4/6: H5 okunuyor...")
            logger.info("ADIM 4: H5 dosyasi okunuyor...")
            h5_reader = H5Reader(h5_path)
            h5_reader.read(group_path=h5_group)
            logger.info("  %d satir Joint Load Cap", len(h5_reader.joint_load_cap))

            # ADIM 5
            self._update_status("Adim 5/6: Korelasyon hesaplaniyor...")
            logger.info("ADIM 5: Korelasyon analizi...")
            engine, correlation_results = run_correlation_analysis(
                connectivity, bar_forces, shell_forces, h5_reader, subcase_id
            )
            correlation_summary = engine.get_summary_dataframe()
            logger.info("  %d korelasyon sonucu", len(correlation_results))

            # ADIM 6
            self._update_status("Adim 6/6: Rapor olusturuluyor...")
            logger.info("ADIM 6: Excel raporu olusturuluyor...")

            bar_forces_df = build_bar_forces_dataframe(bar_forces)
            shell_forces_df = build_shell_forces_dataframe(shell_forces)

            reporter = ReportGenerator(output_path)
            reporter.generate(
                bar_element_ids=bar_element_ids,
                connectivity=connectivity,
                bar_forces_df=bar_forces_df,
                shell_forces_df=shell_forces_df,
                h5_joint_loads=h5_reader.joint_load_cap,
                correlation_results=correlation_results,
                correlation_summary_df=correlation_summary,
            )

            logger.info("=" * 50)
            logger.info("TAMAMLANDI! Cikti: %s", output_path)

            self._update_status(f"Tamamlandi! -> {output_path}")
            self.after(0, lambda: messagebox.showinfo(
                "Basarili",
                f"Analiz tamamlandi!\n\n"
                f"Bar element: {len(bar_element_ids)}\n"
                f"Baglanti: {len(connectivity)}\n"
                f"Korelasyon: {len(correlation_results)}\n\n"
                f"Cikti: {output_path}",
            ))

        except Exception as e:
            logger.error("HATA: %s", str(e), exc_info=True)
            self._update_status(f"Hata: {e}")
            self.after(0, lambda: messagebox.showerror(
                "Hata", f"Islem sirasinda hata olustu:\n\n{e}"
            ))
        finally:
            self._is_running = False
            self.after(0, self._on_analysis_done)

    def _on_analysis_done(self):
        """Analiz bittiğinde UI güncellemesi."""
        self.progress.stop()
        self.run_btn.configure(state="normal")

    def _update_status(self, text: str):
        """Durum metnini güncelle (thread-safe)."""
        self.after(0, lambda: self.status_var.set(text))

    def _poll_log_queue(self):
        """Log queue'yu periyodik olarak kontrol et ve text widget'a yaz."""
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break

            self.log_text.configure(state="normal")

            # Renk etiketi belirle
            tag = None
            if "[INFO]" in msg:
                tag = "INFO"
            elif "[WARNING]" in msg:
                tag = "WARNING"
            elif "[ERROR]" in msg:
                tag = "ERROR"
            elif "TAMAMLANDI" in msg:
                tag = "SUCCESS"

            if tag:
                self.log_text.insert("end", msg + "\n", tag)
            else:
                self.log_text.insert("end", msg + "\n")

            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.after(100, self._poll_log_queue)

    def _clear_log(self):
        """Log alanını temizle."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


def main():
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()
