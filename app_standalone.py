#!/usr/bin/env python3
"""
Joint Load Extraction Force Relationship Tool - Standalone
==========================================================

CLI ve Tkinter GUI'yi tek dosyada birlestirir.
Harici bagimliliklara (main.py, gui.py) ihtiyac duymadan calisir.

Kullanim:
    python app_standalone.py                  # GUI baslatir
    python app_standalone.py --cli --bdf ...  # CLI modunda calistirir
    python app_standalone.py --help           # Yardim
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from joint_load_extractor.bdf_parser import BDFParser, BarElementInfo
from joint_load_extractor.correlation import CorrelationEngine, CorrelationResult
from joint_load_extractor.h5_reader import H5Reader
from joint_load_extractor.op2_reader import OP2Reader, BarForceResult, ShellForceResult
from joint_load_extractor.reporter import ReportGenerator


# ============================================================
#  Ortak Fonksiyonlar (CLI + GUI)
# ============================================================


def setup_logging(verbose: bool = False) -> None:
    """Logging konfigurasyonu."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_bar_element_set(excel_path: str) -> List[int]:
    """Excel dosyasindan bar element listesini oku."""
    logger = logging.getLogger(__name__)
    logger.info("Excel dosyasi okunuyor: %s", excel_path)

    try:
        df = pd.read_excel(excel_path, sheet_name="Bar Element Set")
    except ValueError:
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        logger.info("Mevcut sheet'ler: %s", sheet_names)

        target_sheet = None
        for name in sheet_names:
            if "bar" in name.lower() and "element" in name.lower():
                target_sheet = name
                break

        if target_sheet is None:
            target_sheet = sheet_names[0]
            logger.warning(
                "'Bar Element Set' bulunamadi, '%s' kullaniliyor", target_sheet
            )

        df = pd.read_excel(excel_path, sheet_name=target_sheet)

    first_col = df.columns[0]
    bar_eids = df[first_col].dropna().astype(int).tolist()
    logger.info("%d bar element okundu", len(bar_eids))
    return bar_eids


def build_bar_forces_dataframe(
    bar_forces: Dict[Tuple[int, int], BarForceResult],
) -> pd.DataFrame:
    """Bar kuvvet sonuclarini DataFrame'e donustur."""
    rows = []
    for (sc_id, eid), result in sorted(bar_forces.items()):
        for t_idx in range(len(result.axial_force)):
            rows.append({
                "Subcase_ID": sc_id,
                "Bar_EID": eid,
                "Time_Step": t_idx,
                "Axial_Force": result.axial_force[t_idx],
                "Shear_1": result.shear_1[t_idx],
                "Shear_2": result.shear_2[t_idx],
                "BM_A1": result.bending_moment_a1[t_idx],
                "BM_A2": result.bending_moment_a2[t_idx],
                "BM_B1": result.bending_moment_b1[t_idx],
                "BM_B2": result.bending_moment_b2[t_idx],
                "Torque": result.torque[t_idx],
            })
    return pd.DataFrame(rows)


def build_shell_forces_dataframe(
    shell_forces: Dict[Tuple[int, int], ShellForceResult],
) -> pd.DataFrame:
    """Shell flux sonuclarini DataFrame'e donustur."""
    rows = []
    for (sc_id, eid), result in sorted(shell_forces.items()):
        for t_idx in range(len(result.membrane_x)):
            rows.append({
                "Subcase_ID": sc_id,
                "Shell_EID": eid,
                "Element_Type": result.elem_type,
                "Time_Step": t_idx,
                "NX": result.membrane_x[t_idx],
                "NY": result.membrane_y[t_idx],
                "NXY": result.membrane_xy[t_idx],
                "MX": result.bending_x[t_idx],
                "MY": result.bending_y[t_idx],
                "MXY": result.bending_xy[t_idx],
                "QX": result.shear_xz[t_idx],
                "QY": result.shear_yz[t_idx],
            })
    return pd.DataFrame(rows)


def run_correlation_analysis(
    connectivity: Dict[int, BarElementInfo],
    bar_forces: Dict[Tuple[int, int], BarForceResult],
    shell_forces: Dict[Tuple[int, int], ShellForceResult],
    h5_reader: H5Reader,
    subcase_id: int = None,
) -> Tuple[CorrelationEngine, List[CorrelationResult]]:
    """Korelasyon analizini calistir."""
    logger = logging.getLogger(__name__)
    engine = CorrelationEngine()
    all_results = []

    for bar_eid, info in sorted(connectivity.items()):
        logger.info("Bar %d icin korelasyon hesaplaniyor...", bar_eid)

        h5_data = h5_reader.get_joint_loads_for_bar(bar_eid)
        if h5_data is None or h5_data.empty:
            logger.warning("Bar %d icin H5 verisi bulunamadi", bar_eid)
            continue

        sc_keys = [k for k in bar_forces.keys() if k[1] == bar_eid]
        if not sc_keys:
            logger.warning("Bar %d icin OP2 kuvvet verisi yok", bar_eid)
            continue

        for sc_id_key, _ in sc_keys:
            if subcase_id is not None and sc_id_key != subcase_id:
                continue

            bar_result = bar_forces.get((sc_id_key, bar_eid))
            if bar_result is None:
                continue

            # 1. BAR korelasyonu
            bar_corr = engine.compute_bar_correlation(
                bar_eid=bar_eid,
                subcase_id=sc_id_key,
                bar_axial=bar_result.axial_force,
                bar_shear1=bar_result.shear_1,
                bar_shear2=bar_result.shear_2,
                bar_torque=bar_result.torque,
                h5_data=h5_data,
            )
            all_results.append(bar_corr)

            # 2. QUAD korelasyonu
            for qeid in info.connected_quads:
                shell_res = shell_forces.get((sc_id_key, qeid))
                if shell_res is None:
                    continue
                quad_corr = engine.compute_shell_correlation(
                    bar_eid=bar_eid,
                    subcase_id=sc_id_key,
                    shell_eid=qeid,
                    elem_type="CQUAD4",
                    nx=shell_res.membrane_x,
                    ny=shell_res.membrane_y,
                    nxy=shell_res.membrane_xy,
                    h5_data=h5_data,
                )
                all_results.append(quad_corr)

            # 3. TRIA korelasyonu
            for teid in info.connected_trias:
                shell_res = shell_forces.get((sc_id_key, teid))
                if shell_res is None:
                    continue
                tria_corr = engine.compute_shell_correlation(
                    bar_eid=bar_eid,
                    subcase_id=sc_id_key,
                    shell_eid=teid,
                    elem_type="CTRIA3",
                    nx=shell_res.membrane_x,
                    ny=shell_res.membrane_y,
                    nxy=shell_res.membrane_xy,
                    h5_data=h5_data,
                )
                all_results.append(tria_corr)

            # 4. COMBINED korelasyon
            shell_force_dict = {}
            shell_type_dict = {}
            for seid in list(info.connected_quads.keys()) + list(info.connected_trias.keys()):
                sf = shell_forces.get((sc_id_key, seid))
                if sf is not None:
                    shell_force_dict[seid] = {
                        "NX": sf.membrane_x,
                        "NY": sf.membrane_y,
                        "NXY": sf.membrane_xy,
                    }
                    shell_type_dict[seid] = sf.elem_type

            if shell_force_dict:
                combined_corr = engine.compute_combined_correlation(
                    bar_eid=bar_eid,
                    subcase_id=sc_id_key,
                    bar_axial=bar_result.axial_force,
                    bar_shear1=bar_result.shear_1,
                    bar_shear2=bar_result.shear_2,
                    shell_forces=shell_force_dict,
                    h5_data=h5_data,
                    shell_types=shell_type_dict,
                )
                all_results.append(combined_corr)

    return engine, all_results


# ============================================================
#  GUI Bileşenleri
# ============================================================


class QueueHandler(logging.Handler):
    """Log mesajlarini queue'ya yonlendirir (thread-safe GUI guncellemesi icin)."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class FileSelector(ttk.Frame):
    """Dosya secimi icin label + entry + browse butonu bileseni."""

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

        self.log_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread = None
        self._is_running = False

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self):
        """Arayuz bilesenlerini olustur."""
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)

        # === Baslik ===
        title_lbl = ttk.Label(
            main_frame,
            text="Joint Load Extraction - Force Relationship",
            font=("Segoe UI", 14, "bold"),
        )
        title_lbl.grid(row=0, column=0, pady=(0, 8), sticky="w")

        # === Dosya Secimi ===
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

        # === Ayarlar ===
        settings_frame = ttk.LabelFrame(main_frame, text=" Ayarlar ", padding=10)
        settings_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)

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
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate", length=400)
        self.progress.grid(row=4, column=0, sticky="ew", pady=(0, 4))

        self.status_var = tk.StringVar(value="Hazir.")
        self.status_label = ttk.Label(
            main_frame, textvariable=self.status_var, foreground="gray"
        )
        self.status_label.grid(row=5, column=0, sticky="w", pady=(0, 4))

        # === Log ===
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

        self.log_text.tag_config("INFO", foreground="#4fc1ff")
        self.log_text.tag_config("WARNING", foreground="#cca700")
        self.log_text.tag_config("ERROR", foreground="#f44747")
        self.log_text.tag_config("SUCCESS", foreground="#89d185")

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=7, column=0, sticky="ew", pady=(4, 0))

        ttk.Button(
            bottom_frame, text="Logu Temizle", command=self._clear_log
        ).pack(side="right")

    def _validate_inputs(self) -> bool:
        """Input dosyalarini kontrol et."""
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

        for path, name in checks[:-1]:
            if not Path(path).exists():
                messagebox.showerror("Dosya Bulunamadi", f"{name} bulunamadi:\n{path}")
                return False

        return True

    def _on_run(self):
        """Calistir butonuna tiklandiginda."""
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

        self.worker_thread = threading.Thread(target=self._run_analysis, daemon=True)
        self.worker_thread.start()

    def _run_analysis(self):
        """Analizi ayri thread'de calistir."""
        log_level = logging.DEBUG if self.verbose_var.get() else logging.INFO
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        root_logger.addHandler(queue_handler)

        logger = logging.getLogger(__name__)

        try:
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
            logger.info(
                "  %d bar, %d shell kuvvet verisi",
                len(bar_forces),
                len(shell_forces),
            )

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
            self.after(
                0,
                lambda: messagebox.showinfo(
                    "Basarili",
                    f"Analiz tamamlandi!\n\n"
                    f"Bar element: {len(bar_element_ids)}\n"
                    f"Baglanti: {len(connectivity)}\n"
                    f"Korelasyon: {len(correlation_results)}\n\n"
                    f"Cikti: {output_path}",
                ),
            )

        except Exception as e:
            logger.error("HATA: %s", str(e), exc_info=True)
            self._update_status(f"Hata: {e}")
            self.after(
                0,
                lambda: messagebox.showerror(
                    "Hata", f"Islem sirasinda hata olustu:\n\n{e}"
                ),
            )
        finally:
            self._is_running = False
            self.after(0, self._on_analysis_done)

    def _on_analysis_done(self):
        """Analiz bittiginde UI guncellemesi."""
        self.progress.stop()
        self.run_btn.configure(state="normal")

    def _update_status(self, text: str):
        """Durum metnini guncelle (thread-safe)."""
        self.after(0, lambda: self.status_var.set(text))

    def _poll_log_queue(self):
        """Log queue'yu periyodik olarak kontrol et."""
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break

            self.log_text.configure(state="normal")

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
        """Log alanini temizle."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


# ============================================================
#  CLI Modu
# ============================================================


def run_cli(args: argparse.Namespace) -> None:
    """Komut satiri modunda calistir."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Joint Load Extraction Force Relationship Tool")
    logger.info("=" * 60)

    for path_arg, name in [
        (args.bdf, "BDF"),
        (args.op2, "OP2"),
        (args.h5, "H5"),
        (args.excel, "Excel"),
    ]:
        p = Path(path_arg)
        if not p.exists():
            logger.error("%s dosyasi bulunamadi: %s", name, path_arg)
            sys.exit(1)
        logger.info("%s: %s", name, p.resolve())

    # ADIM 1
    logger.info("-" * 40)
    logger.info("ADIM 1: Bar element listesi okunuyor...")
    bar_element_ids = read_bar_element_set(args.excel)
    logger.info("  %d bar element okundu", len(bar_element_ids))

    # ADIM 2
    logger.info("-" * 40)
    logger.info("ADIM 2: BDF parse ediliyor...")
    bdf_parser = BDFParser(args.bdf)
    bdf_parser.parse()

    connectivity = bdf_parser.get_bar_connectivity(bar_element_ids)
    logger.info("  %d bar element icin baglanti bulundu", len(connectivity))

    all_shell_eids = set()
    for info in connectivity.values():
        all_shell_eids.update(info.connected_quads.keys())
        all_shell_eids.update(info.connected_trias.keys())
    logger.info("  Toplam %d bagli shell element", len(all_shell_eids))

    # ADIM 3
    logger.info("-" * 40)
    logger.info("ADIM 3: OP2 okunuyor...")
    op2_reader = OP2Reader(args.op2)
    op2_reader.read()

    bar_forces = op2_reader.get_bar_forces(
        bar_eids=list(connectivity.keys()),
        subcase_id=args.subcase,
    )
    logger.info("  %d bar element kuvvet verisi okundu", len(bar_forces))

    shell_forces = op2_reader.get_shell_forces(
        shell_eids=list(all_shell_eids),
        subcase_id=args.subcase,
    )
    logger.info("  %d shell element flux verisi okundu", len(shell_forces))

    # ADIM 4
    logger.info("-" * 40)
    logger.info("ADIM 4: H5 dosyasi okunuyor...")
    h5_reader = H5Reader(args.h5)
    h5_reader.read(group_path=args.h5_group)
    logger.info("  Joint Load Cap: %d satir", len(h5_reader.joint_load_cap))

    # ADIM 5
    logger.info("-" * 40)
    logger.info("ADIM 5: Korelasyon analizi yapiliyor...")
    engine, correlation_results = run_correlation_analysis(
        connectivity=connectivity,
        bar_forces=bar_forces,
        shell_forces=shell_forces,
        h5_reader=h5_reader,
        subcase_id=args.subcase,
    )
    correlation_summary = engine.get_summary_dataframe()
    logger.info("  %d korelasyon sonucu hesaplandi", len(correlation_results))

    # ADIM 6
    logger.info("-" * 40)
    logger.info("ADIM 6: Excel raporu olusturuluyor...")

    bar_forces_df = build_bar_forces_dataframe(bar_forces)
    shell_forces_df = build_shell_forces_dataframe(shell_forces)

    reporter = ReportGenerator(args.output)
    output_path = reporter.generate(
        bar_element_ids=bar_element_ids,
        connectivity=connectivity,
        bar_forces_df=bar_forces_df,
        shell_forces_df=shell_forces_df,
        h5_joint_loads=h5_reader.joint_load_cap,
        correlation_results=correlation_results,
        correlation_summary_df=correlation_summary,
    )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("TAMAMLANDI!")
    logger.info("Cikti dosyasi: %s", output_path)
    logger.info("Sure: %.1f saniye", elapsed)
    logger.info("=" * 60)

    print(f"\nSonuc Ozeti:")
    print(f"  Bar element sayisi:    {len(bar_element_ids)}")
    print(f"  Baglanti bulunan:      {len(connectivity)}")
    print(f"  Bagli shell element:   {len(all_shell_eids)}")
    print(f"  OP2 bar kuvvet:        {len(bar_forces)}")
    print(f"  OP2 shell flux:        {len(shell_forces)}")
    print(f"  H5 Joint Load satir:   {len(h5_reader.joint_load_cap)}")
    print(f"  Korelasyon sonucu:     {len(correlation_results)}")
    print(f"\n  Cikti: {output_path}")


# ============================================================
#  Giris Noktasi
# ============================================================


def main():
    """Ana giris noktasi. Parametresiz = GUI, --cli = komut satiri."""
    parser = argparse.ArgumentParser(
        description="Joint Load Extraction Force Relationship Tool (Standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanim modlari:
  python app_standalone.py                  # GUI baslatir (varsayilan)
  python app_standalone.py --cli --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx
  python app_standalone.py --cli --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx --output results.xlsx --subcase 1
        """,
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Komut satiri modunda calistir (GUI yerine)",
    )
    parser.add_argument("--bdf", default=None, help="Nastran BDF dosyasi yolu")
    parser.add_argument("--op2", default=None, help="Nastran OP2 dosyasi yolu")
    parser.add_argument("--h5", default=None, help="Joint Load Extraction H5 dosyasi yolu")
    parser.add_argument("--excel", default=None, help="Bar Element Set iceren Excel dosyasi yolu")
    parser.add_argument(
        "--output",
        default="joint_load_correlation_output.xlsx",
        help="Cikti Excel dosyasi yolu (varsayilan: joint_load_correlation_output.xlsx)",
    )
    parser.add_argument(
        "--subcase", type=int, default=None, help="Belirli bir subcase ID"
    )
    parser.add_argument(
        "--h5-group", default=None, help="H5 dosyasindaki Joint Load Cap tablosunun yolu"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detayli cikti")

    args = parser.parse_args()

    if args.cli:
        # CLI modu: dosya parametreleri zorunlu
        missing = []
        for param in ["bdf", "op2", "h5", "excel"]:
            if getattr(args, param) is None:
                missing.append(f"--{param}")
        if missing:
            parser.error(
                f"CLI modunda su parametreler zorunludur: {', '.join(missing)}"
            )
        run_cli(args)
    else:
        # GUI modu (varsayilan)
        app = Application()
        app.mainloop()


if __name__ == "__main__":
    main()
