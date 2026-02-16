#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predicted Load Generation Tool

Kullanim:
  1. Joint Load Correlation Output Excel'i sec (Correlation Summary + Total Summary sheet'leri)
  2. Prediction H5 dosyasini sec (ELFORCE_BAR_COMBINED + ELFORCE_SHELL_COMBINED)
  3. "Generate" butonuna bas -> CSV olusturulur

Akis:
  - Correlation Summary sheet'inden katsayilar okunur
  - Total Summary sheet'inden connectivity (bar -> shell) cikarilir
  - Prediction H5'ten bar AF ve shell NX/NY/NXY okunur
  - Her (bar_eid, element_type, subcase) icin:
      avg_NX/NY/NXY = bagli tum shell'lerin ortalamasi
      predicted = [AF, avg_NX, avg_NY, avg_NXY] @ coefficients + intercept
  - Cikti: Bar_EID, Element_Type, Subcase_ID, Pred_FX, Pred_FY, Pred_NX, Pred_NY, Pred_NXY

CLI:
  python predicted_load_generation.py --excel output.xlsx --h5 prediction.h5 --output predicted.csv
  python predicted_load_generation.py   (GUI modu)
"""

import argparse
import logging
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger("PredictedLoadGen")

# ============================================================
#  Sabitler
# ============================================================

BAR_TABLE_PATH = "ELFORCE_BAR_COMBINED/table"
SHELL_TABLE_PATH = "ELFORCE_SHELL_COMBINED/table"

TARGET_COL_MAP = {
    "F Bearing X": "Pred_FX",
    "F Bearing Y": "Pred_FY",
    "NX Bypass": "Pred_NX",
    "NY Bypass": "Pred_NY",
    "NXY Bypass": "Pred_NXY",
}

OUTPUT_FIXED_COLUMNS = [
    "Bar_EID", "Element_Type", "Subcase_ID",
]
OUTPUT_PRED_COLUMNS = [
    "Pred_FX", "Pred_FY", "Pred_NX", "Pred_NY", "Pred_NXY",
]


# ============================================================
#  Excel Okuyucu
# ============================================================


def read_coefficients(excel_path: str) -> pd.DataFrame:
    """
    Correlation Summary sheet'inden per-shell katsayilari oku.

    Kolonlar: Bar_EID, Element_Type, Target, Shell_EIDs,
              Coeff_Bar_Axial, Coeff_Shell_{EID1}_Nx, ..._Ny, ..._Nxy, ...,
              Intercept
    """
    xls = pd.ExcelFile(excel_path)
    sheets = xls.sheet_names
    logger.info("Excel sheet'leri: %s", sheets)

    df = None
    for candidate in ["Correlation Summary", "Total Summary"]:
        if candidate in sheets:
            df = pd.read_excel(excel_path, sheet_name=candidate)
            logger.info("Sheet kullanildi: '%s'", candidate)
            break

    if df is None:
        for name in sheets:
            if "summary" in name.lower():
                df = pd.read_excel(excel_path, sheet_name=name)
                logger.info("Sheet kullanildi: '%s'", name)
                break

    if df is None:
        raise ValueError(f"Summary sheet bulunamadi. Mevcut: {sheets}")

    required = ["Bar_EID", "Element_Type", "Target", "Intercept"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}. Mevcut: {list(df.columns)}")

    # Dinamik Coeff_* kolonlarini bul
    coeff_cols = [c for c in df.columns if c.startswith("Coeff_")]
    numeric_cols = coeff_cols + ["Intercept"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Bar_EID", "Element_Type"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Bar_EID"])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    logger.info("  %d katsayi satiri, %d Coeff kolonu, %d unique Bar_EID",
                len(df), len(coeff_cols), df["Bar_EID"].nunique())
    logger.info("  Coeff kolonlari: %s", coeff_cols[:10])

    return df


def extract_shell_eids_from_coefficients(coeff_df: pd.DataFrame) -> dict:
    """
    Katsayi DataFrame'inden bar_eid -> [shell_eids] eslesmesini cikar.
    Shell_EIDs kolonu veya Coeff_Shell_XXXX_Nx isimlerinden.
    """
    import re

    shell_map = {}

    # Yontem 1: Shell_EIDs kolonu
    if "Shell_EIDs" in coeff_df.columns:
        for bar_eid, grp in coeff_df.groupby("Bar_EID"):
            eids_str = grp.iloc[0].get("Shell_EIDs", "")
            if pd.notna(eids_str) and str(eids_str).strip():
                try:
                    eids = [int(float(x.strip())) for x in str(eids_str).split(",") if x.strip()]
                    if eids:
                        shell_map[int(float(bar_eid))] = eids
                except (ValueError, TypeError):
                    pass

    # Yontem 2: Kolon isimlerinden
    if not shell_map:
        shell_pattern = re.compile(r"Coeff_Shell_(\d+)_N[xXyY]+")
        all_eids = set()
        for col in coeff_df.columns:
            m = shell_pattern.match(col)
            if m:
                all_eids.add(int(m.group(1)))

        if all_eids:
            for bar_eid, grp in coeff_df.groupby("Bar_EID"):
                row = grp.iloc[0]
                eids = []
                for seid in sorted(all_eids):
                    col_nx = f"Coeff_Shell_{seid}_Nx"
                    if col_nx in row.index and pd.notna(row.get(col_nx)):
                        eids.append(seid)
                if eids:
                    shell_map[int(float(bar_eid))] = eids

    logger.info("  Shell EIDs: %d bar element", len(shell_map))
    return shell_map


# ============================================================
#  H5 Okuyucu
# ============================================================


def read_prediction_h5(h5_path: str):
    """
    Prediction H5'ten bar ve shell verilerini oku.

    Returns: (bar_df, shell_df)
      bar_df:   Bar_EID, Subcase_ID, AF
      shell_df: Shell_EID, Subcase_ID, NX, NY, NXY
    """
    logger.info("Prediction H5 okunuyor: %s", h5_path)
    bar_df = pd.DataFrame()
    shell_df = pd.DataFrame()

    with h5py.File(h5_path, "r") as f:
        if BAR_TABLE_PATH in f:
            bar_df = _h5_to_df(f[BAR_TABLE_PATH])
            if "Element_ID" in bar_df.columns:
                bar_df = bar_df.rename(columns={"Element_ID": "Bar_EID"})
            # H5 string dtype sorunu: tum kolonlari numerik yap
            bar_df = _force_numeric(bar_df)
            logger.info("  Bar: %d satir, kolonlar: %s, dtypes: %s",
                        len(bar_df), list(bar_df.columns),
                        {c: str(bar_df[c].dtype) for c in bar_df.columns[:5]})
        else:
            logger.error("  '%s' bulunamadi!", BAR_TABLE_PATH)
            _log_h5(f)

        if SHELL_TABLE_PATH in f:
            shell_df = _h5_to_df(f[SHELL_TABLE_PATH])
            renames = {}
            if "Element_ID" in shell_df.columns:
                renames["Element_ID"] = "Shell_EID"
            for old, new in [("MX", "NX"), ("MY", "NY"), ("MXY", "NXY")]:
                if old in shell_df.columns:
                    renames[old] = new
            shell_df = shell_df.rename(columns=renames)
            # H5 string dtype sorunu: tum kolonlari numerik yap
            shell_df = _force_numeric(shell_df)
            logger.info("  Shell: %d satir, kolonlar: %s, dtypes: %s",
                        len(shell_df), list(shell_df.columns),
                        {c: str(shell_df[c].dtype) for c in shell_df.columns[:5]})
        else:
            logger.error("  '%s' bulunamadi!", SHELL_TABLE_PATH)
            _log_h5(f)

    return bar_df, shell_df


def _h5_to_df(dataset) -> pd.DataFrame:
    """H5 dataset -> DataFrame."""
    if dataset.dtype.names:
        data = {}
        for name in dataset.dtype.names:
            col = dataset[name]
            if col.dtype.kind in ("S", "O"):
                try:
                    col = np.array([
                        x.decode("utf-8") if isinstance(x, bytes) else x
                        for x in col
                    ])
                except (UnicodeDecodeError, AttributeError):
                    pass
            data[name] = col
        return pd.DataFrame(data)
    return pd.DataFrame(dataset[:])


def _force_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame'deki tum potansiyel numerik kolonlari pd.to_numeric ile donustur.
    H5 dosyalari bazen numerik verileri string dtype ile saklar.
    """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except (ValueError, TypeError):
            pass
    return df


def _log_h5(f):
    """H5 yapisini logla."""
    items = []

    def _walk(g, prefix=""):
        for k in g:
            p = f"{prefix}/{k}" if prefix else k
            if isinstance(g[k], h5py.Group):
                _walk(g[k], p)
            elif isinstance(g[k], h5py.Dataset):
                cols = list(g[k].dtype.names) if g[k].dtype.names else []
                items.append(f"  {p}: {cols[:8]}")

    _walk(f)
    logger.info("H5 yapisi:\n%s", "\n".join(items))


# ============================================================
#  Kolon Normalizasyon
# ============================================================


def _normalize_bar(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        low = c.lower().replace(" ", "_").replace("-", "_")
        if low in ("bar_eid", "bar_element_id", "bareid", "bar_id", "element_id"):
            col_map[c] = "bar_eid"
        elif low in ("subcase_id", "subcaseid", "subcase", "sc_id"):
            col_map[c] = "subcase_id"
        elif low in ("af", "axial_force", "axialforce", "bar_axial", "axial"):
            col_map[c] = "af"
    return df.rename(columns=col_map)


def _normalize_shell(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        low = c.lower().replace(" ", "_").replace("-", "_")
        if low in ("element_id", "elementid", "shell_eid", "shelleid", "eid"):
            col_map[c] = "element_id"
        elif low in ("subcase_id", "subcaseid", "subcase", "sc_id"):
            col_map[c] = "subcase_id"
        elif low in ("nx", "mx", "membrane_x", "shell_nx"):
            col_map[c] = "nx"
        elif low in ("ny", "my", "membrane_y", "shell_ny"):
            col_map[c] = "ny"
        elif low in ("nxy", "mxy", "membrane_xy", "shell_nxy"):
            col_map[c] = "nxy"
    return df.rename(columns=col_map)


# ============================================================
#  Tahmin Motoru
# ============================================================


def generate_predictions(
    coeff_df: pd.DataFrame,
    bar_df: pd.DataFrame,
    shell_df: pd.DataFrame,
    shell_eid_map: dict,
) -> pd.DataFrame:
    """
    Per-shell katsayilarla tahmin. Reporter ile BIREBIR ayni mantik.

    Reporter'da:
      X = [Bar_Axial, Shell_{EID1}_Nx, Shell_{EID1}_Ny, Shell_{EID1}_Nxy, ...]
      predicted = X @ eq.coefficients + eq.intercept

    Her shell'in AYRI katsayisi var, ortalama ALINMAZ.
    """
    bar_df = _normalize_bar(bar_df)
    shell_df = _normalize_shell(shell_df)

    logger.info("=" * 60)
    logger.info("TAHMIN BASLADI (per-shell)")
    logger.info("  Bar:   %d satir", len(bar_df))
    logger.info("  Shell: %d satir", len(shell_df))
    logger.info("  Coeff: %d satir", len(coeff_df))
    logger.info("  Shell EID map: %d bar", len(shell_eid_map))

    groups = coeff_df.groupby(["Bar_EID", "Element_Type"])
    rows = []

    for (bar_eid, et), grp in groups:
        bar_eid_int = int(float(bar_eid))
        et_int = int(float(et))

        # Bagli shell EID'leri
        connected = shell_eid_map.get(bar_eid_int, [])
        if not connected:
            logger.debug("Bar %d: shell EID yok, atlaniyor", bar_eid_int)
            continue

        sorted_shells = sorted(connected)

        # Her target icin katsayi vektoru hazirla
        eq_list = []
        for _, crow in grp.iterrows():
            target = str(crow["Target"])
            intercept = float(crow.get("Intercept", 0.0) or 0.0)

            # [Bar_Axial, Shell1_Nx, Shell1_Ny, Shell1_Nxy, Shell2_Nx, ...]
            coeffs = [float(crow.get("Coeff_Bar_Axial", 0.0) or 0.0)]
            for seid in sorted_shells:
                coeffs.append(float(crow.get(f"Coeff_Shell_{seid}_Nx", 0.0) or 0.0))
                coeffs.append(float(crow.get(f"Coeff_Shell_{seid}_Ny", 0.0) or 0.0))
                coeffs.append(float(crow.get(f"Coeff_Shell_{seid}_Nxy", 0.0) or 0.0))
            eq_list.append((target, np.array(coeffs, dtype=float), intercept))

        # Bar AF
        bar_sub = bar_df[bar_df["bar_eid"] == bar_eid_int]
        if bar_sub.empty:
            bar_sub = bar_df[bar_df["bar_eid"] == float(bar_eid_int)]
        if bar_sub.empty:
            continue

        # Her subcase icin
        for _, brow in bar_sub.iterrows():
            sc_id = brow["subcase_id"]
            af = float(brow["af"])

            # Her shell'in AYRI kuvvet degeri
            predictor = [af]
            predictor_info = {"Bar_Axial": af}
            all_found = True

            for seid in sorted_shells:
                srow = shell_df[
                    (shell_df["element_id"] == seid) &
                    (shell_df["subcase_id"] == sc_id)
                ]
                if srow.empty:
                    srow = shell_df[
                        (shell_df["element_id"] == float(seid)) &
                        (shell_df["subcase_id"].between(sc_id - 0.5, sc_id + 0.5))
                    ]
                if srow.empty:
                    all_found = False
                    break

                nx_val = float(srow["nx"].iloc[0])
                ny_val = float(srow["ny"].iloc[0])
                nxy_val = float(srow["nxy"].iloc[0])
                predictor.extend([nx_val, ny_val, nxy_val])
                predictor_info[f"Shell_{seid}_Nx"] = nx_val
                predictor_info[f"Shell_{seid}_Ny"] = ny_val
                predictor_info[f"Shell_{seid}_Nxy"] = nxy_val

            if not all_found:
                continue

            predictor_arr = np.array(predictor, dtype=float)

            row = {
                "Bar_EID": bar_eid_int,
                "Element_Type": et_int,
                "Subcase_ID": int(float(sc_id)),
            }
            row.update(predictor_info)

            for target, coeffs, intercept in eq_list:
                predicted = float(predictor_arr @ coeffs + intercept)
                col_name = TARGET_COL_MAP.get(
                    target, f"Pred_{target.replace(' ', '_')}"
                )
                row[col_name] = predicted

            rows.append(row)

    logger.info("=" * 60)

    if not rows:
        logger.warning("Hicbir tahmin uretilmedi!")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info("Toplam %d tahmin satiri", len(df))

    # Kolon sirasi: sabit + predictor'lar + prediction'lar
    fixed = [c for c in OUTPUT_FIXED_COLUMNS if c in df.columns]
    pred_cols = [c for c in OUTPUT_PRED_COLUMNS if c in df.columns]
    predictor_cols = [c for c in df.columns
                      if c.startswith("Bar_Axial") or c.startswith("Shell_")]
    used = set(fixed + predictor_cols + pred_cols)
    extra = [c for c in df.columns if c not in used]
    return df[fixed + sorted(predictor_cols) + pred_cols + extra]


# ============================================================
#  Ana Fonksiyon (CLI)
# ============================================================


def run_prediction(excel_path: str, h5_path: str, output_csv: str) -> pd.DataFrame:
    """Tam tahmin pipeline'i."""
    logger.info("=" * 50)
    logger.info("Predicted Load Generation")
    logger.info("  Excel:  %s", excel_path)
    logger.info("  H5:     %s", h5_path)
    logger.info("  Output: %s", output_csv)
    logger.info("=" * 50)

    # 1. Excel'den katsayilar oku, shell EID'leri cikar
    coeff_df = read_coefficients(excel_path)
    shell_eid_map = extract_shell_eids_from_coefficients(coeff_df)

    # 2. H5'ten kuvvetler oku
    bar_df, shell_df = read_prediction_h5(h5_path)

    if bar_df.empty:
        raise ValueError("Prediction H5'te bar verisi bulunamadi")
    if shell_df.empty:
        raise ValueError("Prediction H5'te shell verisi bulunamadi")
    if not shell_eid_map:
        raise ValueError("Excel'den connectivity bilgisi cikarilmadi")

    # 3. Tahmin
    result = generate_predictions(coeff_df, bar_df, shell_df, shell_eid_map)

    # 4. CSV yaz
    result.to_csv(output_csv, index=False)
    logger.info("Tahmin CSV yazildi: %s (%d satir)", output_csv, len(result))

    return result


# ============================================================
#  GUI
# ============================================================


class PredictedLoadApp(tk.Tk):
    """Predicted Load Generation arayuzu."""

    def __init__(self):
        super().__init__()
        self.title("Predicted Load Generation")
        self.geometry("650x420")
        self.resizable(False, False)

        self._excel_var = tk.StringVar()
        self._h5_var = tk.StringVar()
        self._output_var = tk.StringVar(value="predicted_loads.csv")

        self._build_ui()

    def _build_ui(self):
        # Ana frame
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        # Baslik
        ttk.Label(
            main, text="Predicted Load Generation",
            font=("Helvetica", 14, "bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(0, 15))

        # --- Excel ---
        ttk.Label(main, text="Correlation Output Excel:").grid(
            row=1, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self._excel_var, width=50).grid(
            row=1, column=1, padx=5, pady=4)
        ttk.Button(main, text="Sec...", width=6,
                    command=self._browse_excel).grid(row=1, column=2, pady=4)

        # --- H5 ---
        ttk.Label(main, text="Prediction H5:").grid(
            row=2, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self._h5_var, width=50).grid(
            row=2, column=1, padx=5, pady=4)
        ttk.Button(main, text="Sec...", width=6,
                    command=self._browse_h5).grid(row=2, column=2, pady=4)

        # --- Output ---
        ttk.Label(main, text="Output CSV:").grid(
            row=3, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self._output_var, width=50).grid(
            row=3, column=1, padx=5, pady=4)
        ttk.Button(main, text="Sec...", width=6,
                    command=self._browse_output).grid(row=3, column=2, pady=4)

        # --- Generate butonu ---
        ttk.Button(
            main, text="Generate Predicted Loads",
            command=self._run,
        ).grid(row=4, column=0, columnspan=3, pady=20)

        # --- Status ---
        self._status_var = tk.StringVar(value="Hazir.")
        ttk.Label(main, textvariable=self._status_var,
                  foreground="gray").grid(row=5, column=0, columnspan=3, sticky="w")

        # --- Log text ---
        self._log_text = tk.Text(main, height=8, width=75, state="disabled",
                                  font=("Consolas", 9))
        self._log_text.grid(row=6, column=0, columnspan=3, pady=(10, 0))

    def _browse_excel(self):
        path = filedialog.askopenfilename(
            title="Correlation Output Excel Sec",
            filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")],
        )
        if path:
            self._excel_var.set(path)
            # Output otomatik ayarla
            base = Path(path).stem
            self._output_var.set(
                str(Path(path).parent / f"{base}_predicted.csv")
            )

    def _browse_h5(self):
        path = filedialog.askopenfilename(
            title="Prediction H5 Sec",
            filetypes=[("HDF5", "*.h5 *.hdf5"), ("All", "*.*")],
        )
        if path:
            self._h5_var.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Output CSV Kaydet",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if path:
            self._output_var.set(path)

    def _log(self, msg: str):
        self._log_text.config(state="normal")
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state="disabled")
        self.update_idletasks()

    def _run(self):
        excel_path = self._excel_var.get().strip()
        h5_path = self._h5_var.get().strip()
        output_csv = self._output_var.get().strip()

        # Validasyon
        if not excel_path:
            messagebox.showwarning("Uyari", "Correlation Output Excel secin!")
            return
        if not h5_path:
            messagebox.showwarning("Uyari", "Prediction H5 secin!")
            return
        if not output_csv:
            messagebox.showwarning("Uyari", "Output CSV yolu girin!")
            return
        if not Path(excel_path).exists():
            messagebox.showerror("Hata", f"Excel bulunamadi:\n{excel_path}")
            return
        if not Path(h5_path).exists():
            messagebox.showerror("Hata", f"H5 bulunamadi:\n{h5_path}")
            return

        # Log temizle
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state="disabled")

        self._status_var.set("Calisiyor...")
        self._log(f"Excel: {excel_path}")
        self._log(f"H5:    {h5_path}")

        try:
            # 1. Katsayilar
            self._log("Katsayilar okunuyor...")
            coeff_df = read_coefficients(excel_path)
            self._log(f"  {len(coeff_df)} katsayi satiri okundu")

            # 2. Connectivity
            self._log("Connectivity okunuyor...")
            shell_eid_map = read_connectivity(excel_path)
            self._log(f"  {len(shell_eid_map)} bar element icin connectivity")

            if not shell_eid_map:
                messagebox.showerror(
                    "Hata",
                    "Total Summary sheet'inden connectivity cikarilmadi.\n"
                    "Excel dosyasinda 'Total Summary' sheet'i ve "
                    "'Bar_EID'/'Shell_EID' kolonlari oldugundan emin olun."
                )
                self._status_var.set("Hata: connectivity bulunamadi")
                return

            # 3. H5 oku
            self._log("Prediction H5 okunuyor...")
            bar_df, shell_df = read_prediction_h5(h5_path)

            if bar_df.empty:
                messagebox.showerror("Hata", "H5'te bar element verisi bulunamadi")
                self._status_var.set("Hata: bar verisi yok")
                return
            if shell_df.empty:
                messagebox.showerror("Hata", "H5'te shell force verisi bulunamadi")
                self._status_var.set("Hata: shell verisi yok")
                return

            self._log(f"  Bar: {len(bar_df)} satir, Shell: {len(shell_df)} satir")

            # 4. Tahmin
            self._log("Tahmin hesaplaniyor...")
            result = generate_predictions(coeff_df, bar_df, shell_df, shell_eid_map)

            if result.empty:
                messagebox.showwarning(
                    "Uyari", "Hicbir tahmin uretilmedi.\n"
                    "Bar EID'ler ve Shell EID'ler H5 ile eslesmiyor olabilir."
                )
                self._status_var.set("Uyari: sonuc bos")
                return

            # 5. CSV yaz
            result.to_csv(output_csv, index=False)
            self._log(f"CSV yazildi: {output_csv}")
            self._log(f"  {len(result)} satir, {result['Bar_EID'].nunique()} unique bar")

            self._status_var.set(f"Tamamlandi! {len(result)} satir -> {output_csv}")
            messagebox.showinfo(
                "Basarili",
                f"Predicted loads olusturuldu!\n\n"
                f"Satir: {len(result)}\n"
                f"Bar element: {result['Bar_EID'].nunique()}\n"
                f"Subcase: {result['Subcase_ID'].nunique()}\n\n"
                f"CSV: {output_csv}",
            )

        except Exception as e:
            logger.error("Hata: %s", str(e), exc_info=True)
            self._log(f"HATA: {e}")
            self._status_var.set(f"Hata: {e}")
            messagebox.showerror("Hata", f"Islem sirasinda hata olustu:\n\n{e}")


# ============================================================
#  Giris Noktasi
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Predicted Load Generation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornek:
  python predicted_load_generation.py
  python predicted_load_generation.py --excel output.xlsx --h5 pred.h5
  python predicted_load_generation.py --excel output.xlsx --h5 pred.h5 --output result.csv
        """,
    )
    parser.add_argument(
        "--excel", default=None,
        help="Joint Load Correlation Output Excel dosyasi",
    )
    parser.add_argument(
        "--h5", default=None,
        help="Prediction H5 dosyasi (ELFORCE_BAR_COMBINED + ELFORCE_SHELL_COMBINED)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Cikti CSV dosyasi (varsayilan: <excel_adi>_predicted.csv)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Detayli log",
    )

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # CLI modu: her iki dosya da verilmisse
    if args.excel and args.h5:
        output = args.output
        if not output:
            base = Path(args.excel).stem
            output = str(Path(args.excel).parent / f"{base}_predicted.csv")

        result = run_prediction(args.excel, args.h5, output)
        print(f"\nSonuc: {len(result)} tahmin satiri -> {output}")
        return

    # GUI modu
    try:
        app = PredictedLoadApp()
        app.mainloop()
    except Exception as e:
        print(f"GUI baslatilmadi: {e}")
        print("CLI kullanimi: python predicted_load_generation.py --excel X --h5 Y")
        sys.exit(1)


if __name__ == "__main__":
    main()
