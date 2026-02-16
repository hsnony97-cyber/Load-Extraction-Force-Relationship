"""
Predictor Module

Predicted vs Actual sheet'indeki mantikla birebir ayni:
  predictor = [Bar_Axial, Shell_Nx, Shell_Ny, Shell_Nxy]
  predicted = predictor @ coefficients + intercept

correlation_results kullanir (ortalama shell kuvvetleri).
Her (bar_eid, element_type) icin TEK sonuc uretir.

Prediction H5 yapisi:
  - ELFORCE_BAR_COMBINED/table:   Element_ID, Subcase_ID, AF
  - ELFORCE_SHELL_COMBINED/table: Element_ID, Subcase_ID, MX(=NX), MY(=NY), MXY(=NXY)

Correlation Summary Excel (Correlation Summary sheet):
  Bar_EID | Element_Type | Target |
  Coeff_Bar_Axial | Coeff_Shell_Nx | Coeff_Shell_Ny | Coeff_Shell_Nxy |
  Intercept | R_Squared

Cikti CSV (wide format - her subcase icin tek satir):
  Bar_EID, Element_Type, Subcase_ID,
  Pred_FX, Pred_FY, Pred_NX, Pred_NY, Pred_NXY
"""

import logging
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from joint_load_extractor.correlation import JointCorrelationResult

logger = logging.getLogger(__name__)

# Prediction H5 sabit tablo yollari
BAR_TABLE_PATH = "ELFORCE_BAR_COMBINED/table"
SHELL_TABLE_PATH = "ELFORCE_SHELL_COMBINED/table"

# Target -> CSV kolon adi
TARGET_COL_MAP = {
    "F Bearing X": "Pred_FX",
    "F Bearing Y": "Pred_FY",
    "NX Bypass": "Pred_NX",
    "NY Bypass": "Pred_NY",
    "NXY Bypass": "Pred_NXY",
}

# Cikti kolon sirasi (sabit kolonlar; per-shell kolonlar dinamik eklenir)
OUTPUT_FIXED_COLUMNS = [
    "Bar_EID", "Element_Type", "Subcase_ID",
]
OUTPUT_PRED_COLUMNS = [
    "Pred_FX", "Pred_FY", "Pred_NX", "Pred_NY", "Pred_NXY",
]


# ============================================================
#  Prediction H5 Reader
# ============================================================


def read_prediction_h5(h5_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prediction H5 dosyasini oku.

    Sabit yollar:
      - ELFORCE_BAR_COMBINED/table  -> Element_ID, Subcase_ID, AF
      - ELFORCE_SHELL_COMBINED/table -> Element_ID, Subcase_ID, MX, MY, MXY

    MX->NX, MY->NY, MXY->NXY olarak yeniden adlandirilir.
    """
    logger.info("Prediction H5 okunuyor: %s", h5_path)

    bar_df = pd.DataFrame()
    shell_df = pd.DataFrame()

    with h5py.File(h5_path, "r") as f:
        # --- Bar element combined ---
        if BAR_TABLE_PATH in f:
            bar_df = _read_h5_table(f, BAR_TABLE_PATH)
            if "Element_ID" in bar_df.columns:
                bar_df = bar_df.rename(columns={"Element_ID": "Bar_EID"})
            logger.info("Bar tablosu okundu: %s (%d satir, kolonlar: %s)",
                       BAR_TABLE_PATH, len(bar_df), list(bar_df.columns))
        else:
            logger.warning("Prediction H5'te '%s' bulunamadi", BAR_TABLE_PATH)
            _log_h5_structure(f)

        # --- Shell force combined ---
        if SHELL_TABLE_PATH in f:
            shell_df = _read_h5_table(f, SHELL_TABLE_PATH)
            rename_map = {}
            if "Element_ID" in shell_df.columns:
                rename_map["Element_ID"] = "Shell_EID"
            if "MX" in shell_df.columns:
                rename_map["MX"] = "NX"
            if "MY" in shell_df.columns:
                rename_map["MY"] = "NY"
            if "MXY" in shell_df.columns:
                rename_map["MXY"] = "NXY"
            shell_df = shell_df.rename(columns=rename_map)
            logger.info("Shell tablosu okundu: %s (%d satir, kolonlar: %s)",
                       SHELL_TABLE_PATH, len(shell_df), list(shell_df.columns))
        else:
            logger.warning("Prediction H5'te '%s' bulunamadi", SHELL_TABLE_PATH)
            _log_h5_structure(f)

    return bar_df, shell_df


def _read_h5_table(f: h5py.File, path: str) -> pd.DataFrame:
    """H5 dataset'ini DataFrame'e cevir."""
    dataset = f[path]
    if dataset.dtype.names:
        data = {}
        for col_name in dataset.dtype.names:
            col_data = dataset[col_name]
            if col_data.dtype.kind in ("S", "O"):
                try:
                    col_data = np.array(
                        [x.decode("utf-8") if isinstance(x, bytes) else x for x in col_data]
                    )
                except (UnicodeDecodeError, AttributeError):
                    pass
            data[col_name] = col_data
        return pd.DataFrame(data)
    return pd.DataFrame(dataset[:])


def _log_h5_structure(f: h5py.File):
    """H5 dosyasindaki tum gruplari/dataset'leri logla."""
    paths = []

    def _collect(group, prefix=""):
        for key in group:
            path = f"{prefix}/{key}" if prefix else key
            item = group[key]
            if isinstance(item, h5py.Group):
                _collect(item, path)
            elif isinstance(item, h5py.Dataset):
                cols = list(item.dtype.names) if item.dtype.names else []
                paths.append((path, cols))

    _collect(f)
    logger.info("H5 yapisi (%d dataset):", len(paths))
    for path, cols in paths:
        logger.info("  %s: %s", path, cols[:10])


# ============================================================
#  Coefficient Reader (Excel Correlation Summary)
# ============================================================


def read_coefficients_from_excel(excel_path: str) -> pd.DataFrame:
    """
    Correlation Summary Excel dosyasindaki 'Correlation Summary' sheet'ini oku.
    Bu sheet ortalama shell kuvvetleri ile olusturulan katsayilari icerir.
    """
    logger.info("Katsayilar Excel'den okunuyor: %s", excel_path)

    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names
    logger.info("Excel sheet'leri: %s", sheet_names)

    # Oncelik: Correlation Summary > Total Summary > *summary*
    df = None
    for candidate in ["Correlation Summary", "Total Summary"]:
        if candidate in sheet_names:
            df = pd.read_excel(excel_path, sheet_name=candidate)
            logger.info("'%s' sheet'i kullanildi", candidate)
            break

    if df is None:
        for name in sheet_names:
            if "correlation" in name.lower() and "summary" in name.lower():
                df = pd.read_excel(excel_path, sheet_name=name)
                logger.info("'%s' sheet'i kullanildi", name)
                break

    if df is None:
        for name in sheet_names:
            if "summary" in name.lower():
                df = pd.read_excel(excel_path, sheet_name=name)
                logger.info("'%s' sheet'i kullanildi", name)
                break

    if df is None:
        raise ValueError(
            f"Excel dosyasinda uygun summary sheet'i bulunamadi. "
            f"Mevcut sheet'ler: {sheet_names}"
        )

    # Gerekli kolonlar (Correlation Summary ve Total Summary uyumlu)
    required = ["Bar_EID", "Element_Type", "Target", "Intercept"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Summary sheet'te eksik kolonlar: {missing}. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    # Dinamik Coeff_* kolonlarini bul (per-shell: Coeff_Shell_XXXX_Nx, ...)
    coeff_cols = [c for c in df.columns if c.startswith("Coeff_")]
    numeric_cols = coeff_cols + ["Intercept"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Bar_EID", "Element_Type"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # String satir temizligi: Bar_EID NaN olan satirlari at
    df = df.dropna(subset=["Bar_EID"])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    logger.info("  %d katsayi satiri okundu, %d Coeff kolonu",
                len(df), len(coeff_cols))
    logger.info("  Coeff kolonlari: %s", coeff_cols[:10])
    logger.info("  Unique Bar_EID: %d", df["Bar_EID"].nunique())
    return df


def _extract_shell_eids_from_coefficients(coeff_df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Katsayi DataFrame'inden bar_eid -> [shell_eids] eslesmesini cikar.

    Iki kaynak:
    1. Shell_EIDs kolonu (virgul ayirmali: "2001,2002,2003")
    2. Coeff_Shell_XXXX_Nx kolon isimlerinden parse
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

    # Yontem 2: Coeff_Shell_XXXX_Nx kolon isimlerinden
    if not shell_map:
        shell_eid_pattern = re.compile(r"Coeff_Shell_(\d+)_N[xXyY]+")
        all_shell_eids = set()
        for col in coeff_df.columns:
            m = shell_eid_pattern.match(col)
            if m:
                all_shell_eids.add(int(m.group(1)))

        if all_shell_eids:
            # Her bar icin: sadece degeri NaN olmayan shell EID'leri
            for bar_eid, grp in coeff_df.groupby("Bar_EID"):
                row = grp.iloc[0]
                eids = []
                for seid in sorted(all_shell_eids):
                    col_nx = f"Coeff_Shell_{seid}_Nx"
                    if col_nx in row.index and pd.notna(row.get(col_nx)):
                        eids.append(seid)
                if eids:
                    shell_map[int(float(bar_eid))] = eids

    logger.info("  %d bar element icin shell connectivity bulundu", len(shell_map))
    return shell_map


# ============================================================
#  Prediction Functions
# ============================================================


def predict_from_results(
    correlation_results: List[JointCorrelationResult],
    output_csv: str,
) -> pd.DataFrame:
    """
    In-memory korelasyon sonuclariyla tahmin (Predicted vs Actual mantigi).

    Per-shell predictor'lar kullanir:
      X = [Bar_Axial, Shell_{EID1}_Nx, Shell_{EID1}_Ny, Shell_{EID1}_Nxy, ...]
      predicted = X @ coefficients + intercept
    """
    rows = []

    for res in correlation_results:
        if not res.equations or not res.predictor_data:
            continue

        pred_names = list(res.predictor_data.keys())
        n = min(len(v) for v in res.predictor_data.values())
        if n == 0:
            continue

        X = np.column_stack([res.predictor_data[k][:n] for k in pred_names])
        subcases = res.matched_subcases[:n] if res.matched_subcases else [0] * n

        for i in range(n):
            sc_id = subcases[i] if i < len(subcases) else 0
            row = {
                "Bar_EID": res.bar_eid,
                "Element_Type": res.element_type,
                "Subcase_ID": sc_id,
            }
            # Per-shell predictor degerleri
            for col_idx, pname in enumerate(pred_names):
                row[pname] = float(X[i, col_idx])

            for eq in res.equations:
                predicted = float(X[i] @ eq.coefficients + eq.intercept)
                col_name = TARGET_COL_MAP.get(
                    eq.target_name,
                    f"Pred_{eq.target_name.replace(' ', '_')}"
                )
                row[col_name] = predicted

            rows.append(row)

    if not rows:
        logger.warning("Tahmin icin veri bulunamadi")
        return pd.DataFrame()

    df = _order_output(pd.DataFrame(rows))
    df.to_csv(output_csv, index=False)
    logger.info("Tahmin CSV yazildi: %s (%d satir)", output_csv, len(df))
    return df


def predict_from_h5(
    coefficients_excel: str,
    prediction_h5_path: str,
    output_csv: str,
    connectivity: Dict = None,
) -> pd.DataFrame:
    """
    Output Excel + Prediction H5 -> tahmin CSV.

    Akis:
      1. Output Excel'in Correlation Summary sheet'inden katsayilari oku
      2. Prediction H5'ten bar AF ve shell NX/NY/NXY oku
      3. connectivity ile bagli shell'leri belirle, ortalamasini al
      4. predicted = [AF, avg_NX, avg_NY, avg_NXY] @ coefficients + intercept
      5. CSV yaz
    """
    # 1. Katsayilari Excel'den oku
    coeff_df = read_coefficients_from_excel(coefficients_excel)

    # 2. Prediction H5'ten kuvvetleri oku
    bar_df, shell_df = read_prediction_h5(prediction_h5_path)

    if bar_df.empty:
        logger.error("Prediction H5'te bar element verisi bulunamadi")
        return pd.DataFrame()
    if shell_df.empty:
        logger.error("Prediction H5'te shell force verisi bulunamadi")
        return pd.DataFrame()

    # 3. Shell EID'leri: BDF'den geldiyse kullan, yoksa katsayilardan cikar
    if connectivity:
        shell_eid_map = _build_shell_eid_map(connectivity)
    else:
        shell_eid_map = _extract_shell_eids_from_coefficients(coeff_df)

    return _run_prediction(coeff_df, bar_df, shell_df, output_csv, shell_eid_map)


def predict_from_excel_and_h5(
    coefficients_excel: str,
    prediction_h5_path: str,
    output_csv: str,
    connectivity: Dict = None,
) -> pd.DataFrame:
    """
    Bagimsiz tahmin: Correlation Summary Excel + Prediction H5 -> CSV.
    predict_from_h5 ile ayni islem - iki farkli isimle cagirilabilir.
    """
    return predict_from_h5(
        coefficients_excel=coefficients_excel,
        prediction_h5_path=prediction_h5_path,
        output_csv=output_csv,
        connectivity=connectivity,
    )


def predict_from_dataframes(
    coefficients_df: pd.DataFrame,
    bar_forces_df: pd.DataFrame,
    shell_forces_df: pd.DataFrame,
    output_csv: str,
    shell_eid_map: Dict[int, List[int]] = None,
) -> pd.DataFrame:
    """DataFrame formatindaki verilerle tahmin yap."""
    bar_df = _normalize_bar_df(bar_forces_df)
    shell_df = _normalize_shell_df(shell_forces_df)
    return _run_prediction(coefficients_df, bar_df, shell_df, output_csv, shell_eid_map or {})


# ============================================================
#  Core Prediction Engine
# ============================================================


def _run_prediction(
    coeff_df: pd.DataFrame,
    bar_df: pd.DataFrame,
    shell_df: pd.DataFrame,
    output_csv: str,
    shell_eid_map: Dict[int, List[int]] = None,
) -> pd.DataFrame:
    """
    Per-shell katsayilarla tahmin.

    Her (bar_eid, element_type, subcase) icin:
      predictor = [Bar_Axial, Shell_{EID1}_Nx, Shell_{EID1}_Ny, Shell_{EID1}_Nxy, ...]
      predicted = predictor @ coefficients + intercept

    shell_eid_map: bar_eid -> [shell_eids] (katsayilardan veya BDF'den)
    Her shell'in AYRI NX/NY/NXY degeri kullanilir, ortalama ALINMAZ.
    """
    import re

    if shell_eid_map is None:
        shell_eid_map = {}

    bar_df = _normalize_bar_df(bar_df)
    shell_df = _normalize_shell_df(shell_df)

    logger.info("Prediction engine (per-shell): bar %d, shell %d, coeff %d",
                len(bar_df), len(shell_df), len(coeff_df))

    rows = []
    groups = coeff_df.groupby(["Bar_EID", "Element_Type"])

    for (bar_eid, et), group in groups:
        bar_eid_int = int(float(bar_eid))
        et_int = int(float(et))

        # Bu bar icin bagli shell EID'leri
        connected_shells = shell_eid_map.get(bar_eid_int, [])
        if not connected_shells:
            logger.debug("Bar %d: shell connectivity yok, atlaniyor", bar_eid_int)
            continue

        # Her target icin katsayi vektorunu hazirla
        # Kolon sirasi: Coeff_Bar_Axial, Coeff_Shell_{EID}_Nx, ..._Ny, ..._Nxy, ...
        eq_list = []
        for _, crow in group.iterrows():
            target = str(crow["Target"])
            intercept = float(crow.get("Intercept", 0.0) or 0.0)

            # Katsayi vektoru olustur: [Bar_Axial, Shell1_Nx, Shell1_Ny, Shell1_Nxy, ...]
            coeffs = [float(crow.get("Coeff_Bar_Axial", 0.0) or 0.0)]
            for seid in sorted(connected_shells):
                coeffs.append(float(crow.get(f"Coeff_Shell_{seid}_Nx", 0.0) or 0.0))
                coeffs.append(float(crow.get(f"Coeff_Shell_{seid}_Ny", 0.0) or 0.0))
                coeffs.append(float(crow.get(f"Coeff_Shell_{seid}_Nxy", 0.0) or 0.0))

            eq_list.append((target, np.array(coeffs, dtype=float), intercept))

        # Bar AF verileri
        bar_subset = bar_df[bar_df["bar_eid"] == bar_eid_int]
        if bar_subset.empty:
            bar_subset = bar_df[bar_df["bar_eid"] == float(bar_eid_int)]
        if bar_subset.empty:
            logger.debug("Bar %d: H5 bar verisi yok", bar_eid_int)
            continue

        # Her subcase icin tahmin
        for _, bar_row in bar_subset.iterrows():
            sc_id = bar_row["subcase_id"]
            af = float(bar_row["af"])

            # Her shell'in AYRI NX/NY/NXY degerini al
            predictor = [af]
            predictor_info = {"Bar_Axial": af}
            all_found = True

            for seid in sorted(connected_shells):
                shell_row = shell_df[
                    (shell_df["element_id"] == seid) &
                    (shell_df["subcase_id"] == sc_id)
                ]
                if shell_row.empty:
                    # Float karsilastirma dene
                    shell_row = shell_df[
                        (shell_df["element_id"] == float(seid)) &
                        (shell_df["subcase_id"].between(sc_id - 0.5, sc_id + 0.5))
                    ]
                if shell_row.empty:
                    all_found = False
                    break

                nx_val = float(shell_row["nx"].iloc[0])
                ny_val = float(shell_row["ny"].iloc[0])
                nxy_val = float(shell_row["nxy"].iloc[0])
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

    if not rows:
        logger.warning("Tahmin icin eslesen veri bulunamadi")
        return pd.DataFrame()

    df = _order_output(pd.DataFrame(rows))
    df.to_csv(output_csv, index=False)
    logger.info("Tahmin CSV yazildi: %s (%d satir)", output_csv, len(df))
    return df


# ============================================================
#  Helper Functions
# ============================================================


def _build_shell_eid_map(connectivity) -> Dict[int, List[int]]:
    """Connectivity dict'ten bar_eid -> [shell_eids] mapping olustur."""
    shell_map = {}
    if connectivity is None:
        return shell_map
    for bar_eid, info in connectivity.items():
        eids = list(info.connected_quads.keys()) + list(info.connected_trias.keys())
        if eids:
            shell_map[int(bar_eid)] = eids
    return shell_map


def _order_output(df: pd.DataFrame) -> pd.DataFrame:
    """Cikti kolon siralamasini duzenle: sabit kolonlar + predictor'lar + prediction'lar."""
    fixed = [c for c in OUTPUT_FIXED_COLUMNS if c in df.columns]
    pred_cols = [c for c in OUTPUT_PRED_COLUMNS if c in df.columns]
    # Per-shell predictor kolonlari (Bar_Axial, Shell_*_Nx, ...)
    predictor_cols = [c for c in df.columns
                      if c.startswith("Bar_Axial") or c.startswith("Shell_")]
    # Kalan kolonlar
    used = set(fixed + predictor_cols + pred_cols)
    extra = [c for c in df.columns if c not in used]
    return df[fixed + sorted(predictor_cols) + pred_cols + extra]


def _normalize_bar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Bar forces DataFrame kolon isimlerini normalize et."""
    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_").replace("-", "_")
        if lower in ("bar_eid", "bar_element_id", "bareid", "bar_id", "element_id"):
            col_map[col] = "bar_eid"
        elif lower in ("subcase_id", "subcaseid", "subcase", "sc_id"):
            col_map[col] = "subcase_id"
        elif lower in ("af", "axial_force", "axialforce", "bar_axial", "axial"):
            col_map[col] = "af"

    result = df.rename(columns=col_map)
    for needed in ["bar_eid", "subcase_id", "af"]:
        if needed not in result.columns:
            logger.warning("Bar forces: '%s' kolonu bulunamadi, mevcut: %s",
                          needed, list(result.columns))
    return result


def _normalize_shell_df(df: pd.DataFrame) -> pd.DataFrame:
    """Shell forces DataFrame kolon isimlerini normalize et."""
    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_").replace("-", "_")
        if lower in ("element_id", "elementid", "shell_eid", "shelleid", "eid"):
            col_map[col] = "element_id"
        elif lower in ("subcase_id", "subcaseid", "subcase", "sc_id"):
            col_map[col] = "subcase_id"
        elif lower in ("nx", "mx", "membrane_x", "shell_nx"):
            col_map[col] = "nx"
        elif lower in ("ny", "my", "membrane_y", "shell_ny"):
            col_map[col] = "ny"
        elif lower in ("nxy", "mxy", "membrane_xy", "shell_nxy"):
            col_map[col] = "nxy"

    result = df.rename(columns=col_map)
    for needed in ["element_id", "subcase_id", "nx", "ny", "nxy"]:
        if needed not in result.columns:
            logger.warning("Shell forces: '%s' kolonu bulunamadi, mevcut: %s",
                          needed, list(result.columns))
    return result
