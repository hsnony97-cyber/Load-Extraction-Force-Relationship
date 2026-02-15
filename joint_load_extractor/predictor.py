"""
Predictor Module

Predicted vs Actual sheet'indeki mantikla birebir ayni:
  predictor = [Bar_Axial, Shell_Nx, Shell_Ny, Shell_Nxy]
  predicted = predictor @ coefficients + intercept

Prediction H5 yapisi:
  - ELFORCE_BAR_COMBINED/table:   Element_ID, Subcase_ID, AF
  - ELFORCE_SHELL_COMBINED/table: Element_ID, Subcase_ID, MX(=NX), MY(=NY), MXY(=NXY)

Correlation Summary Excel (Total Summary sheet):
  Bar_EID | Element_Type | Shell_EID | Target |
  Coeff_Bar_Axial | Coeff_Shell_Nx | Coeff_Shell_Ny | Coeff_Shell_Nxy |
  Intercept | R2

Cikti CSV (wide format - her subcase icin tek satir):
  Bar_EID, Element_Type, Subcase_ID,
  Pred_FX, Pred_FY, Pred_NX, Pred_NY, Pred_NXY
"""

import logging
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from joint_load_extractor.correlation import JointCorrelationResult, RegressionEquation

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

# Cikti kolon sirasi
OUTPUT_COLUMNS = [
    "Bar_EID", "Element_Type", "Subcase_ID",
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
#  Coefficient Reader (Excel Total Summary)
# ============================================================


def read_coefficients_from_excel(excel_path: str) -> pd.DataFrame:
    """
    Correlation Summary Excel dosyasindaki 'Total Summary' sheet'ini oku.
    """
    logger.info("Katsayilar Excel'den okunuyor: %s", excel_path)

    try:
        df = pd.read_excel(excel_path, sheet_name="Total Summary")
    except ValueError:
        xls = pd.ExcelFile(excel_path)
        logger.info("Excel sheet'leri: %s", xls.sheet_names)
        found = None
        for name in xls.sheet_names:
            if "total" in name.lower() and "summary" in name.lower():
                found = name
                break
        if found is None:
            for name in xls.sheet_names:
                if "summary" in name.lower():
                    found = name
                    break
        if found is None:
            raise ValueError(
                f"Excel dosyasinda 'Total Summary' sheet'i bulunamadi. "
                f"Mevcut sheet'ler: {xls.sheet_names}"
            )
        df = pd.read_excel(excel_path, sheet_name=found)
        logger.info("'%s' sheet'i kullanildi", found)

    required = ["Bar_EID", "Element_Type", "Shell_EID", "Target", "Intercept"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Total Summary'de eksik kolonlar: {missing}. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    logger.info("  %d katsayi satiri okundu", len(df))
    logger.info("  Unique Bar_EID: %d, Shell_EID: %d",
               df["Bar_EID"].nunique(), df["Shell_EID"].nunique())
    return df


# ============================================================
#  Prediction Functions
# ============================================================


def predict_from_h5(
    per_shell_results: List[JointCorrelationResult],
    prediction_h5_path: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    In-memory korelasyon sonuclari + Prediction H5 -> tahmin CSV.
    Predicted vs Actual sheet'indeki mantikla birebir ayni.
    """
    bar_df, shell_df = read_prediction_h5(prediction_h5_path)

    if bar_df.empty:
        logger.error("Prediction H5'te bar element verisi bulunamadi")
        return pd.DataFrame()
    if shell_df.empty:
        logger.error("Prediction H5'te shell force verisi bulunamadi")
        return pd.DataFrame()

    # In-memory sonuclardan katsayi tablosu olustur
    coeff_rows = []
    for res in per_shell_results:
        if not res.equations:
            continue
        for eq in res.equations:
            row = {
                "Bar_EID": res.bar_eid,
                "Element_Type": res.element_type,
                "Shell_EID": res.shell_eid if res.shell_eid is not None else "",
                "Target": eq.target_name,
            }
            for pname, coeff in zip(eq.predictor_names, eq.coefficients):
                row[f"Coeff_{pname}"] = float(coeff)
            row["Intercept"] = float(eq.intercept)
            coeff_rows.append(row)

    if not coeff_rows:
        logger.warning("Katsayi verisi bulunamadi")
        return pd.DataFrame()

    coeff_df = pd.DataFrame(coeff_rows)
    return _run_prediction(coeff_df, bar_df, shell_df, output_csv)


def predict_from_excel_and_h5(
    coefficients_excel: str,
    prediction_h5_path: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Bagimsiz tahmin: Correlation Summary Excel + Prediction H5 -> CSV.
    Onceki analizden tamamen bagimsiz calisir.
    """
    coeff_df = read_coefficients_from_excel(coefficients_excel)
    bar_df, shell_df = read_prediction_h5(prediction_h5_path)

    if bar_df.empty:
        logger.error("Prediction H5'te bar element verisi bulunamadi")
        return pd.DataFrame()
    if shell_df.empty:
        logger.error("Prediction H5'te shell force verisi bulunamadi")
        return pd.DataFrame()

    return _run_prediction(coeff_df, bar_df, shell_df, output_csv)


def predict_from_results(
    per_shell_results: List[JointCorrelationResult],
    bar_forces: Dict,
    shell_forces: Dict,
    output_csv: str,
) -> pd.DataFrame:
    """
    Mevcut OP2 verileri ve korelasyon katsayilari ile tahmin yap.
    Predicted vs Actual mantigi ile birebir ayni:
      predictor = [bar_axial, shell_nx, shell_ny, shell_nxy]
      predicted = predictor @ coefficients + intercept
    """
    rows = []

    for res in per_shell_results:
        if not res.equations:
            continue
        bar_eid = res.bar_eid
        shell_eid = res.shell_eid
        element_type = res.element_type
        if shell_eid is None:
            continue

        subcases = res.matched_subcases if res.matched_subcases else []

        # Predictor data - aynen reporter gibi
        pred_names = list(res.predictor_data.keys()) if res.predictor_data else []
        n = min(len(v) for v in res.predictor_data.values()) if res.predictor_data else 0

        if n == 0:
            continue

        X = np.column_stack([res.predictor_data[k][:n] for k in pred_names])
        sc_list = subcases[:n]

        # Her subcase icin tum target'lari tek satirda topla
        for i in range(n):
            sc_id = sc_list[i] if i < len(sc_list) else 0
            row = {
                "Bar_EID": bar_eid,
                "Element_Type": element_type,
                "Subcase_ID": sc_id,
            }

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


def predict_from_dataframes(
    coefficients_df: pd.DataFrame,
    bar_forces_df: pd.DataFrame,
    shell_forces_df: pd.DataFrame,
    output_csv: str,
) -> pd.DataFrame:
    """DataFrame formatindaki verilerle tahmin yap (eski uyumluluk)."""
    bar_df = _normalize_bar_df(bar_forces_df)
    shell_df = _normalize_shell_df(shell_forces_df)
    return _run_prediction(coefficients_df, bar_df, shell_df, output_csv)


# ============================================================
#  Core Prediction Engine
# ============================================================


def _run_prediction(
    coeff_df: pd.DataFrame,
    bar_df: pd.DataFrame,
    shell_df: pd.DataFrame,
    output_csv: str,
) -> pd.DataFrame:
    """
    Predicted vs Actual mantigi ile birebir ayni tahmin.

    Her (bar_eid, element_type, shell_eid) icin:
      predictor = [bar_axial(AF), shell_nx, shell_ny, shell_nxy]
      her target icin: predicted = predictor @ coefficients + intercept

    Cikti wide format: her subcase icin tek satir,
    tum target tahminleri ayri kolonlarda.
    """
    bar_df = _normalize_bar_df(bar_df)
    shell_df = _normalize_shell_df(shell_df)

    logger.info("Prediction engine: bar %d satir, shell %d satir, coeff %d satir",
                len(bar_df), len(shell_df), len(coeff_df))

    rows = []

    # Unique (bar_eid, element_type, shell_eid) gruplari
    groups = coeff_df.groupby(["Bar_EID", "Element_Type", "Shell_EID"])

    for (bar_eid, et, shell_eid), group in groups:
        # Bu grubun katsayilari (her target icin)
        eq_map = {}
        for _, coeff_row in group.iterrows():
            target = coeff_row["Target"]
            eq_map[target] = {
                "coeffs": np.array([
                    coeff_row.get("Coeff_Bar_Axial", 0),
                    coeff_row.get("Coeff_Shell_Nx", 0),
                    coeff_row.get("Coeff_Shell_Ny", 0),
                    coeff_row.get("Coeff_Shell_Nxy", 0),
                ], dtype=float),
                "intercept": float(coeff_row.get("Intercept", 0)),
            }

        bar_mask = bar_df["bar_eid"] == int(bar_eid)
        bar_subset = bar_df[bar_mask]

        shell_mask = shell_df["element_id"] == int(shell_eid)
        shell_subset = shell_df[shell_mask]

        if bar_subset.empty or shell_subset.empty:
            continue

        # Subcase bazinda eslestir
        for _, bar_row in bar_subset.iterrows():
            sc_id = bar_row["subcase_id"]
            af = float(bar_row["af"])

            shell_sc = shell_subset[shell_subset["subcase_id"] == sc_id]
            if shell_sc.empty:
                continue

            nx = float(shell_sc["nx"].mean())
            ny = float(shell_sc["ny"].mean())
            nxy = float(shell_sc["nxy"].mean())

            # Predictor vektoru - aynen Predicted vs Actual'daki gibi
            predictor = np.array([af, nx, ny, nxy])

            row = {
                "Bar_EID": int(bar_eid),
                "Element_Type": int(et),
                "Subcase_ID": int(sc_id),
            }

            # Her target icin: predicted = predictor @ coefficients + intercept
            for target_name, eq_info in eq_map.items():
                predicted = float(predictor @ eq_info["coeffs"] + eq_info["intercept"])
                col_name = TARGET_COL_MAP.get(
                    target_name,
                    f"Pred_{target_name.replace(' ', '_')}"
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


def _order_output(df: pd.DataFrame) -> pd.DataFrame:
    """Cikti kolon siralamasini duzenle."""
    present = [c for c in OUTPUT_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    return df[present + extra]


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
