"""
Predictor Module

Korelasyon analizinden elde edilen katsayilari (Correlation Summary Excel)
ve yeni veriler (Prediction H5) kullanarak
Bearing ve Bypass yuklerini tahmin eder.

Prediction H5 yapisi:
  - ELFORCE_BAR_COMBINED/table:   Element_ID, Subcase_ID, AF
  - ELFORCE_SHELL_COMBINED/table: Element_ID, Subcase_ID, MX(=NX), MY(=NY), MXY(=NXY)

Correlation Summary Excel (Total Summary sheet):
  Bar_EID | Element_Type | Subcase_ID | Shell_EID | Shell_Type |
  Target | Coeff_Bar_Axial | Coeff_Shell_Nx | Coeff_Shell_Ny |
  Coeff_Shell_Nxy | Intercept | R2

Cikti CSV:
  Bar_EID, Element_Type, Subcase_ID, Shell_EID,
  AF, NX, NY, NXY,
  Pred_F_Bearing_X, Pred_F_Bearing_Y,
  Pred_NX_Bypass, Pred_NY_Bypass, Pred_NXY_Bypass
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

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (bar_forces_df, shell_forces_df) - normalize edilmis kolonlarla
    """
    logger.info("Prediction H5 okunuyor: %s", h5_path)

    bar_df = pd.DataFrame()
    shell_df = pd.DataFrame()

    with h5py.File(h5_path, "r") as f:
        # --- Bar element combined ---
        if BAR_TABLE_PATH in f:
            bar_df = _read_h5_table(f, BAR_TABLE_PATH)
            # Element_ID -> Bar_EID
            if "Element_ID" in bar_df.columns:
                bar_df = bar_df.rename(columns={"Element_ID": "Bar_EID"})
            logger.info("Bar tablosu okundu: %s (%d satir, kolonlar: %s)",
                       BAR_TABLE_PATH, len(bar_df), list(bar_df.columns))
        else:
            logger.warning("Prediction H5'te '%s' bulunamadi", BAR_TABLE_PATH)
            # Mevcut tum dataset'leri listele
            _log_h5_structure(f)

        # --- Shell force combined ---
        if SHELL_TABLE_PATH in f:
            shell_df = _read_h5_table(f, SHELL_TABLE_PATH)
            # Element_ID -> Shell_EID, MX->NX, MY->NY, MXY->NXY
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

    Beklenen kolonlar:
      Bar_EID, Element_Type, Subcase_ID, Shell_EID, Shell_Type,
      Target, Coeff_Bar_Axial, Coeff_Shell_Nx, Coeff_Shell_Ny,
      Coeff_Shell_Nxy, Intercept, R2

    Returns
    -------
    pd.DataFrame
        Katsayi tablosu.
    """
    logger.info("Katsayilar Excel'den okunuyor: %s", excel_path)

    try:
        df = pd.read_excel(excel_path, sheet_name="Total Summary")
    except ValueError:
        # Sheet adi farkli olabilir
        xls = pd.ExcelFile(excel_path)
        logger.info("Excel sheet'leri: %s", xls.sheet_names)
        # 'Total Summary' veya 'Summary' iceren ilk sheet'i bul
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

    Analiz tamamlandiktan hemen sonra cagirilir.
    Katsayilar hafizadaki per_shell_results'tan alinir.
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
            }
            row["Target"] = eq.target_name
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
    Excel'den katsayilari, H5'ten bar/shell kuvvetlerini okur.

    Parameters
    ----------
    coefficients_excel : str
        Correlation Summary Excel dosyasi (Total Summary sheet'i iceren).
    prediction_h5_path : str
        Prediction H5 dosyasi (ELFORCE_BAR_COMBINED + ELFORCE_SHELL_COMBINED).
    output_csv : str
        Cikti CSV dosya yolu.
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
    (Prediction H5 verilmediginde kullanilan fallback.)
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

        eq_map = {eq.target_name: eq for eq in res.equations}
        subcases = res.matched_subcases if res.matched_subcases else []

        for sc_id in subcases:
            bar_result = bar_forces.get((sc_id, bar_eid))
            if bar_result is None:
                continue
            af = float(np.mean(bar_result.axial_force))

            sf = shell_forces.get((sc_id, shell_eid))
            if sf is None:
                continue
            nx = float(np.mean(sf.membrane_x))
            ny = float(np.mean(sf.membrane_y))
            nxy = float(np.mean(sf.membrane_xy))

            row = {
                "Bar_EID": bar_eid,
                "Element_Type": element_type,
                "Subcase_ID": sc_id,
                "Shell_EID": shell_eid,
                "AF": af, "NX": nx, "NY": ny, "NXY": nxy,
            }
            predictors = np.array([af, nx, ny, nxy])
            for target_name, eq in eq_map.items():
                pred_val = float(np.dot(eq.coefficients, predictors) + eq.intercept)
                row[_target_to_col(target_name)] = pred_val
            rows.append(row)

    if not rows:
        logger.warning("Tahmin icin veri bulunamadi")
        return pd.DataFrame()

    df = _order_columns(pd.DataFrame(rows))
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
    Katsayilar + bar/shell kuvvetleri -> tahmin CSV.

    coeff_df: Bar_EID, Element_Type, Shell_EID, Target, Coeff_*, Intercept
    bar_df:   Bar_EID (veya bar_eid), Subcase_ID, AF
    shell_df: Shell_EID (veya element_id), Subcase_ID, NX, NY, NXY
    """
    # Kolon isimlerini normalize et
    bar_df = _normalize_bar_df(bar_df)
    shell_df = _normalize_shell_df(shell_df)

    logger.info("Prediction engine: bar %d satir, shell %d satir, coeff %d satir",
                len(bar_df), len(shell_df), len(coeff_df))
    logger.info("  Bar kolonlar: %s", list(bar_df.columns))
    logger.info("  Shell kolonlar: %s", list(shell_df.columns))
    logger.info("  Coeff kolonlar: %s", list(coeff_df.columns))

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

        # Bar element icin tum subcase'lerdeki AF
        bar_mask = bar_df["bar_eid"] == int(bar_eid)
        bar_subset = bar_df[bar_mask]

        # Shell element icin tum subcase'lerdeki NX, NY, NXY
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

            row = {
                "Bar_EID": int(bar_eid),
                "Element_Type": int(et),
                "Subcase_ID": int(sc_id),
                "Shell_EID": int(shell_eid),
                "AF": af, "NX": nx, "NY": ny, "NXY": nxy,
            }

            predictors = np.array([af, nx, ny, nxy])
            for target_name, eq_info in eq_map.items():
                pred_val = float(np.dot(eq_info["coeffs"], predictors) + eq_info["intercept"])
                row[_target_to_col(target_name)] = pred_val

            rows.append(row)

    if not rows:
        logger.warning("Tahmin icin eslesen veri bulunamadi")
        return pd.DataFrame()

    df = _order_columns(pd.DataFrame(rows))
    df.to_csv(output_csv, index=False)
    logger.info("Tahmin CSV yazildi: %s (%d satir)", output_csv, len(df))
    return df


# ============================================================
#  Helper Functions
# ============================================================


def _target_to_col(target_name: str) -> str:
    """Hedef ismini tahmin kolon ismine donustur."""
    mapping = {
        "F Bearing X": "Pred_F_Bearing_X",
        "F Bearing Y": "Pred_F_Bearing_Y",
        "NX Bypass": "Pred_NX_Bypass",
        "NY Bypass": "Pred_NY_Bypass",
        "NXY Bypass": "Pred_NXY_Bypass",
    }
    return mapping.get(target_name, f"Pred_{target_name.replace(' ', '_')}")


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Kolon siralamasini duzenle."""
    ordered_cols = [
        "Bar_EID", "Element_Type", "Subcase_ID", "Shell_EID",
        "AF", "NX", "NY", "NXY",
        "Pred_F_Bearing_X", "Pred_F_Bearing_Y",
        "Pred_NX_Bypass", "Pred_NY_Bypass", "Pred_NXY_Bypass",
    ]
    present = [c for c in ordered_cols if c in df.columns]
    extra = [c for c in df.columns if c not in ordered_cols]
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
