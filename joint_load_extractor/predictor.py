"""
Predictor Module

Korelasyon analizinden elde edilen katsayilari kullanarak
yeni veriler icin Bearing ve Bypass yuklerini tahmin eder.

Girdi formatlari:
  - Bar element combined: Bar_EID, Subcase_ID, AF (Axial Force)
  - Shell force combined: Element_ID, Subcase_ID, NX, NY, NXY

Cikti:
  CSV dosyasi: Bar_EID, Element_Type, Subcase_ID, Shell_EID,
               AF, NX, NY, NXY,
               Pred_F_Bearing_X, Pred_F_Bearing_Y,
               Pred_NX_Bypass, Pred_NY_Bypass, Pred_NXY_Bypass
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from joint_load_extractor.correlation import JointCorrelationResult, RegressionEquation

logger = logging.getLogger(__name__)


def predict_from_results(
    per_shell_results: List[JointCorrelationResult],
    bar_forces: Dict,
    shell_forces: Dict,
    output_csv: str,
) -> pd.DataFrame:
    """
    Mevcut OP2 verileri ve korelasyon katsayilari ile tahmin yap.

    Her (bar_eid, shell_eid, element_type, subcase) icin
    predictor degerlerini alir ve regression denklemini uygular.

    Parameters
    ----------
    per_shell_results : List[JointCorrelationResult]
        Per-shell korelasyon sonuclari (katsayilar icin).
    bar_forces : Dict[(sc_id, bar_eid), BarForceResult]
        OP2 bar kuvvetleri.
    shell_forces : Dict[(sc_id, shell_eid), ShellForceResult]
        OP2 shell kuvvetleri.
    output_csv : str
        Cikti CSV dosya yolu.

    Returns
    -------
    pd.DataFrame
        Tahmin sonuclari.
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

        # Katsayilari hedef bazinda topla
        eq_map = {eq.target_name: eq for eq in res.equations}

        # Her eslesen subcase icin tahmin yap
        subcases = res.matched_subcases if res.matched_subcases else []
        for sc_id in subcases:
            # Bar axial force
            bar_result = bar_forces.get((sc_id, bar_eid))
            if bar_result is None:
                continue
            af = float(np.mean(bar_result.axial_force))

            # Shell membrane forces
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
                "AF": af,
                "NX": nx,
                "NY": ny,
                "NXY": nxy,
            }

            # Her hedef icin tahmin
            predictors = np.array([af, nx, ny, nxy])
            for target_name, eq in eq_map.items():
                pred_val = float(np.dot(eq.coefficients, predictors) + eq.intercept)
                col_name = _target_to_col(target_name)
                row[col_name] = pred_val

            rows.append(row)

    if not rows:
        logger.warning("Tahmin icin veri bulunamadi")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Kolon siralamasini duzenle
    ordered_cols = [
        "Bar_EID", "Element_Type", "Subcase_ID", "Shell_EID",
        "AF", "NX", "NY", "NXY",
        "Pred_F_Bearing_X", "Pred_F_Bearing_Y",
        "Pred_NX_Bypass", "Pred_NY_Bypass", "Pred_NXY_Bypass",
    ]
    present = [c for c in ordered_cols if c in df.columns]
    extra = [c for c in df.columns if c not in ordered_cols]
    df = df[present + extra]

    df.to_csv(output_csv, index=False)
    logger.info("Tahmin CSV yazildi: %s (%d satir)", output_csv, len(df))
    return df


def predict_from_dataframes(
    coefficients_df: pd.DataFrame,
    bar_forces_df: pd.DataFrame,
    shell_forces_df: pd.DataFrame,
    output_csv: str,
) -> pd.DataFrame:
    """
    DataFrame formatindaki verilerle tahmin yap.

    Bu fonksiyon CSV/Excel'den okunan verileri destekler.
    Ayrica H5'ten okunan bar/shell combined verileriyle de calisir.

    Parameters
    ----------
    coefficients_df : pd.DataFrame
        Total Summary formati:
        Bar_EID | Element_Type | Subcase_ID | Shell_EID | Shell_Type |
        Target | Coeff_Bar_Axial | Coeff_Shell_Nx | Coeff_Shell_Ny |
        Coeff_Shell_Nxy | Intercept | R2
    bar_forces_df : pd.DataFrame
        Bar element combined:
        Bar_EID | Subcase_ID | AF (Axial Force)
    shell_forces_df : pd.DataFrame
        Shell force combined:
        Element_ID | Subcase_ID | NX | NY | NXY
    output_csv : str
        Cikti CSV dosya yolu.

    Returns
    -------
    pd.DataFrame
        Tahmin sonuclari.
    """
    # Kolon isimlerini normalize et
    bar_df = _normalize_bar_df(bar_forces_df)
    shell_df = _normalize_shell_df(shell_forces_df)
    coeff_df = coefficients_df.copy()

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

            # Birden fazla shell satiri varsa ortalama al
            nx = float(shell_sc["nx"].mean())
            ny = float(shell_sc["ny"].mean())
            nxy = float(shell_sc["nxy"].mean())

            row = {
                "Bar_EID": int(bar_eid),
                "Element_Type": int(et),
                "Subcase_ID": int(sc_id),
                "Shell_EID": int(shell_eid),
                "AF": af,
                "NX": nx,
                "NY": ny,
                "NXY": nxy,
            }

            predictors = np.array([af, nx, ny, nxy])
            for target_name, eq_info in eq_map.items():
                pred_val = float(np.dot(eq_info["coeffs"], predictors) + eq_info["intercept"])
                col_name = _target_to_col(target_name)
                row[col_name] = pred_val

            rows.append(row)

    if not rows:
        logger.warning("Tahmin icin eslesen veri bulunamadi")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    ordered_cols = [
        "Bar_EID", "Element_Type", "Subcase_ID", "Shell_EID",
        "AF", "NX", "NY", "NXY",
        "Pred_F_Bearing_X", "Pred_F_Bearing_Y",
        "Pred_NX_Bypass", "Pred_NY_Bypass", "Pred_NXY_Bypass",
    ]
    present = [c for c in ordered_cols if c in df.columns]
    extra = [c for c in df.columns if c not in ordered_cols]
    df = df[present + extra]

    df.to_csv(output_csv, index=False)
    logger.info("Tahmin CSV yazildi: %s (%d satir)", output_csv, len(df))
    return df


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


def _normalize_bar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Bar forces DataFrame kolon isimlerini normalize et."""
    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_").replace("-", "_")
        if lower in ("bar_eid", "bar_element_id", "bareid", "bar_id"):
            col_map[col] = "bar_eid"
        elif lower in ("subcase_id", "subcaseid", "subcase", "sc_id"):
            col_map[col] = "subcase_id"
        elif lower in ("af", "axial_force", "axialforce", "bar_axial", "axial"):
            col_map[col] = "af"

    result = df.rename(columns=col_map)
    for needed in ["bar_eid", "subcase_id", "af"]:
        if needed not in result.columns:
            logger.warning("Bar forces: '%s' kolonu bulunamadi", needed)
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
            logger.warning("Shell forces: '%s' kolonu bulunamadi", needed)
    return result
