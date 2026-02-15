"""
Predictor Module

Korelasyon analizinden elde edilen katsayilari kullanarak
yeni veriler (ayri bir Prediction H5 dosyasindan) icin
Bearing ve Bypass yuklerini tahmin eder.

Prediction H5 yapisi:
  - Bar element combined tablosu: Bar_EID, Subcase_ID, AF (Axial Force)
  - Shell force combined tablosu: Element_ID, Subcase_ID, NX(=MX), NY(=MY), NXY(=MXY)

Cikti:
  CSV dosyasi: Bar_EID, Element_Type, Subcase_ID, Shell_EID,
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


# ============================================================
#  Prediction H5 Reader
# ============================================================


def read_prediction_h5(h5_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prediction H5 dosyasini oku.

    H5 icinde iki tablo aranir:
      1. Bar element combined (AF / Axial Force iceren)
      2. Shell force combined (NX/MX, NY/MY, NXY/MXY iceren)

    Parameters
    ----------
    h5_path : str
        Prediction H5 dosya yolu.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (bar_forces_df, shell_forces_df)
    """
    logger.info("Prediction H5 okunuyor: %s", h5_path)

    bar_df = pd.DataFrame()
    shell_df = pd.DataFrame()

    with h5py.File(h5_path, "r") as f:
        # Tum dataset'leri tara
        datasets = {}
        _collect_datasets(f, datasets)

        logger.info("H5 icerisinde %d dataset bulundu", len(datasets))
        for path, cols in datasets.items():
            logger.info("  %s: %s", path, cols[:10])

        # Bar element combined tablosunu bul
        bar_path = _find_bar_table(datasets)
        if bar_path:
            bar_df = _read_h5_table(f, bar_path)
            logger.info("Bar element tablosu bulundu: %s (%d satir)", bar_path, len(bar_df))
        else:
            logger.warning("Prediction H5'te bar element tablosu bulunamadi")

        # Shell force combined tablosunu bul
        shell_path = _find_shell_table(datasets)
        if shell_path:
            shell_df = _read_h5_table(f, shell_path)
            logger.info("Shell force tablosu bulundu: %s (%d satir)", shell_path, len(shell_df))
        else:
            logger.warning("Prediction H5'te shell force tablosu bulunamadi")

    return bar_df, shell_df


def _collect_datasets(group, result: Dict, prefix: str = ""):
    """H5 grup icerisindeki tum dataset'leri recursive topla."""
    for key in group:
        path = f"{prefix}/{key}" if prefix else key
        item = group[key]
        if isinstance(item, h5py.Group):
            _collect_datasets(item, result, path)
        elif isinstance(item, h5py.Dataset):
            if item.dtype.names:
                result[path] = list(item.dtype.names)
            else:
                result[path] = []


def _find_bar_table(datasets: Dict[str, List[str]]) -> Optional[str]:
    """Bar element combined tablosunu bul (AF kolonu iceren)."""
    for path, cols in datasets.items():
        cols_lower = [c.lower().replace(" ", "").replace("_", "") for c in cols]
        path_lower = path.lower()

        # AF veya AxialForce kolonu olan tabloyu bul
        has_af = any(c in ("af", "axialforce", "baraxial", "axial") for c in cols_lower)
        # "bar" kelimesi path veya kolon isimlerinde
        has_bar_hint = "bar" in path_lower or any("bar" in c for c in cols_lower)

        if has_af:
            return path
        if has_bar_hint and len(cols) > 0:
            # AF kolonu olmasa bile bar tablosu olabilir
            continue

    # Fallback: path'te "bar" gecen ilk dataset
    for path, cols in datasets.items():
        if "bar" in path.lower() and cols:
            return path

    return None


def _find_shell_table(datasets: Dict[str, List[str]]) -> Optional[str]:
    """Shell force combined tablosunu bul (NX/MX kolonu iceren)."""
    for path, cols in datasets.items():
        cols_lower = [c.lower().replace(" ", "").replace("_", "") for c in cols]
        path_lower = path.lower()

        # NX/MX kolonu olan tabloyu bul
        has_nx = any(c in ("nx", "mx", "membranex", "shellnx") for c in cols_lower)
        has_shell_hint = "shell" in path_lower or any("shell" in c for c in cols_lower)

        if has_nx:
            return path

    # Fallback: path'te "shell" gecen ilk dataset
    for path, cols in datasets.items():
        if "shell" in path.lower() and cols:
            return path

    return None


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


# ============================================================
#  Prediction Functions
# ============================================================


def predict_from_h5(
    per_shell_results: List[JointCorrelationResult],
    prediction_h5_path: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Ayri bir Prediction H5 dosyasindan veri okuyup tahmin yap.

    Parameters
    ----------
    per_shell_results : List[JointCorrelationResult]
        Per-shell korelasyon sonuclari (katsayilar icin).
    prediction_h5_path : str
        Prediction H5 dosya yolu (bar + shell force combined).
    output_csv : str
        Cikti CSV dosya yolu.

    Returns
    -------
    pd.DataFrame
        Tahmin sonuclari.
    """
    bar_df, shell_df = read_prediction_h5(prediction_h5_path)

    if bar_df.empty:
        logger.error("Prediction H5'te bar element verisi bulunamadi")
        return pd.DataFrame()
    if shell_df.empty:
        logger.error("Prediction H5'te shell force verisi bulunamadi")
        return pd.DataFrame()

    # Total Summary formati olustur (katsayilar)
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

    return predict_from_dataframes(
        coefficients_df=coeff_df,
        bar_forces_df=bar_df,
        shell_forces_df=shell_df,
        output_csv=output_csv,
    )


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
                "AF": af,
                "NX": nx,
                "NY": ny,
                "NXY": nxy,
            }

            predictors = np.array([af, nx, ny, nxy])
            for target_name, eq in eq_map.items():
                pred_val = float(np.dot(eq.coefficients, predictors) + eq.intercept)
                row[_target_to_col(target_name)] = pred_val

            rows.append(row)

    if not rows:
        logger.warning("Tahmin icin veri bulunamadi")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _order_columns(df)
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

    Parameters
    ----------
    coefficients_df : pd.DataFrame
        Katsayi tablosu:
        Bar_EID | Element_Type | Shell_EID |
        Target | Coeff_Bar_Axial | Coeff_Shell_Nx | Coeff_Shell_Ny |
        Coeff_Shell_Nxy | Intercept
    bar_forces_df : pd.DataFrame
        Bar element combined: Bar_EID, Subcase_ID, AF
    shell_forces_df : pd.DataFrame
        Shell force combined: Element_ID, Subcase_ID, NX, NY, NXY
    output_csv : str
        Cikti CSV dosya yolu.
    """
    bar_df = _normalize_bar_df(bar_forces_df)
    shell_df = _normalize_shell_df(shell_forces_df)
    coeff_df = coefficients_df.copy()

    logger.info("Prediction: bar_df %d satir, shell_df %d satir, coeff %d satir",
                len(bar_df), len(shell_df), len(coeff_df))
    logger.info("Bar kolonlar: %s", list(bar_df.columns))
    logger.info("Shell kolonlar: %s", list(shell_df.columns))

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
                "AF": af,
                "NX": nx,
                "NY": ny,
                "NXY": nxy,
            }

            predictors = np.array([af, nx, ny, nxy])
            for target_name, eq_info in eq_map.items():
                pred_val = float(np.dot(eq_info["coeffs"], predictors) + eq_info["intercept"])
                row[_target_to_col(target_name)] = pred_val

            rows.append(row)

    if not rows:
        logger.warning("Tahmin icin eslesen veri bulunamadi")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _order_columns(df)
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
        if lower in ("bar_eid", "bar_element_id", "bareid", "bar_id"):
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
