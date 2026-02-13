"""
Correlation Engine Module

Bar element axial force ve bağlı shell element fluxları (OP2) ile
H5 Joint Load Cap değerleri (FBearingX, FBearingY, NX/NY/NXY Bypass)
arasındaki korelasyonu hesaplar.

Fiziksel İlişki (Bolted/Riveted Joint):
=========================================
Bir bağlantı noktasında (fastener = bar element):
- Bearing Load: Bağlantı elemanı üzerinden panele aktarılan yük
  FBearingX, FBearingY = Bar axial force'un X,Y bileşenleri
- Bypass Load: Bağlantı elemanını atlayarak panelden geçen yük
  N_Bypass = N_total - N_bearing
  NX Bypass = Panel NX (total) - FBearingX katkısı
  NY Bypass = Panel NY (total) - FBearingY katkısı
  NXY Bypass = Panel NXY (total) - Bearing shear katkısı

Her bar element ve element tipi (CQUAD4, CTRIA3) için ayrı ayrı
korelasyon matrisi ve regresyon katsayıları hesaplanır.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Bir bar element için korelasyon sonuçları."""
    bar_eid: int
    subcase_id: int
    element_type: str  # "BAR", "CQUAD4", "CTRIA3" vb.

    # OP2'den gelen değerler
    op2_values: Dict[str, np.ndarray] = field(default_factory=dict)

    # H5'ten gelen Joint Load Cap değerleri
    h5_values: Dict[str, np.ndarray] = field(default_factory=dict)

    # Korelasyon matrisi (Pearson r)
    correlation_matrix: Optional[pd.DataFrame] = None

    # Regresyon katsayıları: h5_col -> {op2_col: (slope, intercept, r_value, p_value, std_err)}
    regression_results: Dict[str, Dict[str, Tuple]] = field(default_factory=dict)


class CorrelationEngine:
    """
    OP2 kuvvetleri ile H5 Joint Load Cap arasında korelasyon hesaplar.

    Her bar element için:
    1. Bar axial force vs FBearingX, FBearingY
    2. Shell NX vs NX Bypass
    3. Shell NY vs NY Bypass
    4. Shell NXY vs NXY Bypass

    Ayrıca çoklu regresyon analizi:
    - FBearingX = a1*Axial + a2*Shear1 + a3*Shear2 + b
    - NX Bypass = a1*NX_shell + a2*Axial + b
    """

    H5_TARGET_COLUMNS = ["FBearingX", "FBearingY", "NX Bypass", "NY Bypass", "NXY Bypass"]

    def __init__(self):
        self.results: List[CorrelationResult] = []

    def compute_bar_correlation(
        self,
        bar_eid: int,
        subcase_id: int,
        bar_axial: np.ndarray,
        bar_shear1: np.ndarray,
        bar_shear2: np.ndarray,
        bar_torque: np.ndarray,
        h5_data: pd.DataFrame,
    ) -> CorrelationResult:
        """
        Bar element kuvvetleri ile H5 Joint Load Cap arasında korelasyon.

        Parameters
        ----------
        bar_eid : int
            Bar element ID.
        subcase_id : int
            Subcase ID.
        bar_axial : np.ndarray
            Bar axial force dizisi.
        bar_shear1, bar_shear2 : np.ndarray
            Bar shear force dizileri.
        bar_torque : np.ndarray
            Bar torque dizisi.
        h5_data : pd.DataFrame
            H5 Joint Load Cap verileri (bu bar element için).

        Returns
        -------
        CorrelationResult
        """
        result = CorrelationResult(
            bar_eid=bar_eid,
            subcase_id=subcase_id,
            element_type="BAR",
        )

        # OP2 değerleri
        result.op2_values = {
            "Axial_Force": bar_axial,
            "Shear_1": bar_shear1,
            "Shear_2": bar_shear2,
            "Torque": bar_torque,
        }

        # H5 değerleri
        for col in self.H5_TARGET_COLUMNS:
            matching_cols = [c for c in h5_data.columns if self._normalize(c) == self._normalize(col)]
            if matching_cols:
                result.h5_values[col] = h5_data[matching_cols[0]].values

        # Korelasyon hesapla
        result.correlation_matrix = self._compute_correlation_matrix(
            result.op2_values, result.h5_values
        )

        # Regresyon hesapla
        result.regression_results = self._compute_regressions(
            result.op2_values, result.h5_values
        )

        self.results.append(result)
        return result

    def compute_shell_correlation(
        self,
        bar_eid: int,
        subcase_id: int,
        shell_eid: int,
        elem_type: str,
        nx: np.ndarray,
        ny: np.ndarray,
        nxy: np.ndarray,
        h5_data: pd.DataFrame,
    ) -> CorrelationResult:
        """
        Shell element fluxları ile H5 Joint Load Cap arasında korelasyon.

        Parameters
        ----------
        bar_eid : int
            İlişkili bar element ID.
        subcase_id : int
            Subcase ID.
        shell_eid : int
            Shell element ID.
        elem_type : str
            Element tipi (CQUAD4, CTRIA3, vb.).
        nx, ny, nxy : np.ndarray
            Shell membrane fluxları.
        h5_data : pd.DataFrame
            H5 Joint Load Cap verileri.

        Returns
        -------
        CorrelationResult
        """
        result = CorrelationResult(
            bar_eid=bar_eid,
            subcase_id=subcase_id,
            element_type=f"{elem_type}_{shell_eid}",
        )

        result.op2_values = {
            "NX_Flux": nx,
            "NY_Flux": ny,
            "NXY_Flux": nxy,
        }

        for col in self.H5_TARGET_COLUMNS:
            matching_cols = [c for c in h5_data.columns if self._normalize(c) == self._normalize(col)]
            if matching_cols:
                result.h5_values[col] = h5_data[matching_cols[0]].values

        result.correlation_matrix = self._compute_correlation_matrix(
            result.op2_values, result.h5_values
        )
        result.regression_results = self._compute_regressions(
            result.op2_values, result.h5_values
        )

        self.results.append(result)
        return result

    def compute_combined_correlation(
        self,
        bar_eid: int,
        subcase_id: int,
        bar_axial: np.ndarray,
        bar_shear1: np.ndarray,
        bar_shear2: np.ndarray,
        shell_forces: Dict[int, Dict[str, np.ndarray]],
        h5_data: pd.DataFrame,
        shell_types: Dict[int, str],
    ) -> CorrelationResult:
        """
        Bar + Shell kuvvetlerini birleştirip H5 ile korelasyon.

        Ortalama shell fluxlarını bar kuvvetleri ile birlikte kullanır.

        Parameters
        ----------
        bar_eid : int
            Bar element ID.
        subcase_id : int
            Subcase ID.
        bar_axial : np.ndarray
            Bar axial force.
        bar_shear1, bar_shear2 : np.ndarray
            Bar shear forces.
        shell_forces : Dict[int, Dict[str, np.ndarray]]
            Shell element ID -> {NX, NY, NXY} fluxları.
        h5_data : pd.DataFrame
            H5 Joint Load Cap verileri.
        shell_types : Dict[int, str]
            Shell element ID -> element tipi.
        """
        result = CorrelationResult(
            bar_eid=bar_eid,
            subcase_id=subcase_id,
            element_type="COMBINED",
        )

        # Bar kuvvetleri
        result.op2_values["Bar_Axial"] = bar_axial
        result.op2_values["Bar_Shear1"] = bar_shear1
        result.op2_values["Bar_Shear2"] = bar_shear2

        # Shell fluxlarının ortalaması (QUAD ve TRIA ayrı)
        quad_nx, quad_ny, quad_nxy = [], [], []
        tria_nx, tria_ny, tria_nxy = [], [], []

        for seid, forces in shell_forces.items():
            stype = shell_types.get(seid, "UNKNOWN")
            if "QUAD" in stype:
                quad_nx.append(forces.get("NX", np.zeros_like(bar_axial)))
                quad_ny.append(forces.get("NY", np.zeros_like(bar_axial)))
                quad_nxy.append(forces.get("NXY", np.zeros_like(bar_axial)))
            elif "TRI" in stype:
                tria_nx.append(forces.get("NX", np.zeros_like(bar_axial)))
                tria_ny.append(forces.get("NY", np.zeros_like(bar_axial)))
                tria_nxy.append(forces.get("NXY", np.zeros_like(bar_axial)))

        if quad_nx:
            result.op2_values["Avg_QUAD_NX"] = np.mean(quad_nx, axis=0)
            result.op2_values["Avg_QUAD_NY"] = np.mean(quad_ny, axis=0)
            result.op2_values["Avg_QUAD_NXY"] = np.mean(quad_nxy, axis=0)

        if tria_nx:
            result.op2_values["Avg_TRIA_NX"] = np.mean(tria_nx, axis=0)
            result.op2_values["Avg_TRIA_NY"] = np.mean(tria_ny, axis=0)
            result.op2_values["Avg_TRIA_NXY"] = np.mean(tria_nxy, axis=0)

        for col in self.H5_TARGET_COLUMNS:
            matching_cols = [c for c in h5_data.columns if self._normalize(c) == self._normalize(col)]
            if matching_cols:
                result.h5_values[col] = h5_data[matching_cols[0]].values

        result.correlation_matrix = self._compute_correlation_matrix(
            result.op2_values, result.h5_values
        )
        result.regression_results = self._compute_regressions(
            result.op2_values, result.h5_values
        )

        self.results.append(result)
        return result

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Tüm korelasyon sonuçlarını özet DataFrame olarak döndür."""
        rows = []
        for res in self.results:
            if res.correlation_matrix is None:
                continue

            for h5_col in res.correlation_matrix.columns:
                for op2_col in res.correlation_matrix.index:
                    r_val = res.correlation_matrix.loc[op2_col, h5_col]
                    reg = res.regression_results.get(h5_col, {}).get(op2_col)

                    row = {
                        "Bar_EID": res.bar_eid,
                        "Subcase_ID": res.subcase_id,
                        "Element_Type": res.element_type,
                        "OP2_Parameter": op2_col,
                        "H5_Parameter": h5_col,
                        "Pearson_R": r_val,
                        "R_Squared": r_val ** 2 if not np.isnan(r_val) else np.nan,
                    }

                    if reg is not None:
                        row["Slope"] = reg[0]
                        row["Intercept"] = reg[1]
                        row["P_Value"] = reg[3]
                        row["Std_Error"] = reg[4]

                    rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _compute_correlation_matrix(
        op2_values: Dict[str, np.ndarray],
        h5_values: Dict[str, np.ndarray],
    ) -> Optional[pd.DataFrame]:
        """Pearson korelasyon matrisini hesapla."""
        if not op2_values or not h5_values:
            return None

        # Veri uzunluklarını kontrol et
        lengths = set()
        for v in list(op2_values.values()) + list(h5_values.values()):
            lengths.add(len(v))

        if len(lengths) > 1:
            # Uzunluklar farklıysa en kısa uzunluğa kes
            min_len = min(lengths)
            logger.warning(
                "Farkli veri uzunluklari: %s. %d'ye kesiliyor.",
                lengths, min_len,
            )
            op2_trimmed = {k: v[:min_len] for k, v in op2_values.items()}
            h5_trimmed = {k: v[:min_len] for k, v in h5_values.items()}
        else:
            op2_trimmed = op2_values
            h5_trimmed = h5_values

        op2_keys = list(op2_trimmed.keys())
        h5_keys = list(h5_trimmed.keys())

        matrix = np.full((len(op2_keys), len(h5_keys)), np.nan)

        for i, op2_k in enumerate(op2_keys):
            for j, h5_k in enumerate(h5_keys):
                x = op2_trimmed[op2_k]
                y = h5_trimmed[h5_k]

                # NaN ve sıfır-varyans kontrolü
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 3:
                    continue

                x_valid = x[valid]
                y_valid = y[valid]

                if np.std(x_valid) < 1e-15 or np.std(y_valid) < 1e-15:
                    matrix[i, j] = 0.0
                    continue

                r, _ = stats.pearsonr(x_valid, y_valid)
                matrix[i, j] = r

        return pd.DataFrame(matrix, index=op2_keys, columns=h5_keys)

    @staticmethod
    def _compute_regressions(
        op2_values: Dict[str, np.ndarray],
        h5_values: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Tuple]]:
        """Her H5 kolonu için her OP2 kolonuyla lineer regresyon."""
        results = {}

        # Uzunluk eşitleme
        lengths = set()
        for v in list(op2_values.values()) + list(h5_values.values()):
            lengths.add(len(v))

        min_len = min(lengths) if lengths else 0
        op2_trimmed = {k: v[:min_len] for k, v in op2_values.items()}
        h5_trimmed = {k: v[:min_len] for k, v in h5_values.items()}

        for h5_col, h5_arr in h5_trimmed.items():
            results[h5_col] = {}
            for op2_col, op2_arr in op2_trimmed.items():
                valid = np.isfinite(op2_arr) & np.isfinite(h5_arr)
                if valid.sum() < 3:
                    continue

                x = op2_arr[valid]
                y = h5_arr[valid]

                if np.std(x) < 1e-15:
                    continue

                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    results[h5_col][op2_col] = (slope, intercept, r_value, p_value, std_err)
                except Exception as e:
                    logger.warning(
                        "Regresyon hatasi (%s vs %s): %s", op2_col, h5_col, e
                    )

        return results

    @staticmethod
    def _normalize(s: str) -> str:
        """Kolon isimlerini normalize et (karşılaştırma için)."""
        return s.lower().replace(" ", "").replace("_", "").strip()
