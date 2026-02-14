"""
Correlation Engine Module

Bar element axial force ve bagli shell element fluxlari (OP2) ile
H5 JOINT_LOADS_CAP degerleri (F Bearing X, F Bearing Y, NX/NY/NXY Bypass)
arasinda coklu regresyon denklemi olusturur.

Her bar element icin 4 predictor (bagımsız degisken):
  - Bar Axial Force (OP2)
  - Shell Nx (bagli tum shell'lerin ortalamasi, OP2)
  - Shell Ny (bagli tum shell'lerin ortalamasi, OP2)
  - Shell Nxy (bagli tum shell'lerin ortalamasi, OP2)

5 hedef (bagimli degisken, H5'ten):
  - F Bearing X
  - F Bearing Y
  - NX Bypass
  - NY Bypass
  - NXY Bypass

Denklem:
  Target = a1*Bar_Axial + a2*Shell_Nx + a3*Shell_Ny + a4*Shell_Nxy + b
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


PREDICTOR_NAMES = ["Bar_Axial", "Shell_Nx", "Shell_Ny", "Shell_Nxy"]
H5_TARGET_COLUMNS = ["F Bearing X", "F Bearing Y", "NX Bypass", "NY Bypass", "NXY Bypass"]


@dataclass
class RegressionEquation:
    """Tek bir hedef icin coklu regresyon denklemi."""
    target_name: str
    predictor_names: List[str]
    coefficients: np.ndarray   # [a1, a2, a3, a4]
    intercept: float
    r_squared: float
    n_samples: int

    def equation_str(self) -> str:
        """Denklemi okunabilir string olarak dondur."""
        parts = []
        for name, coeff in zip(self.predictor_names, self.coefficients):
            parts.append(f"{coeff:+.6f}*{name}")
        return f"{self.target_name} = {' '.join(parts)} {self.intercept:+.6f}"


@dataclass
class JointCorrelationResult:
    """Bir bar element + element type icin tum korelasyon sonuclari."""
    bar_eid: int
    subcase_id: int
    element_type: int  # H5'teki Element Type (0, 1, ...)
    n_connected_shells: int
    equations: List[RegressionEquation] = field(default_factory=list)
    predictor_data: Dict[str, np.ndarray] = field(default_factory=dict)
    target_data: Dict[str, np.ndarray] = field(default_factory=dict)
    matched_subcases: List[int] = field(default_factory=list)


class CorrelationEngine:
    """
    OP2 kuvvetleri ile H5 Joint Load Cap arasinda coklu regresyon hesaplar.

    Her bar element icin:
      Target = a1*Bar_Axial + a2*Avg_Nx + a3*Avg_Ny + a4*Avg_Nxy + b
    """

    def __init__(self):
        self.results: List[JointCorrelationResult] = []

    def compute_joint_correlation(
        self,
        bar_eid: int,
        subcase_id: int,
        element_type: int,
        bar_axial: np.ndarray,
        avg_shell_nx: np.ndarray,
        avg_shell_ny: np.ndarray,
        avg_shell_nxy: np.ndarray,
        h5_data: pd.DataFrame,
        n_shells: int,
    ) -> JointCorrelationResult:
        """
        Bar + ortalama shell kuvvetleri ile H5 hedefleri arasinda
        coklu regresyon denklemi olusturur.

        Parameters
        ----------
        bar_eid : int
            Bar element ID.
        subcase_id : int
            Subcase ID.
        element_type : int
            H5 Element Type (0, 1, ...).
        bar_axial : np.ndarray
            Bar axial force (ntimes,).
        avg_shell_nx, avg_shell_ny, avg_shell_nxy : np.ndarray
            Bagli tum shell'lerin ortalama Nx, Ny, Nxy (ntimes,).
        h5_data : pd.DataFrame
            H5 Joint Load Cap verileri (bu element type icin filtrelenmis).
        n_shells : int
            Bagli shell sayisi.
        """
        result = JointCorrelationResult(
            bar_eid=bar_eid,
            subcase_id=subcase_id,
            element_type=element_type,
            n_connected_shells=n_shells,
        )

        # Predictor verileri
        predictors = {
            "Bar_Axial": np.asarray(bar_axial, dtype=float),
            "Shell_Nx": np.asarray(avg_shell_nx, dtype=float),
            "Shell_Ny": np.asarray(avg_shell_ny, dtype=float),
            "Shell_Nxy": np.asarray(avg_shell_nxy, dtype=float),
        }
        result.predictor_data = predictors

        # H5 hedef verileri
        targets = {}
        for col in H5_TARGET_COLUMNS:
            matching = [c for c in h5_data.columns
                        if self._normalize(c) == self._normalize(col)]
            if matching:
                targets[col] = h5_data[matching[0]].values.astype(float)
        result.target_data = targets

        if not targets:
            logger.warning("Bar %d: H5 hedef kolonlari bulunamadi", bar_eid)
            self.results.append(result)
            return result

        # Veri uzunluklarini esitle
        all_arrays = list(predictors.values()) + list(targets.values())
        min_len = min(len(a) for a in all_arrays)
        if min_len < 5:
            logger.warning(
                "Bar %d: Yetersiz veri noktasi (%d), regresyon atlanıyor",
                bar_eid, min_len,
            )
            self.results.append(result)
            return result

        # Predictor matrisi: (n, 4)
        X_raw = np.column_stack([v[:min_len] for v in predictors.values()])

        # Her hedef icin coklu regresyon
        for target_name, target_arr in targets.items():
            y = target_arr[:min_len]

            # NaN satirlarini kaldir
            valid = np.all(np.isfinite(X_raw), axis=1) & np.isfinite(y)
            n_valid = int(valid.sum())
            if n_valid < 5:
                logger.debug(
                    "Bar %d, %s: Yetersiz gecerli veri (%d)",
                    bar_eid, target_name, n_valid,
                )
                continue

            X = X_raw[valid]
            y_valid = y[valid]

            # Intercept kolonu ekle: (n, 5)
            X_aug = np.column_stack([X, np.ones(X.shape[0])])

            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_valid, rcond=None)
            except np.linalg.LinAlgError as e:
                logger.warning("Bar %d, %s: lstsq hatasi: %s", bar_eid, target_name, e)
                continue

            # R² hesapla
            y_pred = X_aug @ coeffs
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

            eq = RegressionEquation(
                target_name=target_name,
                predictor_names=list(predictors.keys()),
                coefficients=coeffs[:-1],
                intercept=coeffs[-1],
                r_squared=max(0.0, r_squared),
                n_samples=n_valid,
            )
            result.equations.append(eq)

            logger.info(
                "Bar %d | %s | R²=%.4f | n=%d",
                bar_eid, eq.equation_str(), r_squared, n_valid,
            )

        self.results.append(result)
        return result

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Tum korelasyon denklemlerini ozet DataFrame olarak dondur."""
        rows = []
        for res in self.results:
            for eq in res.equations:
                row = {
                    "Bar_EID": res.bar_eid,
                    "Subcase_ID": res.subcase_id,
                    "Element_Type": res.element_type,
                    "N_Shells": res.n_connected_shells,
                    "Target": eq.target_name,
                    "R_Squared": eq.r_squared,
                    "N_Samples": eq.n_samples,
                    "Intercept": eq.intercept,
                }
                for pname, coeff in zip(eq.predictor_names, eq.coefficients):
                    row[f"Coeff_{pname}"] = coeff
                row["Equation"] = eq.equation_str()
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _normalize(s: str) -> str:
        """Kolon isimlerini normalize et."""
        return s.lower().replace(" ", "").replace("_", "").strip()
