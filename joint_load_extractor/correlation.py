"""
Correlation Engine Module

Bar element axial force ve bagli her shell elementin AYRI kuvvetleri (OP2) ile
H5 JOINT_LOADS_CAP degerleri (F Bearing X, F Bearing Y, NX/NY/NXY Bypass)
arasinda coklu regresyon denklemi olusturur (intercept yok, orijinden gecer).

Her bar element icin predictor'lar (bagımsız degiskenler):
  - Bar Axial Force (OP2)
  - Shell_{EID1}_Nx, Shell_{EID1}_Ny, Shell_{EID1}_Nxy  (her bagli shell icin ayri)
  - Shell_{EID2}_Nx, Shell_{EID2}_Ny, Shell_{EID2}_Nxy
  - ...

5 hedef (bagimli degisken, H5'ten):
  - F Bearing X
  - F Bearing Y
  - NX Bypass
  - NY Bypass
  - NXY Bypass

Denklem (intercept yok):
  Target = a0*Bar_Axial
         + a1*Shell_{EID1}_Nx + a2*Shell_{EID1}_Ny + a3*Shell_{EID1}_Nxy
         + a4*Shell_{EID2}_Nx + a5*Shell_{EID2}_Ny + a6*Shell_{EID2}_Nxy
         + ...
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


H5_TARGET_COLUMNS = ["F Bearing X", "F Bearing Y", "NX Bypass", "NY Bypass", "NXY Bypass"]


@dataclass
class RegressionEquation:
    """Tek bir hedef icin coklu regresyon denklemi (intercept yok)."""
    target_name: str
    predictor_names: List[str]
    coefficients: np.ndarray
    r_squared: float
    n_samples: int

    def equation_str(self) -> str:
        """Denklemi okunabilir string olarak dondur."""
        parts = []
        for name, coeff in zip(self.predictor_names, self.coefficients):
            parts.append(f"{coeff:+.6f}*{name}")
        return f"{self.target_name} = {' '.join(parts)}"


@dataclass
class JointCorrelationResult:
    """Bir bar element + element type icin tum korelasyon sonuclari."""
    bar_eid: int
    subcase_id: int
    element_type: int  # H5'teki Element Type (0, 1, ...)
    n_connected_shells: int
    shell_eids_used: List[int] = field(default_factory=list)
    shell_eid: Optional[int] = None
    shell_type: Optional[str] = None
    equations: List[RegressionEquation] = field(default_factory=list)
    predictor_data: Dict[str, np.ndarray] = field(default_factory=dict)
    target_data: Dict[str, np.ndarray] = field(default_factory=dict)
    matched_subcases: List[int] = field(default_factory=list)


class CorrelationEngine:
    """
    OP2 kuvvetleri ile H5 Joint Load Cap arasinda coklu regresyon hesaplar.

    Her bar element icin:
      Target = a0*Bar_Axial
             + a1*Shell_{EID1}_Nx + a2*Shell_{EID1}_Ny + a3*Shell_{EID1}_Nxy
             + ...
    """

    def __init__(self):
        self.results: List[JointCorrelationResult] = []

    def compute_joint_correlation(
        self,
        bar_eid: int,
        subcase_id: int,
        element_type: int,
        bar_axial: np.ndarray,
        shell_forces_per_eid: Dict[int, Dict[str, np.ndarray]],
        h5_data: pd.DataFrame,
        n_shells: int,
    ) -> JointCorrelationResult:
        """
        Bar + per-shell kuvvetleri ile H5 hedefleri arasinda
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
            Bar axial force (n_subcases,).
        shell_forces_per_eid : dict
            {shell_eid: {"nx": array, "ny": array, "nxy": array}}
            Her shell icin ayri kuvvet arrayleri (n_subcases,).
        h5_data : pd.DataFrame
            H5 Joint Load Cap verileri (bu element type icin filtrelenmis).
        n_shells : int
            Bagli shell sayisi.
        """
        sorted_shell_eids = sorted(shell_forces_per_eid.keys())

        result = JointCorrelationResult(
            bar_eid=bar_eid,
            subcase_id=subcase_id,
            element_type=element_type,
            n_connected_shells=n_shells,
            shell_eids_used=sorted_shell_eids,
        )

        # Predictor verileri: Bar_Axial + per-shell Nx/Ny/Nxy
        predictors = {"Bar_Axial": np.asarray(bar_axial, dtype=float)}
        for seid in sorted_shell_eids:
            sf = shell_forces_per_eid[seid]
            predictors[f"Shell_{seid}_Nx"] = np.asarray(sf["nx"], dtype=float)
            predictors[f"Shell_{seid}_Ny"] = np.asarray(sf["ny"], dtype=float)
            predictors[f"Shell_{seid}_Nxy"] = np.asarray(sf["nxy"], dtype=float)
        result.predictor_data = predictors

        n_predictors = len(predictors)
        logger.info(
            "Bar %d, ET %d: %d predictor (1 bar + %d shell x 3)",
            bar_eid, element_type, n_predictors,
            len(sorted_shell_eids),
        )

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

        # Minimum veri noktasi: en az predictor sayisi + 1
        min_required = max(5, n_predictors + 1)
        if min_len < min_required:
            logger.warning(
                "Bar %d: Yetersiz veri noktasi (%d < %d gerekli), "
                "regresyon atlanıyor (predictor=%d)",
                bar_eid, min_len, min_required, n_predictors,
            )
            self.results.append(result)
            return result

        # Predictor matrisi: (n, 1 + 3*n_shells)
        pred_names = list(predictors.keys())
        X_raw = np.column_stack([predictors[k][:min_len] for k in pred_names])

        # Her hedef icin coklu regresyon
        for target_name, target_arr in targets.items():
            y = target_arr[:min_len]

            # NaN satirlarini kaldir
            valid = np.all(np.isfinite(X_raw), axis=1) & np.isfinite(y)
            n_valid = int(valid.sum())
            if n_valid < min_required:
                logger.debug(
                    "Bar %d, %s: Yetersiz gecerli veri (%d < %d)",
                    bar_eid, target_name, n_valid, min_required,
                )
                continue

            X = X_raw[valid]
            y_valid = y[valid]

            # Intercept yok - orijinden gecen regresyon
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y_valid, rcond=None)
            except np.linalg.LinAlgError as e:
                logger.warning("Bar %d, %s: lstsq hatasi: %s", bar_eid, target_name, e)
                continue

            # R² hesapla
            y_pred = X @ coeffs
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

            eq = RegressionEquation(
                target_name=target_name,
                predictor_names=pred_names,
                coefficients=coeffs,
                r_squared=max(0.0, r_squared),
                n_samples=n_valid,
            )
            result.equations.append(eq)

            logger.info(
                "Bar %d | %s | R²=%.4f | n=%d | predictors=%d",
                bar_eid, target_name, r_squared, n_valid, n_predictors,
            )

        self.results.append(result)
        return result

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Tum korelasyon denklemlerini long/vertical format DataFrame olarak dondur.

        Her predictor icin ayri satir:
          Bar_EID | Element_Type | N_Shells | Shell_EIDs | Target |
          Predictor | Coefficient | R_Squared | N_Samples
        """
        rows = []
        for res in self.results:
            shell_eids_str = ",".join(str(s) for s in res.shell_eids_used)
            for eq in res.equations:
                for pname, coeff in zip(eq.predictor_names, eq.coefficients):
                    rows.append({
                        "Bar_EID": res.bar_eid,
                        "Element_Type": res.element_type,
                        "N_Shells": res.n_connected_shells,
                        "Shell_EIDs": shell_eids_str,
                        "Target": eq.target_name,
                        "Predictor": pname,
                        "Coefficient": float(coeff),
                        "R_Squared": eq.r_squared,
                        "N_Samples": eq.n_samples,
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _normalize(s: str) -> str:
        """Kolon isimlerini normalize et."""
        return s.lower().replace(" ", "").replace("_", "").strip()
