"""
OP2 Reader Module

OP2 dosyasından bar elementler için axial force,
CQUAD4/CTRIA3 elementler için membrane fluxlarını (NX, NY, NXY) okur.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyNastran.op2.op2 import OP2

logger = logging.getLogger(__name__)


@dataclass
class BarForceResult:
    """Bir bar element için kuvvet sonuçları."""
    eid: int
    subcase_id: int
    axial_force: np.ndarray          # (ntimes,)
    shear_1: np.ndarray              # (ntimes,)
    shear_2: np.ndarray              # (ntimes,)
    bending_moment_a1: np.ndarray    # (ntimes,)
    bending_moment_a2: np.ndarray    # (ntimes,)
    bending_moment_b1: np.ndarray    # (ntimes,)
    bending_moment_b2: np.ndarray    # (ntimes,)
    torque: np.ndarray               # (ntimes,)


@dataclass
class ShellForceResult:
    """Bir shell element için flux sonuçları."""
    eid: int
    elem_type: str
    subcase_id: int
    membrane_x: np.ndarray     # NX (ntimes,)
    membrane_y: np.ndarray     # NY (ntimes,)
    membrane_xy: np.ndarray    # NXY (ntimes,)
    bending_x: np.ndarray      # MX (ntimes,)
    bending_y: np.ndarray      # MY (ntimes,)
    bending_xy: np.ndarray     # MXY (ntimes,)
    shear_xz: np.ndarray       # QX (ntimes,)
    shear_yz: np.ndarray       # QY (ntimes,)


class OP2Reader:
    """OP2 dosyasından element kuvvetlerini okur."""

    def __init__(self, op2_path: str):
        self.op2_path = op2_path
        self.op2: OP2 = None
        self._available_subcases: List[int] = []

    def read(self) -> None:
        """OP2 dosyasını oku."""
        logger.info("OP2 dosyasi okunuyor: %s", self.op2_path)
        self.op2 = OP2(debug=False)
        self.op2.read_op2(self.op2_path)
        self._available_subcases = self._collect_subcase_ids()
        logger.info("OP2 okundu. Subcases: %s", self._available_subcases)

    @staticmethod
    def read_multiple(
        op2_paths: List[str],
        bar_eids: List[int],
        shell_eids: List[int],
        subcase_id: Optional[int] = None,
    ) -> Tuple[
        Dict[Tuple[int, int], "BarForceResult"],
        Dict[Tuple[int, int], "ShellForceResult"],
        List[int],
    ]:
        """
        Birden fazla OP2 dosyasini oku ve sonuclari birlestir.

        Returns
        -------
        bar_forces : Dict
            Birlestirilmis bar kuvvet sonuclari.
        shell_forces : Dict
            Birlestirilmis shell flux sonuclari.
        all_subcases : List[int]
            Tum OP2'lerdeki subcase ID'leri.
        """
        merged_bar: Dict[Tuple[int, int], BarForceResult] = {}
        merged_shell: Dict[Tuple[int, int], ShellForceResult] = {}
        all_subcases: set = set()

        for op2_path in op2_paths:
            reader = OP2Reader(op2_path)
            reader.read()
            all_subcases.update(reader.subcases)

            bar_f = reader.get_bar_forces(bar_eids, subcase_id)
            for key, val in bar_f.items():
                if key not in merged_bar:
                    merged_bar[key] = val

            shell_f = reader.get_shell_forces(shell_eids, subcase_id)
            for key, val in shell_f.items():
                if key not in merged_shell:
                    merged_shell[key] = val

        logger.info(
            "Toplu OP2: %d dosya, %d subcase, %d bar, %d shell sonuc",
            len(op2_paths), len(all_subcases),
            len(merged_bar), len(merged_shell),
        )
        return merged_bar, merged_shell, sorted(all_subcases)

    @property
    def subcases(self) -> List[int]:
        return self._available_subcases

    def _collect_subcase_ids(self) -> List[int]:
        """OP2 sonuc tablolarindan mevcut subcase ID'lerini topla."""
        sc_ids: set = set()
        # Tum kuvvet sonuc dictionary'lerinden subcase key'lerini topla
        for attr in (
            "cbar_force", "cbeam_force",
            "cquad4_force", "ctria3_force",
            "cquad8_force", "ctria6_force",
            "cshear_force",
        ):
            result_dict = getattr(self.op2, attr, None)
            if result_dict:
                sc_ids.update(result_dict.keys())
        # Ek olarak subcases varsa onu da ekle (bazi pyNastran versiyonlari)
        if hasattr(self.op2, "subcases") and self.op2.subcases:
            sc_ids.update(self.op2.subcases.keys())
        return sorted(sc_ids)

    def _get_subcase_ids(self, subcase_id: Optional[int] = None) -> List[int]:
        """Kullanılacak subcase ID'lerini belirle."""
        if subcase_id is not None:
            return [subcase_id]
        # Tüm subcaseleri kullan
        return self._available_subcases

    def get_bar_forces(
        self,
        bar_eids: List[int],
        subcase_id: Optional[int] = None,
    ) -> Dict[Tuple[int, int], BarForceResult]:
        """
        Bar elementler için axial force ve diğer kuvvetleri çıkar.

        Parameters
        ----------
        bar_eids : List[int]
            Bar element ID listesi.
        subcase_id : Optional[int]
            Belirli bir subcase. None ise tüm subcaseler.

        Returns
        -------
        Dict[Tuple[int, int], BarForceResult]
            (subcase_id, eid) -> BarForceResult mapping.
        """
        results = {}

        # CBAR forces
        for sc_id, force_obj in self.op2.cbar_force.items():
            if subcase_id is not None and sc_id != subcase_id:
                continue

            eids = force_obj.element
            headers = force_obj.get_headers()
            logger.debug("CBAR force headers (SC %d): %s", sc_id, headers)

            # Header index mapping
            h_map = {h.lower().strip(): i for i, h in enumerate(headers)}

            axial_idx = self._find_header_index(h_map, ["axial_force", "axial"])
            shear1_idx = self._find_header_index(h_map, ["shear1", "shear_1", "shear_plane1"])
            shear2_idx = self._find_header_index(h_map, ["shear2", "shear_2", "shear_plane2"])
            bma1_idx = self._find_header_index(h_map, ["bending_moment_a1", "bending_moment1a", "bm_end_a_plane1"])
            bma2_idx = self._find_header_index(h_map, ["bending_moment_a2", "bending_moment2a", "bm_end_a_plane2"])
            bmb1_idx = self._find_header_index(h_map, ["bending_moment_b1", "bending_moment1b", "bm_end_b_plane1"])
            bmb2_idx = self._find_header_index(h_map, ["bending_moment_b2", "bending_moment2b", "bm_end_b_plane2"])
            torque_idx = self._find_header_index(h_map, ["torque"])

            for bar_eid in bar_eids:
                mask = eids == bar_eid
                if not np.any(mask):
                    continue

                idx = np.where(mask)[0][0]
                data = force_obj.data  # (ntimes, nelements, ncolumns)

                results[(sc_id, bar_eid)] = BarForceResult(
                    eid=bar_eid,
                    subcase_id=sc_id,
                    axial_force=data[:, idx, axial_idx] if axial_idx is not None else np.zeros(data.shape[0]),
                    shear_1=data[:, idx, shear1_idx] if shear1_idx is not None else np.zeros(data.shape[0]),
                    shear_2=data[:, idx, shear2_idx] if shear2_idx is not None else np.zeros(data.shape[0]),
                    bending_moment_a1=data[:, idx, bma1_idx] if bma1_idx is not None else np.zeros(data.shape[0]),
                    bending_moment_a2=data[:, idx, bma2_idx] if bma2_idx is not None else np.zeros(data.shape[0]),
                    bending_moment_b1=data[:, idx, bmb1_idx] if bmb1_idx is not None else np.zeros(data.shape[0]),
                    bending_moment_b2=data[:, idx, bmb2_idx] if bmb2_idx is not None else np.zeros(data.shape[0]),
                    torque=data[:, idx, torque_idx] if torque_idx is not None else np.zeros(data.shape[0]),
                )

        # CBEAM forces
        for sc_id, force_obj in self.op2.cbeam_force.items():
            if subcase_id is not None and sc_id != subcase_id:
                continue

            eids = force_obj.element
            headers = force_obj.get_headers()
            h_map = {h.lower().strip(): i for i, h in enumerate(headers)}

            axial_idx = self._find_header_index(h_map, ["axial_force", "axial"])
            shear1_idx = self._find_header_index(h_map, ["shear1", "shear_1"])
            shear2_idx = self._find_header_index(h_map, ["shear2", "shear_2"])
            torque_idx = self._find_header_index(h_map, ["torque"])

            for bar_eid in bar_eids:
                if (sc_id, bar_eid) in results:
                    continue
                mask = eids == bar_eid
                if not np.any(mask):
                    continue

                idx = np.where(mask)[0][0]
                data = force_obj.data

                results[(sc_id, bar_eid)] = BarForceResult(
                    eid=bar_eid,
                    subcase_id=sc_id,
                    axial_force=data[:, idx, axial_idx] if axial_idx is not None else np.zeros(data.shape[0]),
                    shear_1=data[:, idx, shear1_idx] if shear1_idx is not None else np.zeros(data.shape[0]),
                    shear_2=data[:, idx, shear2_idx] if shear2_idx is not None else np.zeros(data.shape[0]),
                    bending_moment_a1=np.zeros(data.shape[0]),
                    bending_moment_a2=np.zeros(data.shape[0]),
                    bending_moment_b1=np.zeros(data.shape[0]),
                    bending_moment_b2=np.zeros(data.shape[0]),
                    torque=data[:, idx, torque_idx] if torque_idx is not None else np.zeros(data.shape[0]),
                )

        found = len(results)
        logger.info(
            "%d bar element icin kuvvet verisi bulundu (istenen: %d)",
            found,
            len(bar_eids),
        )
        return results

    def get_shell_forces(
        self,
        shell_eids: List[int],
        subcase_id: Optional[int] = None,
    ) -> Dict[Tuple[int, int], ShellForceResult]:
        """
        CQUAD4/CTRIA3 elementler için membrane fluxlarını çıkar.

        Parameters
        ----------
        shell_eids : List[int]
            Shell element ID listesi.
        subcase_id : Optional[int]
            Belirli bir subcase. None ise tüm subcaseler.

        Returns
        -------
        Dict[Tuple[int, int], ShellForceResult]
            (subcase_id, eid) -> ShellForceResult mapping.
        """
        results = {}

        # Force result attributes ve karşılık gelen element tipleri
        force_attrs = [
            ("cquad4_force", "CQUAD4"),
            ("cquad8_force", "CQUAD8"),
            ("ctria3_force", "CTRIA3"),
            ("ctria6_force", "CTRIA6"),
        ]

        for attr_name, elem_type in force_attrs:
            force_dict = getattr(self.op2, attr_name, {})
            if not force_dict:
                continue

            for sc_id, force_obj in force_dict.items():
                if subcase_id is not None and sc_id != subcase_id:
                    continue

                eids = force_obj.element
                headers = force_obj.get_headers()
                h_map = {h.lower().strip(): i for i, h in enumerate(headers)}
                logger.debug(
                    "%s force headers (SC %d): %s", elem_type, sc_id, headers
                )

                mx_idx = self._find_header_index(
                    h_map,
                    ["membrane_x", "mx", "nx", "membrane_force_x", "oxx_membrane"],
                )
                my_idx = self._find_header_index(
                    h_map,
                    ["membrane_y", "my", "ny", "membrane_force_y", "oyy_membrane"],
                )
                mxy_idx = self._find_header_index(
                    h_map,
                    ["membrane_xy", "mxy", "nxy", "membrane_force_xy", "oxy_membrane"],
                )
                bx_idx = self._find_header_index(
                    h_map,
                    ["bending_x", "bmx", "bending_moment_x"],
                )
                by_idx = self._find_header_index(
                    h_map,
                    ["bending_y", "bmy", "bending_moment_y"],
                )
                bxy_idx = self._find_header_index(
                    h_map,
                    ["bending_xy", "bmxy", "bending_moment_xy"],
                )
                sx_idx = self._find_header_index(
                    h_map,
                    ["shear_xz", "tx", "qx", "transverse_shear_x"],
                )
                sy_idx = self._find_header_index(
                    h_map,
                    ["shear_yz", "ty", "qy", "transverse_shear_y"],
                )

                for shell_eid in shell_eids:
                    if (sc_id, shell_eid) in results:
                        continue
                    mask = eids == shell_eid
                    if not np.any(mask):
                        continue

                    idx = np.where(mask)[0][0]
                    data = force_obj.data
                    nt = data.shape[0]

                    results[(sc_id, shell_eid)] = ShellForceResult(
                        eid=shell_eid,
                        elem_type=elem_type,
                        subcase_id=sc_id,
                        membrane_x=data[:, idx, mx_idx] if mx_idx is not None else np.zeros(nt),
                        membrane_y=data[:, idx, my_idx] if my_idx is not None else np.zeros(nt),
                        membrane_xy=data[:, idx, mxy_idx] if mxy_idx is not None else np.zeros(nt),
                        bending_x=data[:, idx, bx_idx] if bx_idx is not None else np.zeros(nt),
                        bending_y=data[:, idx, by_idx] if by_idx is not None else np.zeros(nt),
                        bending_xy=data[:, idx, bxy_idx] if bxy_idx is not None else np.zeros(nt),
                        shear_xz=data[:, idx, sx_idx] if sx_idx is not None else np.zeros(nt),
                        shear_yz=data[:, idx, sy_idx] if sy_idx is not None else np.zeros(nt),
                    )

        found = len(results)
        logger.info(
            "%d shell element icin flux verisi bulundu (istenen: %d)",
            found,
            len(shell_eids),
        )
        return results

    def get_load_case_info(self) -> pd.DataFrame:
        """Mevcut subcaselerin bilgisini döndür."""
        rows = []
        subcases_dict = getattr(self.op2, "subcases", None) or {}
        for sc_id in self._available_subcases:
            label = str(sc_id)
            if sc_id in subcases_dict:
                sub = subcases_dict[sc_id]
                if isinstance(sub, dict):
                    label = str(sub.get("SUBTITLE", [""])[0])
            rows.append({"Subcase_ID": sc_id, "Label": label})
        return pd.DataFrame(rows)

    @staticmethod
    def _find_header_index(h_map: Dict[str, int], candidates: List[str]) -> Optional[int]:
        """Header isimlerinden birini h_map'te ara."""
        for c in candidates:
            c_lower = c.lower().strip()
            if c_lower in h_map:
                return h_map[c_lower]
        return None
