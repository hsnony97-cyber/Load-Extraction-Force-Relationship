#!/usr/bin/env python3
"""
Joint Load Extraction Force Relationship Tool - Standalone
==========================================================

Tum modulleri tek dosyada birlestiren bagimsiz uygulama.
Harici 'joint_load_extractor' paketine ihtiyac duymadan calisir.

Kullanim:
    python app_standalone.py                  # GUI baslatir
    python app_standalone.py --cli --bdf ...  # CLI modunda calistirir
    python app_standalone.py --help           # Yardim
"""

import argparse
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2



# ============================================================
#  BDF Parser
# ============================================================


@dataclass
class BarElementInfo:
    """Bir bar element ve bagli elemanlarinin bilgisi."""
    eid: int
    pid: int
    node_a: int
    node_b: int
    connected_quads: Dict[int, List[int]] = field(default_factory=dict)
    connected_trias: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def nodes(self) -> Set[int]:
        return {self.node_a, self.node_b}

    @property
    def all_connected_eids(self) -> List[int]:
        return list(self.connected_quads.keys()) + list(self.connected_trias.keys())


class BDFParser:
    """BDF dosyasini parse eder ve element baglantilarini cikarir."""

    BAR_TYPES = {"CBAR", "CBEAM"}
    QUAD_TYPES = {"CQUAD4", "CQUAD8", "CQUADR"}
    TRIA_TYPES = {"CTRIA3", "CTRIA6", "CTRIAR"}

    def __init__(self, bdf_path: str):
        self.bdf_path = bdf_path
        self.model: BDF = None
        self._node_to_quads: Dict[int, Set[int]] = defaultdict(set)
        self._node_to_trias: Dict[int, Set[int]] = defaultdict(set)
        self._element_nodes: Dict[int, List[int]] = {}
        self._element_types: Dict[int, str] = {}
        self._element_pids: Dict[int, int] = {}

    def parse(self) -> None:
        """BDF dosyasini oku ve element verilerini indeksle."""
        logger = logging.getLogger(__name__)
        logger.info("BDF dosyasi okunuyor: %s", self.bdf_path)

        # Tum INCLUDE'lari recursive olarak ac, bulunamayanlari atla
        skipped_count = [0]
        expanded = self._expand_includes_recursive(
            self.bdf_path, visited=set(), skipped_count=skipped_count,
        )
        if skipped_count[0] > 0:
            logger.info("%d erisilemez INCLUDE atlandi", skipped_count[0])

        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bdf")
            os.close(tmp_fd)
            with open(tmp_path, "w", encoding="utf-8") as dst:
                dst.write("\n".join(expanded))

            self.model = BDF(debug=False)
            self.model.read_bdf(
                tmp_path, validate=False, xref=False,
                read_includes=False, encoding="utf-8",
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        logger.info(
            "BDF okundu: %d element, %d node",
            len(self.model.elements),
            len(self.model.nodes),
        )
        self._build_node_element_maps()

    @classmethod
    def _expand_includes_recursive(
        cls, bdf_path: str, visited: set, skipped_count: list,
    ) -> list:
        """INCLUDE dosyalarini recursive olarak acar, bulunamayanlari atlar."""
        logger = logging.getLogger(__name__)
        abs_path = os.path.abspath(bdf_path)
        if abs_path in visited:
            return [f"$ CIRCULAR INCLUDE SKIPPED: {abs_path}"]
        visited.add(abs_path)

        content = cls._read_file_safe(bdf_path)
        bdf_dir = os.path.dirname(abs_path)
        result = []

        for line in content.split("\n"):
            stripped = line.strip().upper()
            if stripped.startswith("INCLUDE"):
                inc_file = cls._parse_include_path(line)
                if inc_file is not None:
                    if not os.path.isabs(inc_file):
                        inc_full = os.path.join(bdf_dir, inc_file)
                    else:
                        inc_full = inc_file
                    if os.path.exists(inc_full):
                        result.append(f"$ EXPANDED: {inc_file}")
                        result.extend(
                            cls._expand_includes_recursive(
                                inc_full, visited, skipped_count,
                            )
                        )
                    else:
                        logger.warning("INCLUDE bulunamadi, atlaniyor: %s", inc_file)
                        result.append(f"$ SKIPPED: {line.strip()}")
                        skipped_count[0] += 1
                else:
                    result.append(line)
            else:
                result.append(line)

        return result

    @staticmethod
    def _parse_include_path(line: str) -> str:
        """INCLUDE satirindan dosya yolunu cikarir."""
        raw = line.strip()
        for q in ("'", '"'):
            start = raw.find(q)
            if start != -1:
                end = raw.find(q, start + 1)
                if end != -1:
                    return raw[start + 1:end]
        return None

    @staticmethod
    def _read_file_safe(fpath: str) -> str:
        """Dosyayi guvenli sekilde oku (multi-encoding)."""
        for enc in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
            try:
                with open(fpath, "r", encoding=enc, errors="replace") as f:
                    return f.read()
            except Exception:
                continue
        return ""

    def _build_node_element_maps(self) -> None:
        """Node-to-element haritalarini olustur."""
        logger = logging.getLogger(__name__)
        for eid, elem in self.model.elements.items():
            elem_type = elem.type
            try:
                node_ids = [n if isinstance(n, int) else n.nid for n in elem.nodes]
            except AttributeError:
                node_ids = list(elem.node_ids)

            self._element_nodes[eid] = node_ids
            self._element_types[eid] = elem_type

            try:
                self._element_pids[eid] = elem.pid if isinstance(elem.pid, int) else elem.pid_ref.pid
            except (AttributeError, TypeError):
                self._element_pids[eid] = 0

            if elem_type in self.QUAD_TYPES:
                for nid in node_ids:
                    if nid is not None:
                        self._node_to_quads[nid].add(eid)
            elif elem_type in self.TRIA_TYPES:
                for nid in node_ids:
                    if nid is not None:
                        self._node_to_trias[nid].add(eid)

        logger.info(
            "Node-element haritalari olusturuldu: %d QUAD node, %d TRIA node",
            len(self._node_to_quads),
            len(self._node_to_trias),
        )

    def get_bar_connectivity(
        self, bar_element_ids: List[int]
    ) -> Dict[int, BarElementInfo]:
        """Verilen bar element ID'leri icin bagli QUAD/TRIA elementlerini bul."""
        logger = logging.getLogger(__name__)
        result = {}
        missing_eids = []

        for eid in bar_element_ids:
            if eid not in self.model.elements:
                missing_eids.append(eid)
                logger.warning("Bar element %d BDF'de bulunamadi", eid)
                continue

            elem = self.model.elements[eid]
            elem_type = elem.type

            if elem_type not in self.BAR_TYPES:
                logger.warning(
                    "Element %d tipi %s, bar/beam degil - atlaniyor",
                    eid, elem_type,
                )
                continue

            try:
                nodes = [n if isinstance(n, int) else n.nid for n in elem.nodes]
            except AttributeError:
                nodes = list(elem.node_ids)

            node_a, node_b = nodes[0], nodes[1]

            try:
                pid = elem.pid if isinstance(elem.pid, int) else elem.pid_ref.pid
            except (AttributeError, TypeError):
                pid = 0

            bar_info = BarElementInfo(
                eid=eid, pid=pid, node_a=node_a, node_b=node_b,
            )

            # Bar'in HER IKI node'una da bagli olanlar (intersection)
            quads_at_a = self._node_to_quads.get(node_a, set())
            quads_at_b = self._node_to_quads.get(node_b, set())
            connected_quad_eids = quads_at_a & quads_at_b

            for qeid in sorted(connected_quad_eids):
                bar_info.connected_quads[qeid] = self._element_nodes[qeid]

            trias_at_a = self._node_to_trias.get(node_a, set())
            trias_at_b = self._node_to_trias.get(node_b, set())
            connected_tria_eids = trias_at_a & trias_at_b

            for teid in sorted(connected_tria_eids):
                bar_info.connected_trias[teid] = self._element_nodes[teid]

            result[eid] = bar_info
            logger.info(
                "Bar %d: NodeA=%d, NodeB=%d, %d QUAD, %d TRIA bagli",
                eid, node_a, node_b,
                len(bar_info.connected_quads),
                len(bar_info.connected_trias),
            )

        if missing_eids:
            logger.warning(
                "BDF'de bulunamayan %d bar element var: %s",
                len(missing_eids), missing_eids[:10],
            )

        return result

    def get_element_type(self, eid: int) -> str:
        return self._element_types.get(eid, "UNKNOWN")

    def get_element_nodes(self, eid: int) -> List[int]:
        return self._element_nodes.get(eid, [])

    def get_element_pid(self, eid: int) -> int:
        return self._element_pids.get(eid, 0)


# ============================================================
#  OP2 Reader
# ============================================================


@dataclass
class BarForceResult:
    """Bir bar element icin kuvvet sonuclari."""
    eid: int
    subcase_id: int
    axial_force: np.ndarray
    shear_1: np.ndarray
    shear_2: np.ndarray
    bending_moment_a1: np.ndarray
    bending_moment_a2: np.ndarray
    bending_moment_b1: np.ndarray
    bending_moment_b2: np.ndarray
    torque: np.ndarray


@dataclass
class ShellForceResult:
    """Bir shell element icin flux sonuclari."""
    eid: int
    elem_type: str
    subcase_id: int
    membrane_x: np.ndarray
    membrane_y: np.ndarray
    membrane_xy: np.ndarray
    bending_x: np.ndarray
    bending_y: np.ndarray
    bending_xy: np.ndarray
    shear_xz: np.ndarray
    shear_yz: np.ndarray


class OP2Reader:
    """OP2 dosyasindan element kuvvetlerini okur."""

    def __init__(self, op2_path: str):
        self.op2_path = op2_path
        self.op2: OP2 = None
        self._available_subcases: List[int] = []

    def read(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info("OP2 dosyasi okunuyor: %s", self.op2_path)
        self.op2 = OP2(debug=False)
        self.op2.read_op2(self.op2_path)
        self._available_subcases = self._collect_subcase_ids()
        logger.info("OP2 okundu. Subcases: %s", self._available_subcases)

    @property
    def subcases(self) -> List[int]:
        return self._available_subcases

    def _collect_subcase_ids(self) -> List[int]:
        """OP2 sonuc tablolarindan mevcut subcase ID'lerini topla."""
        sc_ids: set = set()
        for attr in (
            "cbar_force", "cbeam_force",
            "cquad4_force", "ctria3_force",
            "cquad8_force", "ctria6_force",
            "cshear_force",
        ):
            result_dict = getattr(self.op2, attr, None)
            if result_dict:
                sc_ids.update(result_dict.keys())
        if hasattr(self.op2, "subcases") and self.op2.subcases:
            sc_ids.update(self.op2.subcases.keys())
        return sorted(sc_ids)

    def get_bar_forces(
        self,
        bar_eids: List[int],
        subcase_id: Optional[int] = None,
    ) -> Dict[Tuple[int, int], BarForceResult]:
        """Bar elementler icin kuvvetleri cikar."""
        logger = logging.getLogger(__name__)
        results = {}

        # CBAR forces
        for sc_id, force_obj in self.op2.cbar_force.items():
            if subcase_id is not None and sc_id != subcase_id:
                continue

            eids = force_obj.element
            headers = force_obj.get_headers()
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
                data = force_obj.data

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

        logger.info(
            "%d bar element icin kuvvet verisi bulundu (istenen: %d)",
            len(results), len(bar_eids),
        )
        return results

    def get_shell_forces(
        self,
        shell_eids: List[int],
        subcase_id: Optional[int] = None,
    ) -> Dict[Tuple[int, int], ShellForceResult]:
        """CQUAD4/CTRIA3 elementler icin membrane fluxlarini cikar."""
        logger = logging.getLogger(__name__)
        results = {}

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

                mx_idx = self._find_header_index(
                    h_map, ["membrane_x", "mx", "nx", "membrane_force_x", "oxx_membrane"],
                )
                my_idx = self._find_header_index(
                    h_map, ["membrane_y", "my", "ny", "membrane_force_y", "oyy_membrane"],
                )
                mxy_idx = self._find_header_index(
                    h_map, ["membrane_xy", "mxy", "nxy", "membrane_force_xy", "oxy_membrane"],
                )
                bx_idx = self._find_header_index(
                    h_map, ["bending_x", "bmx", "bending_moment_x"],
                )
                by_idx = self._find_header_index(
                    h_map, ["bending_y", "bmy", "bending_moment_y"],
                )
                bxy_idx = self._find_header_index(
                    h_map, ["bending_xy", "bmxy", "bending_moment_xy"],
                )
                sx_idx = self._find_header_index(
                    h_map, ["shear_xz", "tx", "qx", "transverse_shear_x"],
                )
                sy_idx = self._find_header_index(
                    h_map, ["shear_yz", "ty", "qy", "transverse_shear_y"],
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

        logger.info(
            "%d shell element icin flux verisi bulundu (istenen: %d)",
            len(results), len(shell_eids),
        )
        return results

    def get_load_case_info(self) -> pd.DataFrame:
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
        bar_forces, shell_forces, all_subcases
        """
        logger = logging.getLogger(__name__)
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

    @staticmethod
    def _find_header_index(h_map: Dict[str, int], candidates: List[str]) -> Optional[int]:
        for c in candidates:
            c_lower = c.lower().strip()
            if c_lower in h_map:
                return h_map[c_lower]
        return None


# ============================================================
#  H5 Reader
# ============================================================


class H5Reader:
    """H5 dosyasindan JOINT_LOADS_CAP verilerini okur."""

    EXPECTED_COLUMNS = [
        "F Bearing X",
        "F Bearing Y",
        "NX Bypass",
        "NY Bypass",
        "NXY Bypass",
    ]

    METADATA_COLUMNS = [
        "Subcase ID",
        "Element Type",
    ]

    ID_COLUMN_CANDIDATES = [
        "Bar Element ID",
        "Bar EID",
        "BarEID",
        "Bar_EID",
        "Bar_Element_ID",
        "EID",
        "Element_ID",
        "ElementID",
        "Fastener_ID",
        "FastenerID",
        "ID",
        "Joint_ID",
        "JointID",
    ]

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._joint_load_cap_df: Optional[pd.DataFrame] = None
        self._h5_structure: Dict = {}

    def read(self, group_path: str = None) -> None:
        logger = logging.getLogger(__name__)
        logger.info("H5 dosyasi okunuyor: %s", self.h5_path)

        with h5py.File(self.h5_path, "r") as f:
            self._h5_structure = self._explore_structure(f)
            logger.info("H5 agac yapisi:\n%s", self._format_structure(self._h5_structure))

            if group_path:
                table_path = group_path
            else:
                table_path = self._find_joint_load_cap_table(f)

            if table_path is None:
                raise ValueError(
                    "H5 dosyasinda JOINT_LOADS_CAP tablosu bulunamadi. "
                    "Lutfen group_path parametresini belirtin."
                )

            logger.info("JOINT_LOADS_CAP tablosu bulundu: %s", table_path)
            self._joint_load_cap_df = self._read_table(f, table_path)

        logger.info(
            "JOINT_LOADS_CAP tablosu okundu: %d satir, %d kolon",
            len(self._joint_load_cap_df),
            len(self._joint_load_cap_df.columns),
        )
        logger.info("Kolonlar: %s", list(self._joint_load_cap_df.columns))

    @property
    def joint_load_cap(self) -> pd.DataFrame:
        if self._joint_load_cap_df is None:
            raise RuntimeError("Henuz H5 dosyasi okunmadi. Once read() cagiriniz.")
        return self._joint_load_cap_df

    @property
    def structure(self) -> Dict:
        return self._h5_structure

    def get_joint_loads_for_bar(self, bar_eid: int) -> Optional[pd.DataFrame]:
        logger = logging.getLogger(__name__)
        df = self.joint_load_cap
        id_col = self._find_id_column(df)

        if id_col is None:
            logger.warning("JOINT_LOADS_CAP tablosunda ID kolonu bulunamadi")
            return None

        mask = df[id_col] == bar_eid
        if not mask.any():
            return None

        return df[mask].copy()

    def get_all_bar_eids(self) -> List[int]:
        df = self.joint_load_cap
        id_col = self._find_id_column(df)
        if id_col is None:
            return []
        return sorted(df[id_col].unique().tolist())

    def _find_joint_load_cap_table(self, f: h5py.File) -> Optional[str]:
        candidates = []

        def _visitor(name, obj):
            name_lower = name.lower()
            if "joint" in name_lower and "load" in name_lower and "cap" in name_lower:
                if isinstance(obj, h5py.Dataset):
                    candidates.append(name)
                elif isinstance(obj, h5py.Group):
                    if "table" in obj:
                        candidates.append(f"{name}/table")

        f.visititems(_visitor)

        if not candidates:
            def _visitor_table(name, obj):
                if name.endswith("/table") and isinstance(obj, h5py.Dataset):
                    candidates.append(name)
            f.visititems(_visitor_table)

        if candidates:
            for c in candidates:
                if "cap" in c.lower() and "table" in c.lower():
                    return c
            return candidates[0]

        return None

    def _read_table(self, f: h5py.File, path: str) -> pd.DataFrame:
        dataset = f[path]

        if dataset.dtype.names:
            data = {}
            for col_name in dataset.dtype.names:
                col_data = dataset[col_name]
                if col_data.dtype.kind == "S" or col_data.dtype.kind == "O":
                    try:
                        col_data = np.array(
                            [x.decode("utf-8") if isinstance(x, bytes) else x for x in col_data]
                        )
                    except (UnicodeDecodeError, AttributeError):
                        pass
                data[col_name] = col_data
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(dataset[:])

    def _find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        logger = logging.getLogger(__name__)
        columns_lower = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

        for candidate in self.ID_COLUMN_CANDIDATES:
            candidate_normalized = candidate.lower().replace(" ", "").replace("_", "")
            if candidate_normalized in columns_lower:
                return columns_lower[candidate_normalized]

        for col in df.columns:
            if df[col].dtype in [np.int32, np.int64, np.uint32, np.uint64]:
                logger.info("ID kolonu olarak '%s' kullaniliyor", col)
                return col

        return None

    def _explore_structure(self, f: h5py.File, max_depth: int = 5) -> Dict:
        structure = {}

        def _visit(group, current_dict, depth=0):
            if depth > max_depth:
                return
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Group):
                    current_dict[key] = {"_type": "group", "_children": {}}
                    _visit(item, current_dict[key]["_children"], depth + 1)
                elif isinstance(item, h5py.Dataset):
                    info = {
                        "_type": "dataset",
                        "shape": item.shape,
                        "dtype": str(item.dtype),
                    }
                    if item.dtype.names:
                        info["columns"] = list(item.dtype.names)
                    current_dict[key] = info

        _visit(f, structure)
        return structure

    def _format_structure(self, structure: Dict, indent: int = 0) -> str:
        lines = []
        prefix = "  " * indent
        for key, value in structure.items():
            if isinstance(value, dict):
                if value.get("_type") == "group":
                    lines.append(f"{prefix}[G] {key}/")
                    lines.append(
                        self._format_structure(value.get("_children", {}), indent + 1)
                    )
                elif value.get("_type") == "dataset":
                    cols = value.get("columns", [])
                    shape = value.get("shape", "?")
                    if cols:
                        lines.append(f"{prefix}[D] {key} {shape} columns={cols}")
                    else:
                        lines.append(f"{prefix}[D] {key} {shape} dtype={value.get('dtype', '?')}")
        return "\n".join(lines)


# ============================================================
#  Correlation Engine
# ============================================================


@dataclass
class RegressionEquation:
    """Tek bir hedef icin coklu regresyon denklemi."""
    target_name: str
    predictor_names: List[str]
    coefficients: np.ndarray
    r_squared: float
    n_samples: int

    def equation_str(self) -> str:
        parts = []
        for name, coeff in zip(self.predictor_names, self.coefficients):
            parts.append(f"{coeff:+.6f}*{name}")
        return f"{self.target_name} = {' '.join(parts)}"


@dataclass
class JointCorrelationResult:
    """Bir bar element + element type icin tum korelasyon sonuclari."""
    bar_eid: int
    subcase_id: int
    element_type: int  # H5 Element Type (0, 1, ...)
    n_connected_shells: int
    shell_eids_used: List[int] = field(default_factory=list)
    shell_eid: Optional[int] = None       # Per-shell analiz icin shell element ID
    shell_type: Optional[str] = None      # Per-shell analiz icin shell tipi (CQUAD4, CTRIA3, ...)
    equations: List[RegressionEquation] = field(default_factory=list)
    predictor_data: Dict[str, np.ndarray] = field(default_factory=dict)
    target_data: Dict[str, np.ndarray] = field(default_factory=dict)
    matched_subcases: List[int] = field(default_factory=list)


# Geriye uyumluluk icin alias
CorrelationResult = JointCorrelationResult


class CorrelationEngine:
    """OP2 kuvvetleri ile H5 JOINT_LOADS_CAP arasinda coklu regresyon hesaplar."""

    H5_TARGET_COLUMNS = ["F Bearing X", "F Bearing Y", "NX Bypass", "NY Bypass", "NXY Bypass"]

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
        bar_axial : np.ndarray
            Bar axial force (n_subcases,).
        shell_forces_per_eid : dict
            {shell_eid: {"nx": array, "ny": array, "nxy": array}}
            Her shell icin ayri kuvvet arrayleri (n_subcases,).
        """
        logger = logging.getLogger(__name__)
        sorted_shell_eids = sorted(shell_forces_per_eid.keys())

        result = JointCorrelationResult(
            bar_eid=bar_eid, subcase_id=subcase_id,
            element_type=element_type, n_connected_shells=n_shells,
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

        targets = {}
        for col in self.H5_TARGET_COLUMNS:
            matching = [c for c in h5_data.columns
                        if self._normalize(c) == self._normalize(col)]
            if matching:
                targets[col] = h5_data[matching[0]].values.astype(float)
        result.target_data = targets

        if not targets:
            logger.warning("Bar %d: H5 hedef kolonlari bulunamadi", bar_eid)
            self.results.append(result)
            return result

        all_arrays = list(predictors.values()) + list(targets.values())
        min_len = min(len(a) for a in all_arrays)

        # Minimum veri noktasi: en az predictor sayisi + 1
        min_required = max(5, n_predictors + 1)
        if min_len < min_required:
            logger.warning(
                "Bar %d: Yetersiz veri noktasi (%d < %d gerekli), "
                "regresyon atlaniyor (predictor=%d)",
                bar_eid, min_len, min_required, n_predictors,
            )
            self.results.append(result)
            return result

        # Predictor matrisi: (n, 1 + 3*n_shells)
        pred_names = list(predictors.keys())
        X_raw = np.column_stack([predictors[k][:min_len] for k in pred_names])

        for target_name, target_arr in targets.items():
            y = target_arr[:min_len]
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

            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y_valid, rcond=None)
            except np.linalg.LinAlgError as e:
                logger.warning("Bar %d, %s: lstsq hatasi: %s", bar_eid, target_name, e)
                continue

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
        """Tum korelasyon denklemlerini ozet DataFrame olarak dondur (long format)."""
        rows = []
        for res in self.results:
            for eq in res.equations:
                for pname, coeff in zip(eq.predictor_names, eq.coefficients):
                    row = {
                        "Bar_EID": res.bar_eid,
                        "Element_Type": res.element_type,
                        "N_Shells": res.n_connected_shells,
                        "Shell_EIDs": ",".join(str(s) for s in res.shell_eids_used),
                        "Target": eq.target_name,
                        "Predictor": pname,
                        "Coefficient": float(coeff),
                        "R_Squared": eq.r_squared,
                        "N_Samples": eq.n_samples,
                    }
                    rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _normalize(s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "").strip()


# ============================================================
#  Report Generator
# ============================================================


class ReportGenerator:
    """Excel rapor olusturucu."""

    HEADER_COLOR = "#4472C4"
    HEADER_FONT_COLOR = "#FFFFFF"
    BAR_COLOR = "#D6E4F0"
    QUAD_COLOR = "#E2EFDA"
    TRIA_COLOR = "#FCE4D6"
    CORR_POS_COLOR = "#C6EFCE"
    CORR_NEG_COLOR = "#FFC7CE"

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.writer = None

    def generate(
        self,
        bar_element_ids: List[int],
        connectivity: Dict[int, BarElementInfo],
        bar_forces_df: pd.DataFrame,
        shell_forces_df: pd.DataFrame,
        h5_joint_loads: pd.DataFrame,
        correlation_results: List[CorrelationResult],
        correlation_summary_df: pd.DataFrame,
        per_shell_results: Optional[List[JointCorrelationResult]] = None,
    ) -> str:
        logger = logging.getLogger(__name__)
        logger.info("Excel raporu olusturuluyor: %s", self.output_path)

        with pd.ExcelWriter(self.output_path, engine="xlsxwriter") as writer:
            self.writer = writer
            workbook = writer.book

            header_fmt = workbook.add_format({
                "bold": True,
                "bg_color": self.HEADER_COLOR,
                "font_color": self.HEADER_FONT_COLOR,
                "border": 1,
                "text_wrap": True,
                "valign": "vcenter",
                "align": "center",
            })
            bar_fmt = workbook.add_format({"bg_color": self.BAR_COLOR, "border": 1})
            quad_fmt = workbook.add_format({"bg_color": self.QUAD_COLOR, "border": 1})
            tria_fmt = workbook.add_format({"bg_color": self.TRIA_COLOR, "border": 1})
            number_fmt = workbook.add_format({"num_format": "0.0000", "border": 1})
            int_fmt = workbook.add_format({"num_format": "0", "border": 1})
            border_fmt = workbook.add_format({"border": 1})

            self._write_bar_element_set(writer, workbook, bar_element_ids, header_fmt, int_fmt)

            self._write_connectivity(
                writer, workbook, connectivity, header_fmt,
                bar_fmt, quad_fmt, tria_fmt, int_fmt, border_fmt,
            )

            if not bar_forces_df.empty:
                self._write_dataframe(
                    writer, workbook, "OP2 Bar Forces", bar_forces_df,
                    header_fmt, number_fmt, int_fmt,
                )

            if not shell_forces_df.empty:
                self._write_dataframe(
                    writer, workbook, "OP2 Shell Fluxes", shell_forces_df,
                    header_fmt, number_fmt, int_fmt,
                )

            if not h5_joint_loads.empty:
                self._write_dataframe(
                    writer, workbook, "H5 Joint Loads", h5_joint_loads,
                    header_fmt, number_fmt, int_fmt,
                )

            if not correlation_summary_df.empty:
                self._write_dataframe(
                    writer, workbook, "Correlation Summary", correlation_summary_df,
                    header_fmt, number_fmt, int_fmt,
                )

            self._write_predicted_vs_actual(
                writer, workbook, correlation_results,
                header_fmt, number_fmt, int_fmt, border_fmt,
            )

            # Total Summary - tum bar elementler icin tek tablo (per-shell)
            self._write_total_summary(
                writer, workbook, connectivity,
                per_shell_results if per_shell_results else correlation_results,
                header_fmt, number_fmt, int_fmt, border_fmt,
            )

            self._write_per_bar_details(
                writer, workbook, connectivity, correlation_results,
                header_fmt, number_fmt, int_fmt, bar_fmt, quad_fmt, tria_fmt, border_fmt,
            )

        logger.info("Rapor olusturuldu: %s", self.output_path)
        return self.output_path

    def _write_bar_element_set(self, writer, workbook, bar_eids, header_fmt, int_fmt):
        sheet_name = "Bar Element Set"
        df = pd.DataFrame({"Bar_Element_ID": bar_eids})
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
        ws = writer.sheets[sheet_name]
        ws.write(0, 0, "Bar_Element_ID", header_fmt)
        ws.set_column(0, 0, 18)
        for row_idx in range(len(bar_eids)):
            ws.write(row_idx + 1, 0, bar_eids[row_idx], int_fmt)

    def _write_connectivity(
        self, writer, workbook, connectivity, header_fmt,
        bar_fmt, quad_fmt, tria_fmt, int_fmt, border_fmt,
    ):
        sheet_name = "Element Connectivity"
        rows = []
        for bar_eid, info in sorted(connectivity.items()):
            rows.append({
                "Bar_EID": bar_eid, "Node_A": info.node_a, "Node_B": info.node_b,
                "Connected_Type": "BAR", "Connected_EID": bar_eid,
                "Connected_Nodes": f"{info.node_a}, {info.node_b}", "PID": info.pid,
            })
            for qeid, qnodes in info.connected_quads.items():
                rows.append({
                    "Bar_EID": bar_eid, "Node_A": info.node_a, "Node_B": info.node_b,
                    "Connected_Type": "CQUAD4", "Connected_EID": qeid,
                    "Connected_Nodes": ", ".join(str(n) for n in qnodes if n is not None),
                    "PID": "",
                })
            for teid, tnodes in info.connected_trias.items():
                rows.append({
                    "Bar_EID": bar_eid, "Node_A": info.node_a, "Node_B": info.node_b,
                    "Connected_Type": "CTRIA3", "Connected_EID": teid,
                    "Connected_Nodes": ", ".join(str(n) for n in tnodes if n is not None),
                    "PID": "",
                })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
        ws = writer.sheets[sheet_name]
        columns = list(df.columns)
        for col_idx, col_name in enumerate(columns):
            ws.write(0, col_idx, col_name, header_fmt)
        ws.set_column(0, 0, 12)
        ws.set_column(1, 2, 10)
        ws.set_column(3, 3, 14)
        ws.set_column(4, 4, 14)
        ws.set_column(5, 5, 30)
        ws.set_column(6, 6, 10)

        type_col_idx = columns.index("Connected_Type")
        for row_idx, row in enumerate(rows):
            ctype = row["Connected_Type"]
            fmt = border_fmt
            if ctype == "BAR":
                fmt = bar_fmt
            elif "QUAD" in ctype:
                fmt = quad_fmt
            elif "TRI" in ctype:
                fmt = tria_fmt
            for col_idx in range(len(columns)):
                val = row[columns[col_idx]]
                ws.write(row_idx + 1, col_idx, val, fmt)

    def _write_dataframe(self, writer, workbook, sheet_name, df, header_fmt, number_fmt, int_fmt):
        sheet_name = sheet_name[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
        ws = writer.sheets[sheet_name]
        for col_idx, col_name in enumerate(df.columns):
            ws.write(0, col_idx, col_name, header_fmt)
            max_len = max(len(str(col_name)), df[col_name].astype(str).str.len().max())
            ws.set_column(col_idx, col_idx, min(max_len + 2, 25))

    def _write_total_summary(
        self, writer, workbook, connectivity, correlation_results,
        header_fmt, number_fmt, int_fmt, border_fmt,
    ):
        """
        Total Summary sheet'i - long format, her predictor ayri satir.

        Kolonlar:
          Bar_EID | Element_Type | N_Shells | Shell_EIDs |
          Target | Predictor | Coefficient | R2
        """
        logger = logging.getLogger(__name__)
        rows = []
        for res in correlation_results:
            base = {
                "Bar_EID": res.bar_eid,
                "Element_Type": res.element_type,
                "N_Shells": res.n_connected_shells,
                "Shell_EIDs": ",".join(str(s) for s in res.shell_eids_used),
            }

            for eq in res.equations:
                for pname, coeff in zip(eq.predictor_names, eq.coefficients):
                    row = dict(base)
                    row["Target"] = eq.target_name
                    row["Predictor"] = pname
                    row["Coefficient"] = float(coeff)
                    row["R2"] = eq.r_squared
                    rows.append(row)

        if not rows:
            logger.info("Total Summary: veri yok, sheet atlaniyor")
            return

        df = pd.DataFrame(rows)
        sheet_name = "Total Summary"
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
        ws = writer.sheets[sheet_name]

        for col_idx, col_name in enumerate(df.columns):
            ws.write(0, col_idx, col_name, header_fmt)

        col_widths = {
            "Bar_EID": 12,
            "Element_Type": 12,
            "N_Shells": 10,
            "Shell_EIDs": 20,
            "Target": 16,
            "Predictor": 20,
            "Coefficient": 14,
            "R2": 10,
        }
        for col_idx, col_name in enumerate(df.columns):
            width = col_widths.get(col_name, 14)
            ws.set_column(col_idx, col_idx, width)

        # R2 renklendirme
        good_r2_fmt = workbook.add_format({
            "num_format": "0.0000", "border": 1, "bg_color": self.CORR_POS_COLOR,
        })
        bad_r2_fmt = workbook.add_format({
            "num_format": "0.0000", "border": 1, "bg_color": self.CORR_NEG_COLOR,
        })

        r2_col_idx = list(df.columns).index("R2")
        for row_idx, row_data in enumerate(rows):
            val = row_data.get("R2", "")
            if isinstance(val, (int, float)):
                fmt = good_r2_fmt if val >= 0.7 else bad_r2_fmt
                ws.write(row_idx + 1, r2_col_idx, val, fmt)

        # Freeze panes: baslik satiri ve ilk 4 kolon sabit
        ws.freeze_panes(1, 4)

        logger.info("Total Summary: %d satir yazildi", len(rows))

    def _write_predicted_vs_actual(
        self, writer, workbook, correlation_results,
        header_fmt, number_fmt, int_fmt, border_fmt,
    ):
        """Predicted vs Actual sheet'i - her subcase icin tahmin ve hata orani."""
        logger = logging.getLogger(__name__)
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

            for eq in res.equations:
                actual_arr = res.target_data.get(eq.target_name)
                if actual_arr is None:
                    continue
                actual_arr = actual_arr[:n]

                predicted_arr = X @ eq.coefficients

                for i in range(n):
                    actual_val = float(actual_arr[i])
                    pred_val = float(predicted_arr[i])

                    if abs(actual_val) > 1e-10:
                        error_pct = (pred_val - actual_val) / actual_val * 100.0
                    else:
                        error_pct = 0.0

                    row = {
                        "Bar_EID": res.bar_eid,
                        "Element_Type": res.element_type,
                        "Subcase_ID": subcases[i] if i < len(subcases) else 0,
                        "Target": eq.target_name,
                    }
                    # Per-shell predictor degerleri
                    for col_idx, pname in enumerate(pred_names):
                        row[pname] = float(X[i, col_idx])

                    row["Actual"] = actual_val
                    row["Predicted"] = pred_val
                    row["Error_%"] = error_pct
                    rows.append(row)

        if not rows:
            logger.info("Predicted vs Actual: veri yok, sheet atlaniyor")
            return

        df = pd.DataFrame(rows)
        sheet_name = "Predicted vs Actual"
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
        ws = writer.sheets[sheet_name]

        for col_idx, col_name in enumerate(df.columns):
            ws.write(0, col_idx, col_name, header_fmt)

        # Kolon genislikleri: sabit kolonlar + dinamik predictor + sonuc kolonlari
        for col_idx, col_name in enumerate(df.columns):
            max_len = max(len(str(col_name)), 12)
            ws.set_column(col_idx, col_idx, min(max_len + 2, 20))

        good_err_fmt = workbook.add_format({
            "num_format": "0.00", "border": 1, "bg_color": self.CORR_POS_COLOR,
        })
        bad_err_fmt = workbook.add_format({
            "num_format": "0.00", "border": 1, "bg_color": self.CORR_NEG_COLOR,
        })
        if "Error_%" in df.columns:
            err_col_idx = list(df.columns).index("Error_%")
            for row_idx, row_data in enumerate(rows):
                err_val = abs(row_data["Error_%"])
                if err_val <= 10:
                    ws.write(row_idx + 1, err_col_idx, row_data["Error_%"], good_err_fmt)
                elif err_val > 25:
                    ws.write(row_idx + 1, err_col_idx, row_data["Error_%"], bad_err_fmt)

        logger.info("Predicted vs Actual: %d satir yazildi", len(rows))

    def _write_per_bar_details(
        self, writer, workbook, connectivity, correlation_results,
        header_fmt, number_fmt, int_fmt, bar_fmt, quad_fmt, tria_fmt, border_fmt,
    ):
        """Her bar element icin detay sheet'i - regresyon denklemleri."""
        bar_results = {}
        for res in correlation_results:
            if res.bar_eid not in bar_results:
                bar_results[res.bar_eid] = []
            bar_results[res.bar_eid].append(res)

        good_r2_fmt = workbook.add_format({
            "num_format": "0.0000", "border": 1, "bg_color": self.CORR_POS_COLOR,
        })
        eq_fmt = workbook.add_format({"border": 1, "text_wrap": True, "font_size": 9})

        for bar_eid, results in sorted(bar_results.items()):
            sheet_name = f"Bar_{bar_eid}"[:31]
            ws = workbook.add_worksheet(sheet_name)
            row = 0

            title_fmt = workbook.add_format({"bold": True, "font_size": 14, "bottom": 2})
            ws.write(row, 0, f"Bar Element {bar_eid} - Regression", title_fmt)
            row += 2

            ws.write(row, 0, "Element Connectivity", header_fmt)
            ws.merge_range(row, 0, row, 1, "Element Connectivity", header_fmt)
            row += 1
            info = connectivity.get(bar_eid)
            if info:
                ws.write(row, 0, "Bar EID:", bar_fmt)
                ws.write(row, 1, bar_eid, int_fmt)
                row += 1
                ws.write(row, 0, "Node A:", bar_fmt)
                ws.write(row, 1, info.node_a, int_fmt)
                row += 1
                ws.write(row, 0, "Node B:", bar_fmt)
                ws.write(row, 1, info.node_b, int_fmt)
                row += 1
                ws.write(row, 0, "Connected QUAD:", quad_fmt)
                ws.write(row, 1, ", ".join(str(e) for e in info.connected_quads.keys()), border_fmt)
                row += 1
                ws.write(row, 0, "Connected TRIA:", tria_fmt)
                ws.write(row, 1, ", ".join(str(e) for e in info.connected_trias.keys()), border_fmt)
                row += 2

            for res in results:
                # Predictor isimlerini ilk equation'dan al
                pred_names = res.equations[0].predictor_names if res.equations else ["Bar_Axial"]
                n_coeff_cols = len(pred_names)
                merge_end = 3  # Target + Predictor + Coefficient + R2

                ws.write(row, 0, f"SC {res.subcase_id} ET {res.element_type}", header_fmt)
                ws.merge_range(
                    row, 0, row, merge_end,
                    f"Multiple Regression (SC {res.subcase_id}, Element Type {res.element_type}, "
                    f"{res.n_connected_shells} shells, {n_coeff_cols} predictors)",
                    header_fmt,
                )
                row += 1

                if not res.equations:
                    ws.write(row, 0, "Denklem hesaplanamadi", border_fmt)
                    row += 2
                    continue

                eq_headers = ["Target", "Predictor", "Coefficient", "R\u00b2"]

                for j, h in enumerate(eq_headers):
                    ws.write(row, j, h, header_fmt)
                row += 1

                for eq in res.equations:
                    for pname, coeff in zip(eq.predictor_names, eq.coefficients):
                        ws.write(row, 0, eq.target_name, border_fmt)
                        ws.write(row, 1, pname, border_fmt)
                        ws.write(row, 2, float(coeff), number_fmt)
                        r2_fmt = good_r2_fmt if eq.r_squared >= 0.7 else number_fmt
                        ws.write(row, 3, eq.r_squared, r2_fmt)
                        row += 1

                row += 1

                ws.write(row, 0, "Equations", header_fmt)
                ws.merge_range(row, 0, row, merge_end, "Equations", header_fmt)
                row += 1
                for eq in res.equations:
                    ws.merge_range(row, 0, row, merge_end, eq.equation_str(), eq_fmt)
                    row += 1
                row += 1

            ws.set_column(0, 0, 18)
            ws.set_column(1, 1, 20)
            ws.set_column(2, 2, 14)
            ws.set_column(3, 3, 12)


# ============================================================
#  Predictor
# ============================================================

# Prediction H5 sabit tablo yollari
_BAR_TABLE_PATH = "ELFORCE_BAR_COMBINED/table"
_SHELL_TABLE_PATH = "ELFORCE_SHELL_COMBINED/table"

# Target -> CSV kolon adi (Predicted vs Actual ile ayni mantik)
_TARGET_COL_MAP = {
    "F Bearing X": "Pred_FX",
    "F Bearing Y": "Pred_FY",
    "NX Bypass": "Pred_NX",
    "NY Bypass": "Pred_NY",
    "NXY Bypass": "Pred_NXY",
}

# Cikti kolon sirasi (sabit kolonlar; per-shell kolonlar dinamik eklenir)
_OUTPUT_FIXED_COLUMNS = [
    "Bar_EID", "Element_Type", "Subcase_ID",
]
_OUTPUT_PRED_COLUMNS = [
    "Pred_FX", "Pred_FY", "Pred_NX", "Pred_NY", "Pred_NXY",
]


def _pred_order_output(df: pd.DataFrame) -> pd.DataFrame:
    """Cikti kolon siralamasini duzenle: sabit kolonlar + predictor'lar + prediction'lar."""
    fixed = [c for c in _OUTPUT_FIXED_COLUMNS if c in df.columns]
    pred_cols = [c for c in _OUTPUT_PRED_COLUMNS if c in df.columns]
    # Per-shell predictor kolonlari (Bar_Axial, Shell_*_Nx, ...)
    predictor_cols = [c for c in df.columns
                      if c.startswith("Bar_Axial") or c.startswith("Shell_")]
    # Kalan kolonlar
    used = set(fixed + predictor_cols + pred_cols)
    extra = [c for c in df.columns if c not in used]
    return df[fixed + sorted(predictor_cols) + pred_cols + extra]


def _pred_normalize_bar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Bar forces DataFrame kolon isimlerini normalize et."""
    logger = logging.getLogger(__name__)
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


def _pred_normalize_shell_df(df: pd.DataFrame) -> pd.DataFrame:
    """Shell forces DataFrame kolon isimlerini normalize et."""
    logger = logging.getLogger(__name__)
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


def _read_h5_table(f, ds_path: str) -> pd.DataFrame:
    """H5 dataset'ini DataFrame'e cevir."""
    dataset = f[ds_path]
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


def read_prediction_h5(h5_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prediction H5 dosyasini oku.

    Sabit yollar:
      - ELFORCE_BAR_COMBINED/table  -> Element_ID, Subcase_ID, AF
      - ELFORCE_SHELL_COMBINED/table -> Element_ID, Subcase_ID, MX, MY, MXY

    MX->NX, MY->NY, MXY->NXY olarak yeniden adlandirilir.
    Element_ID -> Bar_EID (bar), Shell_EID (shell).
    """
    logger = logging.getLogger(__name__)
    logger.info("Prediction H5 okunuyor: %s", h5_path)

    bar_df = pd.DataFrame()
    shell_df = pd.DataFrame()

    with h5py.File(h5_path, "r") as f:
        # --- Bar element combined ---
        if _BAR_TABLE_PATH in f:
            bar_df = _read_h5_table(f, _BAR_TABLE_PATH)
            if "Element_ID" in bar_df.columns:
                bar_df = bar_df.rename(columns={"Element_ID": "Bar_EID"})
            logger.info("Bar tablosu okundu: %s (%d satir, kolonlar: %s)",
                       _BAR_TABLE_PATH, len(bar_df), list(bar_df.columns))
        else:
            logger.warning("Prediction H5'te '%s' bulunamadi", _BAR_TABLE_PATH)
            def _log_struct(group, prefix=""):
                for key in group:
                    path = f"{prefix}/{key}" if prefix else key
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        _log_struct(item, path)
                    elif isinstance(item, h5py.Dataset):
                        cols = list(item.dtype.names) if item.dtype.names else []
                        logger.info("  %s: %s", path, cols[:10])
            _log_struct(f)

        # --- Shell force combined ---
        if _SHELL_TABLE_PATH in f:
            shell_df = _read_h5_table(f, _SHELL_TABLE_PATH)
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
                       _SHELL_TABLE_PATH, len(shell_df), list(shell_df.columns))
        else:
            logger.warning("Prediction H5'te '%s' bulunamadi", _SHELL_TABLE_PATH)

    return bar_df, shell_df


def read_coefficients_from_excel(excel_path: str) -> pd.DataFrame:
    """Correlation Summary Excel'deki katsayilari oku (ortalama shell)."""
    logger = logging.getLogger(__name__)
    logger.info("Katsayilar Excel'den okunuyor: %s", excel_path)

    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names
    logger.info("Excel sheet'leri: %s", sheet_names)

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

    required = ["Bar_EID", "Element_Type", "Target", "Predictor", "Coefficient"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Summary sheet'te eksik kolonlar: {missing}. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    # Numeric kolonlari donustur
    for col in ["Coefficient"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Bar_EID", "Element_Type"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # String satir temizligi: Bar_EID NaN olan satirlari at
    df = df.dropna(subset=["Bar_EID"])
    df["Coefficient"] = df["Coefficient"].fillna(0.0)

    logger.info("  %d katsayi satiri okundu (long format)", len(df))
    logger.info("  Unique Bar_EID: %d", df["Bar_EID"].nunique())
    logger.info("  Unique Predictor: %s", list(df["Predictor"].unique()[:10]))
    return df


def _extract_shell_eids_from_coefficients(coeff_df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Katsayi DataFrame'inden bar_eid -> [shell_eids] eslesmesini cikar.

    Iki kaynak:
    1. Shell_EIDs kolonu (virgul ayirmali: "2001,2002,2003")
    2. Predictor kolon degerlerinden parse (Shell_{EID}_Nx)
    """
    import re

    logger = logging.getLogger(__name__)
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

    # Yontem 2: Predictor kolon degerlerinden parse (Shell_{EID}_Nx)
    if not shell_map and "Predictor" in coeff_df.columns:
        shell_eid_pattern = re.compile(r"Shell_(\d+)_N[xXyY]+")
        for bar_eid, grp in coeff_df.groupby("Bar_EID"):
            eids = set()
            for pred_val in grp["Predictor"].dropna().unique():
                m = shell_eid_pattern.match(str(pred_val))
                if m:
                    eids.add(int(m.group(1)))
            if eids:
                shell_map[int(float(bar_eid))] = sorted(eids)

    logger.info("  %d bar element icin shell connectivity bulundu", len(shell_map))
    return shell_map


def predict_from_h5(
    coefficients_excel: str,
    prediction_h5_path: str,
    output_csv: str,
    connectivity: Dict = None,
) -> pd.DataFrame:
    """Output Excel + Prediction H5 -> tahmin CSV."""
    logger = logging.getLogger(__name__)

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
    """Bagimsiz tahmin: predict_from_h5 ile ayni."""
    return predict_from_h5(
        coefficients_excel=coefficients_excel,
        prediction_h5_path=prediction_h5_path,
        output_csv=output_csv,
        connectivity=connectivity,
    )


def predict_from_results(
    correlation_results: List[JointCorrelationResult],
    output_csv: str,
) -> pd.DataFrame:
    """
    In-memory korelasyon sonuclariyla tahmin (Predicted vs Actual mantigi).

    Per-shell predictor'lar kullanir:
      X = [Bar_Axial, Shell_{EID1}_Nx, Shell_{EID1}_Ny, Shell_{EID1}_Nxy, ...]
      predicted = X @ coefficients
    """
    logger = logging.getLogger(__name__)
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
                predicted = float(X[i] @ eq.coefficients)
                col_name = _TARGET_COL_MAP.get(
                    eq.target_name,
                    f"Pred_{eq.target_name.replace(' ', '_')}"
                )
                row[col_name] = predicted

            rows.append(row)

    if not rows:
        logger.warning("Tahmin icin veri bulunamadi")
        return pd.DataFrame()

    df = _pred_order_output(pd.DataFrame(rows))
    df.to_csv(output_csv, index=False)
    logger.info("Tahmin CSV yazildi: %s (%d satir)", output_csv, len(df))
    return df


def predict_from_dataframes(
    coefficients_df: pd.DataFrame,
    bar_forces_df: pd.DataFrame,
    shell_forces_df: pd.DataFrame,
    output_csv: str,
    shell_eid_map: Dict[int, List[int]] = None,
) -> pd.DataFrame:
    """DataFrame formatindaki verilerle tahmin yap."""
    bar_df = _pred_normalize_bar_df(bar_forces_df)
    shell_df = _pred_normalize_shell_df(shell_forces_df)
    return _run_prediction(coefficients_df, bar_df, shell_df, output_csv, shell_eid_map or {})


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
      predicted = predictor @ coefficients

    shell_eid_map: bar_eid -> [shell_eids] (katsayilardan veya BDF'den)
    Her shell'in AYRI NX/NY/NXY degeri kullanilir, ortalama ALINMAZ.
    """
    if shell_eid_map is None:
        shell_eid_map = {}

    logger = logging.getLogger(__name__)
    bar_df = _pred_normalize_bar_df(bar_df)
    shell_df = _pred_normalize_shell_df(shell_df)

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

        # Her target icin katsayi vektorunu hazirla (long format)
        # Predictor sirasi: Bar_Axial, Shell_{EID1}_Nx, Shell_{EID1}_Ny, Shell_{EID1}_Nxy, ...
        expected_predictors = ["Bar_Axial"]
        for seid in sorted(connected_shells):
            expected_predictors.append(f"Shell_{seid}_Nx")
            expected_predictors.append(f"Shell_{seid}_Ny")
            expected_predictors.append(f"Shell_{seid}_Nxy")

        eq_list = []
        for target, target_grp in group.groupby("Target"):
            # Predictor -> Coefficient mapping olustur
            pred_coeff_map = {}
            for _, crow in target_grp.iterrows():
                pred_name = str(crow["Predictor"])
                coeff_val = float(crow.get("Coefficient", 0.0) or 0.0)
                pred_coeff_map[pred_name] = coeff_val

            # Beklenen sirada katsayi vektoru olustur
            coeffs = [pred_coeff_map.get(p, 0.0) for p in expected_predictors]
            eq_list.append((str(target), np.array(coeffs, dtype=float)))

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

            for target, coeffs in eq_list:
                predicted = float(predictor_arr @ coeffs)
                col_name = _TARGET_COL_MAP.get(
                    target, f"Pred_{target.replace(' ', '_')}"
                )
                row[col_name] = predicted

            rows.append(row)

    if not rows:
        logger.warning("Tahmin icin eslesen veri bulunamadi")
        return pd.DataFrame()

    df = _pred_order_output(pd.DataFrame(rows))
    df.to_csv(output_csv, index=False)
    logger.info("Tahmin CSV yazildi: %s (%d satir)", output_csv, len(df))
    return df


# ============================================================
#  Ortak Fonksiyonlar (CLI + GUI)
# ============================================================


def setup_logging(verbose: bool = False) -> None:
    """Logging konfigurasyonu."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_bar_element_set(excel_path: str) -> List[int]:
    """Excel dosyasindan bar element listesini oku."""
    logger = logging.getLogger(__name__)
    logger.info("Excel dosyasi okunuyor: %s", excel_path)

    try:
        df = pd.read_excel(excel_path, sheet_name="Bar Element Set")
    except ValueError:
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        logger.info("Mevcut sheet'ler: %s", sheet_names)

        target_sheet = None
        for name in sheet_names:
            if "bar" in name.lower() and "element" in name.lower():
                target_sheet = name
                break

        if target_sheet is None:
            target_sheet = sheet_names[0]
            logger.warning(
                "'Bar Element Set' bulunamadi, '%s' kullaniliyor", target_sheet
            )

        df = pd.read_excel(excel_path, sheet_name=target_sheet)

    first_col = df.columns[0]
    bar_eids = df[first_col].dropna().astype(int).tolist()
    logger.info("%d bar element okundu", len(bar_eids))
    return bar_eids


def build_bar_forces_dataframe(
    bar_forces: Dict[Tuple[int, int], BarForceResult],
) -> pd.DataFrame:
    """Bar kuvvet sonuclarini DataFrame'e donustur."""
    rows = []
    for (sc_id, eid), result in sorted(bar_forces.items()):
        for t_idx in range(len(result.axial_force)):
            rows.append({
                "Subcase_ID": sc_id,
                "Bar_EID": eid,
                "Time_Step": t_idx,
                "Axial_Force": result.axial_force[t_idx],
                "Shear_1": result.shear_1[t_idx],
                "Shear_2": result.shear_2[t_idx],
                "BM_A1": result.bending_moment_a1[t_idx],
                "BM_A2": result.bending_moment_a2[t_idx],
                "BM_B1": result.bending_moment_b1[t_idx],
                "BM_B2": result.bending_moment_b2[t_idx],
                "Torque": result.torque[t_idx],
            })
    return pd.DataFrame(rows)


def build_shell_forces_dataframe(
    shell_forces: Dict[Tuple[int, int], ShellForceResult],
) -> pd.DataFrame:
    """Shell flux sonuclarini DataFrame'e donustur."""
    rows = []
    for (sc_id, eid), result in sorted(shell_forces.items()):
        for t_idx in range(len(result.membrane_x)):
            rows.append({
                "Subcase_ID": sc_id,
                "Shell_EID": eid,
                "Element_Type": result.elem_type,
                "Time_Step": t_idx,
                "NX": result.membrane_x[t_idx],
                "NY": result.membrane_y[t_idx],
                "NXY": result.membrane_xy[t_idx],
                "MX": result.bending_x[t_idx],
                "MY": result.bending_y[t_idx],
                "MXY": result.bending_xy[t_idx],
                "QX": result.shear_xz[t_idx],
                "QY": result.shear_yz[t_idx],
            })
    return pd.DataFrame(rows)


def _find_element_type_col(df: pd.DataFrame) -> str:
    """H5 DataFrame'inde Element Type kolonunu bul."""
    for col in df.columns:
        if col.lower().replace(" ", "").replace("_", "") == "elementtype":
            return col
    return None


def _find_subcase_col(df: pd.DataFrame) -> str:
    """H5 DataFrame'inde Subcase ID kolonunu bul."""
    for col in df.columns:
        normalized = col.lower().replace(" ", "").replace("_", "")
        if normalized in ("subcaseid", "subcase"):
            return col
    return None


def run_correlation_analysis(
    connectivity: Dict[int, BarElementInfo],
    bar_forces: Dict[Tuple[int, int], BarForceResult],
    shell_forces: Dict[Tuple[int, int], ShellForceResult],
    h5_reader: H5Reader,
    subcase_id: int = None,
) -> Tuple[CorrelationEngine, List[JointCorrelationResult]]:
    """
    Korelasyon analizini calistir.

    Her bar element + element type icin TUM SUBCASE'LER uzerinden
    per-shell kuvvetlerle coklu regresyon denklemi olusturur:
      Target = a0*Bar_Axial
             + a1*Shell_{EID1}_Nx + a2*Shell_{EID1}_Ny + a3*Shell_{EID1}_Nxy
             + a4*Shell_{EID2}_Nx + ...
    """
    logger = logging.getLogger(__name__)
    engine = CorrelationEngine()
    all_results = []

    for bar_eid, info in sorted(connectivity.items()):
        logger.info("Bar %d icin korelasyon hesaplaniyor...", bar_eid)

        h5_data = h5_reader.get_joint_loads_for_bar(bar_eid)
        if h5_data is None or h5_data.empty:
            logger.warning("Bar %d icin H5 verisi bulunamadi", bar_eid)
            continue

        sc_keys = [k for k in bar_forces.keys() if k[1] == bar_eid]
        if not sc_keys:
            logger.warning("Bar %d icin OP2 kuvvet verisi yok", bar_eid)
            continue

        # Bagli shell EID listesi
        shell_eids = list(info.connected_quads.keys()) + list(info.connected_trias.keys())

        # ---- Per-shell veri toplama ----
        # Once bar verisi olan subcase'leri belirle
        valid_sc_keys = []
        for sc_id_key, _ in sc_keys:
            if subcase_id is not None and sc_id_key != subcase_id:
                continue
            bar_result = bar_forces.get((sc_id_key, bar_eid))
            if bar_result is not None:
                valid_sc_keys.append(sc_id_key)

        if not valid_sc_keys:
            logger.warning("Bar %d: Hicbir subcase icin bar verisi yok", bar_eid)
            continue

        # Tum subcase'lerde verisi olan shell'leri bul
        shells_with_full_data = []
        for seid in shell_eids:
            has_all = all(
                shell_forces.get((sc_id, seid)) is not None
                for sc_id in valid_sc_keys
            )
            if has_all:
                shells_with_full_data.append(seid)

        if not shells_with_full_data:
            logger.warning("Bar %d: Hicbir shell tum subcase'lerde veri yok", bar_eid)
            continue

        # Per-shell veri toplama
        collected_sc_ids = []
        collected_axial = []
        # {shell_eid: {"nx": [], "ny": [], "nxy": []}}
        collected_shell = {
            seid: {"nx": [], "ny": [], "nxy": []}
            for seid in shells_with_full_data
        }

        for sc_id in valid_sc_keys:
            bar_result = bar_forces.get((sc_id, bar_eid))
            axial_mean = np.mean(bar_result.axial_force)

            collected_sc_ids.append(sc_id)
            collected_axial.append(axial_mean)

            for seid in shells_with_full_data:
                sf = shell_forces.get((sc_id, seid))
                collected_shell[seid]["nx"].append(np.mean(sf.membrane_x))
                collected_shell[seid]["ny"].append(np.mean(sf.membrane_y))
                collected_shell[seid]["nxy"].append(np.mean(sf.membrane_xy))

        n_shells = len(shells_with_full_data)
        logger.info(
            "  Bar %d: %d subcase, %d shell (per-shell predictor)",
            bar_eid, len(collected_sc_ids), n_shells,
        )

        # OP2 verilerini subcase_id'ye gore dict'e koy (H5 eslestirme icin)
        op2_by_sc = {}
        for i, sc_id in enumerate(collected_sc_ids):
            sc_data = {"axial": collected_axial[i], "shells": {}}
            for seid in shells_with_full_data:
                sc_data["shells"][seid] = {
                    "nx": collected_shell[seid]["nx"][i],
                    "ny": collected_shell[seid]["ny"][i],
                    "nxy": collected_shell[seid]["nxy"][i],
                }
            op2_by_sc[sc_id] = sc_data

        # Element Type'a gore grupla
        et_col = _find_element_type_col(h5_data)
        if et_col is not None:
            element_types = sorted(h5_data[et_col].unique())
        else:
            element_types = [0]

        # H5'teki Subcase ID kolonunu bul
        sc_col = _find_subcase_col(h5_data)

        for et in element_types:
            if et_col is not None:
                h5_subset = h5_data[h5_data[et_col] == et].copy()
            else:
                h5_subset = h5_data

            if h5_subset.empty:
                continue

            # --- Subcase ID eslestirmesi ---
            if sc_col is not None:
                matched_axial = []
                matched_shell = {
                    seid: {"nx": [], "ny": [], "nxy": []}
                    for seid in shells_with_full_data
                }
                matched_h5_indices = []
                matched_sc_ids = []

                for idx, row in h5_subset.iterrows():
                    h5_sc = int(row[sc_col])
                    if h5_sc in op2_by_sc:
                        sc_data = op2_by_sc[h5_sc]
                        matched_axial.append(sc_data["axial"])
                        for seid in shells_with_full_data:
                            matched_shell[seid]["nx"].append(sc_data["shells"][seid]["nx"])
                            matched_shell[seid]["ny"].append(sc_data["shells"][seid]["ny"])
                            matched_shell[seid]["nxy"].append(sc_data["shells"][seid]["nxy"])
                        matched_h5_indices.append(idx)
                        matched_sc_ids.append(h5_sc)

                n_matched = len(matched_h5_indices)
                n_h5_total = len(h5_subset)

                logger.info(
                    "  Bar %d, ET %s: %d/%d H5 satir OP2 ile eslesti",
                    bar_eid, et, n_matched, n_h5_total,
                )

                if not matched_h5_indices:
                    logger.warning(
                        "  Bar %d, ET %s: Hicbir subcase eslesmiyor! "
                        "OP2 SC: %s, H5 SC: %s",
                        bar_eid, et,
                        sorted(op2_by_sc.keys()),
                        sorted(h5_subset[sc_col].unique().tolist()),
                    )
                    continue

                h5_matched = h5_subset.loc[matched_h5_indices].reset_index(drop=True)
                pred_axial = np.array(matched_axial)
                shell_forces_per_eid = {
                    seid: {
                        "nx": np.array(matched_shell[seid]["nx"]),
                        "ny": np.array(matched_shell[seid]["ny"]),
                        "nxy": np.array(matched_shell[seid]["nxy"]),
                    }
                    for seid in shells_with_full_data
                }
            else:
                # Subcase kolonu yoksa sirali eslestirme
                logger.warning(
                    "  Bar %d: H5'te Subcase ID kolonu bulunamadi, "
                    "sirali eslestirme yapiliyor",
                    bar_eid,
                )
                h5_matched = h5_subset
                pred_axial = np.array(collected_axial)
                shell_forces_per_eid = {
                    seid: {
                        "nx": np.array(collected_shell[seid]["nx"]),
                        "ny": np.array(collected_shell[seid]["ny"]),
                        "nxy": np.array(collected_shell[seid]["nxy"]),
                    }
                    for seid in shells_with_full_data
                }
                matched_sc_ids = list(collected_sc_ids)

            logger.info("  Bar %d, ElementType %s, %d eslesen veri noktasi",
                        bar_eid, et, len(pred_axial))

            result = engine.compute_joint_correlation(
                bar_eid=bar_eid,
                subcase_id=0,
                element_type=int(et),
                bar_axial=pred_axial,
                shell_forces_per_eid=shell_forces_per_eid,
                h5_data=h5_matched,
                n_shells=n_shells,
            )
            result.matched_subcases = matched_sc_ids
            all_results.append(result)

    return engine, all_results


def run_per_shell_correlation_analysis(
    connectivity: Dict[int, BarElementInfo],
    bar_forces: Dict[Tuple[int, int], BarForceResult],
    shell_forces: Dict[Tuple[int, int], ShellForceResult],
    h5_reader: H5Reader,
    subcase_id: int = None,
) -> List[JointCorrelationResult]:
    """
    Her bagli shell element icin AYRI AYRI korelasyon analizi calistir.

    Her shell element icin bireysel regresyon (tek shell predictor ile):
      Target = a1*Bar_Axial + a2*Shell_{EID}_Nx + a3*Shell_{EID}_Ny + a4*Shell_{EID}_Nxy + b
    """
    logger = logging.getLogger(__name__)
    per_shell_engine = CorrelationEngine()
    per_shell_results = []

    for bar_eid, info in sorted(connectivity.items()):
        h5_data = h5_reader.get_joint_loads_for_bar(bar_eid)
        if h5_data is None or h5_data.empty:
            continue

        sc_keys = [k for k in bar_forces.keys() if k[1] == bar_eid]
        if not sc_keys:
            continue

        shell_info = {}
        for qeid in info.connected_quads:
            shell_info[qeid] = "CQUAD4"
        for teid in info.connected_trias:
            shell_info[teid] = "CTRIA3"

        collected_sc_ids = []
        collected_axial = []

        for sc_id_key, _ in sc_keys:
            if subcase_id is not None and sc_id_key != subcase_id:
                continue
            bar_result = bar_forces.get((sc_id_key, bar_eid))
            if bar_result is None:
                continue
            collected_sc_ids.append(sc_id_key)
            collected_axial.append(np.mean(bar_result.axial_force))

        if not collected_sc_ids:
            continue

        et_col = _find_element_type_col(h5_data)
        sc_col = _find_subcase_col(h5_data)
        element_types = sorted(h5_data[et_col].unique()) if et_col else [0]

        for shell_eid, shell_type_str in shell_info.items():
            per_shell_sc_ids = []
            per_shell_axial = []
            per_shell_nx = []
            per_shell_ny = []
            per_shell_nxy = []

            for i, sc_id in enumerate(collected_sc_ids):
                sf = shell_forces.get((sc_id, shell_eid))
                if sf is None:
                    continue
                per_shell_sc_ids.append(sc_id)
                per_shell_axial.append(collected_axial[i])
                per_shell_nx.append(np.mean(sf.membrane_x))
                per_shell_ny.append(np.mean(sf.membrane_y))
                per_shell_nxy.append(np.mean(sf.membrane_xy))

            if not per_shell_sc_ids:
                continue

            op2_by_sc = {}
            for i, sc_id in enumerate(per_shell_sc_ids):
                op2_by_sc[sc_id] = {
                    "axial": per_shell_axial[i],
                    "nx": per_shell_nx[i],
                    "ny": per_shell_ny[i],
                    "nxy": per_shell_nxy[i],
                }

            n_shells_total = len(shell_info)

            for et in element_types:
                h5_subset = h5_data[h5_data[et_col] == et].copy() if et_col else h5_data
                if h5_subset.empty:
                    continue

                if sc_col is not None:
                    matched_axial = []
                    matched_nx = []
                    matched_ny = []
                    matched_nxy = []
                    matched_h5_indices = []
                    matched_sc_ids = []

                    for idx, row in h5_subset.iterrows():
                        h5_sc = int(row[sc_col])
                        if h5_sc in op2_by_sc:
                            sc_data = op2_by_sc[h5_sc]
                            matched_axial.append(sc_data["axial"])
                            matched_nx.append(sc_data["nx"])
                            matched_ny.append(sc_data["ny"])
                            matched_nxy.append(sc_data["nxy"])
                            matched_h5_indices.append(idx)
                            matched_sc_ids.append(h5_sc)

                    if not matched_h5_indices:
                        continue

                    h5_matched = h5_subset.loc[matched_h5_indices].reset_index(drop=True)
                    pred_axial = np.array(matched_axial)
                    single_shell_forces = {
                        shell_eid: {
                            "nx": np.array(matched_nx),
                            "ny": np.array(matched_ny),
                            "nxy": np.array(matched_nxy),
                        }
                    }
                else:
                    h5_matched = h5_subset
                    pred_axial = np.array(per_shell_axial)
                    single_shell_forces = {
                        shell_eid: {
                            "nx": np.array(per_shell_nx),
                            "ny": np.array(per_shell_ny),
                            "nxy": np.array(per_shell_nxy),
                        }
                    }
                    matched_sc_ids = list(per_shell_sc_ids)

                result = per_shell_engine.compute_joint_correlation(
                    bar_eid=bar_eid,
                    subcase_id=0,
                    element_type=int(et),
                    bar_axial=pred_axial,
                    shell_forces_per_eid=single_shell_forces,
                    h5_data=h5_matched,
                    n_shells=n_shells_total,
                )
                result.shell_eid = shell_eid
                result.shell_type = shell_type_str
                result.matched_subcases = matched_sc_ids
                per_shell_results.append(result)

                logger.info(
                    "  Bar %d, Shell %d (%s), ET %s: %d denklem",
                    bar_eid, shell_eid, shell_type_str, et, len(result.equations),
                )

    logger.info("Per-shell korelasyon: %d sonuc uretildi", len(per_shell_results))
    return per_shell_results


# ============================================================
#  GUI Bilesenleri
# ============================================================


class QueueHandler(logging.Handler):
    """Log mesajlarini queue'ya yonlendirir (thread-safe GUI guncellemesi icin)."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class FileSelector(ttk.Frame):
    """Dosya secimi icin label + entry + browse butonu bileseni."""

    def __init__(
        self,
        parent,
        label_text: str,
        filetypes: list,
        is_save: bool = False,
        default_ext: str = "",
        multiple: bool = False,
    ):
        super().__init__(parent)
        self.filetypes = filetypes
        self.is_save = is_save
        self.default_ext = default_ext
        self.multiple = multiple

        self.columnconfigure(1, weight=1)

        self.label = ttk.Label(self, text=label_text, width=14, anchor="w")
        self.label.grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.var)
        self.entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))

        self.btn = ttk.Button(self, text="Gozat...", width=9, command=self._browse)
        self.btn.grid(row=0, column=2)

    def _browse(self):
        if self.is_save:
            path = filedialog.asksaveasfilename(
                defaultextension=self.default_ext,
                filetypes=self.filetypes,
            )
            if path:
                self.var.set(path)
        elif self.multiple:
            paths = filedialog.askopenfilenames(filetypes=self.filetypes)
            if paths:
                self.var.set("; ".join(paths))
        else:
            path = filedialog.askopenfilename(filetypes=self.filetypes)
            if path:
                self.var.set(path)

    def get(self) -> str:
        return self.var.get().strip()

    def get_multiple(self) -> List[str]:
        """Birden fazla dosya yolu dondur (';' ile ayrilmis)."""
        raw = self.var.get().strip()
        if not raw:
            return []
        return [p.strip() for p in raw.split(";") if p.strip()]

    def set(self, value: str):
        self.var.set(value)


class Application(tk.Tk):
    """Ana uygulama penceresi."""

    WINDOW_TITLE = "Joint Load Extraction - Force Relationship Tool"
    MIN_WIDTH = 750
    MIN_HEIGHT = 620

    def __init__(self):
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.geometry("800x680")

        self.log_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread = None
        self._is_running = False

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self):
        """Arayuz bilesenlerini olustur."""
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)

        title_lbl = ttk.Label(
            main_frame,
            text="Joint Load Extraction - Force Relationship",
            font=("Segoe UI", 14, "bold"),
        )
        title_lbl.grid(row=0, column=0, pady=(0, 8), sticky="w")

        # === Dosya Secimi ===
        file_frame = ttk.LabelFrame(main_frame, text=" Dosya Secimi ", padding=10)
        file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(0, weight=1)

        self.bdf_selector = FileSelector(
            file_frame,
            "BDF Dosyasi:",
            [("Nastran BDF", "*.bdf *.dat *.nas *.BDF *.DAT"), ("Tum Dosyalar", "*.*")],
        )
        self.bdf_selector.grid(row=0, column=0, sticky="ew", pady=2)

        self.op2_selector = FileSelector(
            file_frame,
            "OP2 Dosyasi:",
            [("Nastran OP2", "*.op2 *.OP2"), ("Tum Dosyalar", "*.*")],
            multiple=True,
        )
        self.op2_selector.grid(row=1, column=0, sticky="ew", pady=2)

        self.h5_selector = FileSelector(
            file_frame,
            "H5 Dosyasi:",
            [("HDF5", "*.h5 *.hdf5 *.H5 *.HDF5"), ("Tum Dosyalar", "*.*")],
        )
        self.h5_selector.grid(row=2, column=0, sticky="ew", pady=2)

        self.excel_selector = FileSelector(
            file_frame,
            "Excel Dosyasi:",
            [("Excel", "*.xlsx *.xls *.XLSX"), ("Tum Dosyalar", "*.*")],
        )
        self.excel_selector.grid(row=3, column=0, sticky="ew", pady=2)

        self.pred_h5_selector = FileSelector(
            file_frame,
            "Prediction H5:",
            [("HDF5", "*.h5 *.hdf5 *.H5 *.HDF5"), ("Tum Dosyalar", "*.*")],
        )
        self.pred_h5_selector.grid(row=4, column=0, sticky="ew", pady=2)

        self.output_selector = FileSelector(
            file_frame,
            "Cikti Dosyasi:",
            [("Excel", "*.xlsx")],
            is_save=True,
            default_ext=".xlsx",
        )
        self.output_selector.grid(row=5, column=0, sticky="ew", pady=2)
        self.output_selector.set("joint_load_correlation_output.xlsx")

        # === Ayarlar ===
        settings_frame = ttk.LabelFrame(main_frame, text=" Ayarlar ", padding=10)
        settings_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)

        ttk.Label(settings_frame, text="Subcase ID:", anchor="w").grid(
            row=0, column=0, padx=(0, 5), sticky="w"
        )
        self.subcase_var = tk.StringVar(value="")
        self.subcase_entry = ttk.Entry(
            settings_frame, textvariable=self.subcase_var, width=12
        )
        self.subcase_entry.grid(row=0, column=1, sticky="w", padx=(0, 20))

        ttk.Label(
            settings_frame, text="(Bos = tum subcaseler)", foreground="gray"
        ).grid(row=0, column=2, sticky="w")

        ttk.Label(settings_frame, text="H5 Grup Yolu:", anchor="w").grid(
            row=1, column=0, padx=(0, 5), pady=(5, 0), sticky="w"
        )
        self.h5_group_var = tk.StringVar(value="")
        self.h5_group_entry = ttk.Entry(
            settings_frame, textvariable=self.h5_group_var
        )
        self.h5_group_entry.grid(
            row=1, column=1, columnspan=3, sticky="ew", pady=(5, 0)
        )

        ttk.Label(
            settings_frame,
            text='(Bos = otomatik ara, ornek: "JOINT_LOADS_CAP/table")',
            foreground="gray",
        ).grid(row=2, column=1, columnspan=3, sticky="w")

        self.verbose_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            settings_frame, text="Detayli cikti (verbose)", variable=self.verbose_var
        ).grid(row=3, column=0, columnspan=2, pady=(5, 0), sticky="w")

        # === Butonlar ===
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        btn_frame.columnconfigure(0, weight=1)

        self.run_btn = ttk.Button(
            btn_frame,
            text="  Calistir  ",
            command=self._on_run,
            style="Accent.TButton",
        )
        self.run_btn.grid(row=0, column=0, pady=2)

        # === Progress Bar ===
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate", length=400)
        self.progress.grid(row=4, column=0, sticky="ew", pady=(0, 4))

        self.status_var = tk.StringVar(value="Hazir.")
        self.status_label = ttk.Label(
            main_frame, textvariable=self.status_var, foreground="gray"
        )
        self.status_label.grid(row=5, column=0, sticky="w", pady=(0, 4))

        # === Log ===
        log_frame = ttk.LabelFrame(main_frame, text=" Log ", padding=4)
        log_frame.grid(row=6, column=0, sticky="nsew", pady=(0, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        self.log_text.tag_config("INFO", foreground="#4fc1ff")
        self.log_text.tag_config("WARNING", foreground="#cca700")
        self.log_text.tag_config("ERROR", foreground="#f44747")
        self.log_text.tag_config("SUCCESS", foreground="#89d185")

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=7, column=0, sticky="ew", pady=(4, 0))

        ttk.Button(
            bottom_frame, text="Logu Temizle", command=self._clear_log
        ).pack(side="right")

    def _validate_inputs(self) -> bool:
        single_checks = [
            (self.bdf_selector.get(), "BDF dosyasi"),
            (self.h5_selector.get(), "H5 dosyasi"),
            (self.excel_selector.get(), "Excel dosyasi"),
            (self.output_selector.get(), "Cikti dosyasi"),
        ]

        for path, name in single_checks:
            if not path:
                messagebox.showwarning("Eksik Alan", f"{name} secilmedi!")
                return False

        # Tekli dosya kontrol (son eleman cikti, var olmasina gerek yok)
        for path, name in single_checks[:-1]:
            if not Path(path).exists():
                messagebox.showerror("Dosya Bulunamadi", f"{name} bulunamadi:\n{path}")
                return False

        # OP2 dosyalari (toplu secim destegi)
        op2_paths = self.op2_selector.get_multiple()
        if not op2_paths:
            messagebox.showwarning("Eksik Alan", "OP2 dosyasi secilmedi!")
            return False

        for op2_path in op2_paths:
            if not Path(op2_path).exists():
                messagebox.showerror(
                    "Dosya Bulunamadi", f"OP2 dosyasi bulunamadi:\n{op2_path}"
                )
                return False

        return True

    def _on_run(self):
        if self._is_running:
            messagebox.showinfo("Bilgi", "Islem devam ediyor, lutfen bekleyin.")
            return

        if not self._validate_inputs():
            return

        self._is_running = True
        self.run_btn.configure(state="disabled")
        self.progress.start(15)
        self.status_var.set("Islem baslatiliyor...")
        self._clear_log()

        self.worker_thread = threading.Thread(target=self._run_analysis, daemon=True)
        self.worker_thread.start()

    def _run_analysis(self):
        """Analizi ayri thread'de calistir."""
        log_level = logging.DEBUG if self.verbose_var.get() else logging.INFO
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        root_logger.addHandler(queue_handler)

        logger = logging.getLogger(__name__)

        try:
            bdf_path = self.bdf_selector.get()
            op2_paths = self.op2_selector.get_multiple()
            h5_path = self.h5_selector.get()
            excel_path = self.excel_selector.get()
            output_path = self.output_selector.get()
            subcase_str = self.subcase_var.get().strip()
            subcase_id = int(subcase_str) if subcase_str else None
            h5_group = self.h5_group_var.get().strip() or None
            pred_h5_path = self.pred_h5_selector.get() or None

            # ADIM 1
            self._update_status("Adim 1/7: Excel okunuyor...")
            logger.info("=" * 50)
            logger.info("ADIM 1: Bar element listesi okunuyor...")
            bar_element_ids = read_bar_element_set(excel_path)
            logger.info("  %d bar element okundu", len(bar_element_ids))

            # ADIM 2
            self._update_status("Adim 2/7: BDF parse ediliyor...")
            logger.info("ADIM 2: BDF parse ediliyor...")
            bdf_parser = BDFParser(bdf_path)
            bdf_parser.parse()
            connectivity = bdf_parser.get_bar_connectivity(bar_element_ids)
            logger.info("  %d bar element baglantisi bulundu", len(connectivity))

            all_shell_eids = set()
            for info in connectivity.values():
                all_shell_eids.update(info.connected_quads.keys())
                all_shell_eids.update(info.connected_trias.keys())
            logger.info("  %d bagli shell element", len(all_shell_eids))

            # ADIM 3
            self._update_status("Adim 3/7: OP2 okunuyor (%d dosya)..." % len(op2_paths))
            logger.info("ADIM 3: OP2 okunuyor (%d dosya)...", len(op2_paths))

            bar_forces, shell_forces, op2_subcases = OP2Reader.read_multiple(
                op2_paths=op2_paths,
                bar_eids=list(connectivity.keys()),
                shell_eids=list(all_shell_eids),
                subcase_id=subcase_id,
            )
            logger.info(
                "  %d bar, %d shell kuvvet verisi (%d OP2 dosyasi)",
                len(bar_forces), len(shell_forces), len(op2_paths),
            )

            # ADIM 4
            self._update_status("Adim 4/7: H5 okunuyor...")
            logger.info("ADIM 4: H5 dosyasi okunuyor...")
            h5_reader = H5Reader(h5_path)
            h5_reader.read(group_path=h5_group)
            logger.info("  %d satir JOINT_LOADS_CAP", len(h5_reader.joint_load_cap))

            # ADIM 5
            self._update_status("Adim 5/7: Korelasyon hesaplaniyor...")
            logger.info("ADIM 5: Korelasyon analizi...")
            engine, correlation_results = run_correlation_analysis(
                connectivity, bar_forces, shell_forces, h5_reader, subcase_id
            )
            correlation_summary = engine.get_summary_dataframe()
            logger.info("  %d korelasyon sonucu", len(correlation_results))

            # ADIM 5b: Per-shell korelasyon
            logger.info("ADIM 5b: Per-shell korelasyon analizi...")
            per_shell_results = run_per_shell_correlation_analysis(
                connectivity, bar_forces, shell_forces, h5_reader, subcase_id
            )
            logger.info("  %d per-shell korelasyon sonucu", len(per_shell_results))

            # ADIM 6
            self._update_status("Adim 6/7: Rapor olusturuluyor...")
            logger.info("ADIM 6: Excel raporu olusturuluyor...")

            bar_forces_df = build_bar_forces_dataframe(bar_forces)
            shell_forces_df = build_shell_forces_dataframe(shell_forces)

            reporter = ReportGenerator(output_path)
            reporter.generate(
                bar_element_ids=bar_element_ids,
                connectivity=connectivity,
                bar_forces_df=bar_forces_df,
                shell_forces_df=shell_forces_df,
                h5_joint_loads=h5_reader.joint_load_cap,
                correlation_results=correlation_results,
                correlation_summary_df=correlation_summary,
                per_shell_results=per_shell_results,
            )

            # ADIM 7: Tahmin CSV
            self._update_status("Adim 7/7: Tahmin CSV olusturuluyor...")
            logger.info("ADIM 7: Tahmin CSV olusturuluyor...")

            csv_path = str(Path(output_path).with_suffix(".csv"))

            if pred_h5_path and Path(pred_h5_path).exists():
                logger.info("  Prediction H5: %s", pred_h5_path)
                logger.info("  Katsayilar:    %s (Correlation Summary sheet)", output_path)
                pred_df = predict_from_h5(
                    coefficients_excel=output_path,
                    prediction_h5_path=pred_h5_path,
                    output_csv=csv_path,
                    connectivity=connectivity,
                )
            else:
                pred_df = predict_from_results(
                    correlation_results=correlation_results,
                    output_csv=csv_path,
                )
            logger.info("  %d tahmin satiri yazildi: %s", len(pred_df), csv_path)

            logger.info("=" * 50)
            logger.info("TAMAMLANDI! Cikti: %s", output_path)
            logger.info("Tahmin CSV: %s", csv_path)

            self._update_status(f"Tamamlandi! -> {output_path}")
            self.after(
                0,
                lambda: messagebox.showinfo(
                    "Basarili",
                    f"Analiz tamamlandi!\n\n"
                    f"OP2 dosya: {len(op2_paths)}\n"
                    f"Bar element: {len(bar_element_ids)}\n"
                    f"Baglanti: {len(connectivity)}\n"
                    f"Korelasyon: {len(correlation_results)}\n"
                    f"Tahmin: {len(pred_df)} satir\n\n"
                    f"Excel: {output_path}\n"
                    f"CSV: {csv_path}",
                ),
            )

        except Exception as e:
            logger.error("HATA: %s", str(e), exc_info=True)
            self._update_status(f"Hata: {e}")
            self.after(
                0,
                lambda: messagebox.showerror(
                    "Hata", f"Islem sirasinda hata olustu:\n\n{e}"
                ),
            )
        finally:
            self._is_running = False
            self.after(0, self._on_analysis_done)

    def _on_analysis_done(self):
        self.progress.stop()
        self.run_btn.configure(state="normal")

    def _update_status(self, text: str):
        self.after(0, lambda: self.status_var.set(text))

    def _poll_log_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break

            self.log_text.configure(state="normal")

            tag = None
            if "[INFO]" in msg:
                tag = "INFO"
            elif "[WARNING]" in msg:
                tag = "WARNING"
            elif "[ERROR]" in msg:
                tag = "ERROR"
            elif "TAMAMLANDI" in msg:
                tag = "SUCCESS"

            if tag:
                self.log_text.insert("end", msg + "\n", tag)
            else:
                self.log_text.insert("end", msg + "\n")

            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.after(100, self._poll_log_queue)

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


# ============================================================
#  CLI Modu
# ============================================================


def run_cli(args: argparse.Namespace) -> None:
    """Komut satiri modunda calistir."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Joint Load Extraction Force Relationship Tool")
    logger.info("=" * 60)

    for path_arg, name in [
        (args.bdf, "BDF"),
        (args.h5, "H5"),
        (args.excel, "Excel"),
    ]:
        p = Path(path_arg)
        if not p.exists():
            logger.error("%s dosyasi bulunamadi: %s", name, path_arg)
            sys.exit(1)
        logger.info("%s: %s", name, p.resolve())

    # OP2 dosyalarini kontrol et
    op2_paths = args.op2
    for op2_path in op2_paths:
        p = Path(op2_path)
        if not p.exists():
            logger.error("OP2 dosyasi bulunamadi: %s", op2_path)
            sys.exit(1)
        logger.info("OP2: %s", p.resolve())
    logger.info("Toplam %d OP2 dosyasi", len(op2_paths))

    # ADIM 1
    logger.info("-" * 40)
    logger.info("ADIM 1: Bar element listesi okunuyor...")
    bar_element_ids = read_bar_element_set(args.excel)
    logger.info("  %d bar element okundu", len(bar_element_ids))

    # ADIM 2
    logger.info("-" * 40)
    logger.info("ADIM 2: BDF parse ediliyor...")
    bdf_parser = BDFParser(args.bdf)
    bdf_parser.parse()

    connectivity = bdf_parser.get_bar_connectivity(bar_element_ids)
    logger.info("  %d bar element icin baglanti bulundu", len(connectivity))

    all_shell_eids = set()
    for info in connectivity.values():
        all_shell_eids.update(info.connected_quads.keys())
        all_shell_eids.update(info.connected_trias.keys())
    logger.info("  Toplam %d bagli shell element", len(all_shell_eids))

    # ADIM 3
    logger.info("-" * 40)
    logger.info("ADIM 3: OP2 okunuyor (%d dosya)...", len(op2_paths))

    bar_forces, shell_forces, op2_subcases = OP2Reader.read_multiple(
        op2_paths=op2_paths,
        bar_eids=list(connectivity.keys()),
        shell_eids=list(all_shell_eids),
        subcase_id=args.subcase,
    )
    logger.info("  %d bar element kuvvet verisi okundu", len(bar_forces))
    logger.info("  %d shell element flux verisi okundu", len(shell_forces))

    # ADIM 4
    logger.info("-" * 40)
    logger.info("ADIM 4: H5 dosyasi okunuyor...")
    h5_reader = H5Reader(args.h5)
    h5_reader.read(group_path=args.h5_group)
    logger.info("  JOINT_LOADS_CAP: %d satir", len(h5_reader.joint_load_cap))

    # ADIM 5
    logger.info("-" * 40)
    logger.info("ADIM 5: Korelasyon analizi yapiliyor...")
    engine, correlation_results = run_correlation_analysis(
        connectivity=connectivity,
        bar_forces=bar_forces,
        shell_forces=shell_forces,
        h5_reader=h5_reader,
        subcase_id=args.subcase,
    )
    correlation_summary = engine.get_summary_dataframe()
    logger.info("  %d korelasyon sonucu hesaplandi", len(correlation_results))

    # ADIM 5b: Per-shell korelasyon
    logger.info("-" * 40)
    logger.info("ADIM 5b: Per-shell korelasyon analizi yapiliyor...")
    per_shell_results = run_per_shell_correlation_analysis(
        connectivity=connectivity,
        bar_forces=bar_forces,
        shell_forces=shell_forces,
        h5_reader=h5_reader,
        subcase_id=args.subcase,
    )
    logger.info("  %d per-shell korelasyon sonucu", len(per_shell_results))

    # ADIM 6
    logger.info("-" * 40)
    logger.info("ADIM 6: Excel raporu olusturuluyor...")

    bar_forces_df = build_bar_forces_dataframe(bar_forces)
    shell_forces_df = build_shell_forces_dataframe(shell_forces)

    reporter = ReportGenerator(args.output)
    output_path = reporter.generate(
        bar_element_ids=bar_element_ids,
        connectivity=connectivity,
        bar_forces_df=bar_forces_df,
        shell_forces_df=shell_forces_df,
        h5_joint_loads=h5_reader.joint_load_cap,
        correlation_results=correlation_results,
        correlation_summary_df=correlation_summary,
        per_shell_results=per_shell_results,
    )

    # ADIM 7: Tahmin CSV
    logger.info("-" * 40)
    logger.info("ADIM 7: Tahmin CSV olusturuluyor...")

    csv_path = str(Path(args.output).with_suffix(".csv"))

    prediction_h5 = getattr(args, "prediction_h5", None)
    if prediction_h5 and Path(prediction_h5).exists():
        logger.info("  Prediction H5: %s", prediction_h5)
        logger.info("  Katsayilar:    %s (Correlation Summary sheet)", output_path)
        pred_df = predict_from_h5(
            coefficients_excel=output_path,
            prediction_h5_path=prediction_h5,
            output_csv=csv_path,
            connectivity=connectivity,
        )
    else:
        pred_df = predict_from_results(
            correlation_results=correlation_results,
            output_csv=csv_path,
        )
    logger.info("  %d tahmin satiri yazildi: %s", len(pred_df), csv_path)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("TAMAMLANDI!")
    logger.info("Cikti dosyasi: %s", output_path)
    logger.info("Tahmin CSV:    %s", csv_path)
    logger.info("Sure: %.1f saniye", elapsed)
    logger.info("=" * 60)

    print(f"\nSonuc Ozeti:")
    print(f"  Bar element sayisi:    {len(bar_element_ids)}")
    print(f"  Baglanti bulunan:      {len(connectivity)}")
    print(f"  Bagli shell element:   {len(all_shell_eids)}")
    print(f"  OP2 dosya sayisi:      {len(op2_paths)}")
    print(f"  OP2 bar kuvvet:        {len(bar_forces)}")
    print(f"  OP2 shell flux:        {len(shell_forces)}")
    print(f"  H5 Joint Load satir:   {len(h5_reader.joint_load_cap)}")
    print(f"  Korelasyon sonucu:     {len(correlation_results)}")
    print(f"  Tahmin satiri:         {len(pred_df)}")
    print(f"\n  Excel cikti: {output_path}")
    print(f"  Tahmin CSV:  {csv_path}")


# ============================================================
#  Giris Noktasi
# ============================================================


def main():
    """Ana giris noktasi. Parametresiz = GUI, --cli = komut satiri."""
    parser = argparse.ArgumentParser(
        description="Joint Load Extraction Force Relationship Tool (Standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanim modlari:
  python app_standalone.py                  # GUI baslatir (varsayilan)
  python app_standalone.py --cli --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx
  python app_standalone.py --cli --bdf model.bdf --op2 file1.op2 file2.op2 file3.op2 --h5 joint_loads.h5 --excel input.xlsx
  python app_standalone.py --cli --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx --output results.xlsx --subcase 1
        """,
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Komut satiri modunda calistir (GUI yerine)",
    )
    parser.add_argument("--bdf", default=None, help="Nastran BDF dosyasi yolu")
    parser.add_argument("--op2", default=None, nargs="+", help="Nastran OP2 dosyasi yolu (birden fazla dosya verilebilir)")
    parser.add_argument("--h5", default=None, help="Joint Load Extraction H5 dosyasi yolu")
    parser.add_argument("--excel", default=None, help="Bar Element Set iceren Excel dosyasi yolu")
    parser.add_argument(
        "--output",
        default="joint_load_correlation_output.xlsx",
        help="Cikti Excel dosyasi yolu (varsayilan: joint_load_correlation_output.xlsx)",
    )
    parser.add_argument(
        "--subcase", type=int, default=None, help="Belirli bir subcase ID"
    )
    parser.add_argument(
        "--h5-group", default=None, help="H5 dosyasindaki JOINT_LOADS_CAP tablosunun yolu (ornek: 'JOINT_LOADS_CAP/table')"
    )
    parser.add_argument(
        "--prediction-h5", default=None,
        help="Tahmin icin ayri H5 dosyasi (bar element combined + shell force combined)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detayli cikti")

    args = parser.parse_args()

    if args.cli:
        missing = []
        for param in ["bdf", "op2", "h5", "excel"]:
            if getattr(args, param) is None:
                missing.append(f"--{param}")
        if missing:
            parser.error(
                f"CLI modunda su parametreler zorunludur: {', '.join(missing)}"
            )
        run_cli(args)
    else:
        app = Application()
        app.mainloop()


if __name__ == "__main__":
    main()
