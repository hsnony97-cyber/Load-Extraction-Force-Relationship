"""
Force H5 Reader Module

Main H5 dosyasindan bar elementler icin axial force,
shell elementler icin membrane fluxlarini (NX, NY, NXY) okur.

H5 dosyasi prediction H5 ile ayni formatta:
  - ELFORCE_BAR_COMBINED/table:   Element_ID, Subcase_ID, AF
  - ELFORCE_SHELL_COMBINED/table: Element_ID, Subcase_ID, MX(=NX), MY(=NY), MXY(=NXY)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# H5 tablo yollari (prediction H5 ile ayni format)
BAR_TABLE_PATH = "ELFORCE_BAR_COMBINED/table"
SHELL_TABLE_PATH = "ELFORCE_SHELL_COMBINED/table"


@dataclass
class BarForceResult:
    """Bir bar element icin kuvvet sonuclari."""
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
    """Bir shell element icin flux sonuclari."""
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
    """Main H5 dosyasindan element kuvvetlerini okur.

    H5 dosyasi prediction H5 ile ayni formatta olmalidir:
      - ELFORCE_BAR_COMBINED/table:   Element_ID, Subcase_ID, AF
      - ELFORCE_SHELL_COMBINED/table: Element_ID, Subcase_ID, MX, MY, MXY
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._bar_df: pd.DataFrame = pd.DataFrame()
        self._shell_df: pd.DataFrame = pd.DataFrame()
        self._available_subcases: List[int] = []

    def read(self) -> None:
        """Main H5 dosyasini oku."""
        logger.info("Main H5 dosyasi okunuyor: %s", self.h5_path)

        with h5py.File(self.h5_path, "r") as f:
            # --- Bar element combined ---
            if BAR_TABLE_PATH in f:
                self._bar_df = self._read_h5_table(f, BAR_TABLE_PATH)
                self._normalize_bar_columns()
                logger.info(
                    "Bar tablosu okundu: %s (%d satir, kolonlar: %s)",
                    BAR_TABLE_PATH, len(self._bar_df), list(self._bar_df.columns),
                )
            else:
                logger.warning("Main H5'te '%s' bulunamadi", BAR_TABLE_PATH)
                self._log_h5_structure(f)

            # --- Shell force combined ---
            if SHELL_TABLE_PATH in f:
                self._shell_df = self._read_h5_table(f, SHELL_TABLE_PATH)
                self._normalize_shell_columns()
                logger.info(
                    "Shell tablosu okundu: %s (%d satir, kolonlar: %s)",
                    SHELL_TABLE_PATH, len(self._shell_df), list(self._shell_df.columns),
                )
            else:
                logger.warning("Main H5'te '%s' bulunamadi", SHELL_TABLE_PATH)
                self._log_h5_structure(f)

        # Subcase ID'leri topla
        self._available_subcases = self._collect_subcase_ids()
        logger.info("Main H5 okundu. Subcases: %s", self._available_subcases)

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
        Birden fazla Main H5 dosyasini oku ve sonuclari birlestir.

        Returns
        -------
        bar_forces : Dict
            Birlestirilmis bar kuvvet sonuclari.
        shell_forces : Dict
            Birlestirilmis shell flux sonuclari.
        all_subcases : List[int]
            Tum H5'lerdeki subcase ID'leri.
        """
        merged_bar: Dict[Tuple[int, int], BarForceResult] = {}
        merged_shell: Dict[Tuple[int, int], ShellForceResult] = {}
        all_subcases: set = set()

        for h5_path in op2_paths:
            reader = OP2Reader(h5_path)
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
            "Toplu H5: %d dosya, %d subcase, %d bar, %d shell sonuc",
            len(op2_paths), len(all_subcases),
            len(merged_bar), len(merged_shell),
        )
        return merged_bar, merged_shell, sorted(all_subcases)

    @property
    def subcases(self) -> List[int]:
        return self._available_subcases

    def _collect_subcase_ids(self) -> List[int]:
        """H5 tablolarindan mevcut subcase ID'lerini topla."""
        sc_ids: set = set()

        if not self._bar_df.empty and "subcase_id" in self._bar_df.columns:
            sc_ids.update(self._bar_df["subcase_id"].unique().tolist())

        if not self._shell_df.empty and "subcase_id" in self._shell_df.columns:
            sc_ids.update(self._shell_df["subcase_id"].unique().tolist())

        return sorted(int(s) for s in sc_ids)

    def get_bar_forces(
        self,
        bar_eids: List[int],
        subcase_id: Optional[int] = None,
    ) -> Dict[Tuple[int, int], BarForceResult]:
        """
        Bar elementler icin axial force cikart.

        Parameters
        ----------
        bar_eids : List[int]
            Bar element ID listesi.
        subcase_id : Optional[int]
            Belirli bir subcase. None ise tum subcaseler.

        Returns
        -------
        Dict[Tuple[int, int], BarForceResult]
            (subcase_id, eid) -> BarForceResult mapping.
        """
        results = {}

        if self._bar_df.empty:
            logger.warning("Main H5'te bar element verisi bulunamadi")
            return results

        df = self._bar_df

        # Subcase filtresi
        if subcase_id is not None:
            df = df[df["subcase_id"] == subcase_id]

        # Istenen bar element ID'lerini filtrele
        df = df[df["element_id"].isin(bar_eids)]

        for _, row in df.iterrows():
            eid = int(row["element_id"])
            sc_id = int(row["subcase_id"])
            af = float(row.get("af", 0.0))

            # H5'te sadece AF var, diger kuvvetler sifir
            results[(sc_id, eid)] = BarForceResult(
                eid=eid,
                subcase_id=sc_id,
                axial_force=np.array([af]),
                shear_1=np.array([0.0]),
                shear_2=np.array([0.0]),
                bending_moment_a1=np.array([0.0]),
                bending_moment_a2=np.array([0.0]),
                bending_moment_b1=np.array([0.0]),
                bending_moment_b2=np.array([0.0]),
                torque=np.array([0.0]),
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
        Shell elementler icin membrane fluxlarini cikart.

        Parameters
        ----------
        shell_eids : List[int]
            Shell element ID listesi.
        subcase_id : Optional[int]
            Belirli bir subcase. None ise tum subcaseler.

        Returns
        -------
        Dict[Tuple[int, int], ShellForceResult]
            (subcase_id, eid) -> ShellForceResult mapping.
        """
        results = {}

        if self._shell_df.empty:
            logger.warning("Main H5'te shell element verisi bulunamadi")
            return results

        df = self._shell_df

        # Subcase filtresi
        if subcase_id is not None:
            df = df[df["subcase_id"] == subcase_id]

        # Istenen shell element ID'lerini filtrele
        df = df[df["element_id"].isin(shell_eids)]

        for _, row in df.iterrows():
            eid = int(row["element_id"])
            sc_id = int(row["subcase_id"])

            nx = float(row.get("nx", 0.0))
            ny = float(row.get("ny", 0.0))
            nxy = float(row.get("nxy", 0.0))

            # H5'te element tipi bilgisi yok, genel "SHELL" kullanilir
            elem_type = str(row.get("elem_type", "SHELL"))

            results[(sc_id, eid)] = ShellForceResult(
                eid=eid,
                elem_type=elem_type,
                subcase_id=sc_id,
                membrane_x=np.array([nx]),
                membrane_y=np.array([ny]),
                membrane_xy=np.array([nxy]),
                bending_x=np.array([0.0]),
                bending_y=np.array([0.0]),
                bending_xy=np.array([0.0]),
                shear_xz=np.array([0.0]),
                shear_yz=np.array([0.0]),
            )

        found = len(results)
        logger.info(
            "%d shell element icin flux verisi bulundu (istenen: %d)",
            found,
            len(shell_eids),
        )
        return results

    def get_load_case_info(self) -> pd.DataFrame:
        """Mevcut subcaselerin bilgisini dondur."""
        rows = []
        for sc_id in self._available_subcases:
            rows.append({"Subcase_ID": sc_id, "Label": str(sc_id)})
        return pd.DataFrame(rows)

    # ================================================================
    #  H5 okuma ve normalizasyon yardimci metodlari
    # ================================================================

    @staticmethod
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

    def _normalize_bar_columns(self) -> None:
        """Bar forces DataFrame kolon isimlerini normalize et."""
        col_map = {}
        for col in self._bar_df.columns:
            lower = col.lower().replace(" ", "_").replace("-", "_")
            if lower in ("element_id", "elementid", "bar_eid", "bareid", "bar_id", "eid", "id"):
                col_map[col] = "element_id"
            elif lower in ("subcase_id", "subcaseid", "subcase", "sc_id"):
                col_map[col] = "subcase_id"
            elif lower in ("af", "axial_force", "axialforce", "bar_axial", "axial"):
                col_map[col] = "af"

        self._bar_df = self._bar_df.rename(columns=col_map)

        for needed in ["element_id", "subcase_id", "af"]:
            if needed not in self._bar_df.columns:
                logger.warning(
                    "Bar H5 tablosu: '%s' kolonu bulunamadi, mevcut: %s",
                    needed, list(self._bar_df.columns),
                )

    def _normalize_shell_columns(self) -> None:
        """Shell forces DataFrame kolon isimlerini normalize et."""
        col_map = {}
        for col in self._shell_df.columns:
            lower = col.lower().replace(" ", "_").replace("-", "_")
            if lower in ("element_id", "elementid", "shell_eid", "shelleid", "eid", "id"):
                col_map[col] = "element_id"
            elif lower in ("subcase_id", "subcaseid", "subcase", "sc_id"):
                col_map[col] = "subcase_id"
            elif lower in ("mx", "nx", "membrane_x", "shell_nx"):
                col_map[col] = "nx"
            elif lower in ("my", "ny", "membrane_y", "shell_ny"):
                col_map[col] = "ny"
            elif lower in ("mxy", "nxy", "membrane_xy", "shell_nxy"):
                col_map[col] = "nxy"

        self._shell_df = self._shell_df.rename(columns=col_map)

        for needed in ["element_id", "subcase_id", "nx", "ny", "nxy"]:
            if needed not in self._shell_df.columns:
                logger.warning(
                    "Shell H5 tablosu: '%s' kolonu bulunamadi, mevcut: %s",
                    needed, list(self._shell_df.columns),
                )

    @staticmethod
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
