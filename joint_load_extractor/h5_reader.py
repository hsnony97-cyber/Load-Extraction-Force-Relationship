"""
H5 Reader Module

Joint Load Extraction H5 dosyasından Joint Load Cap tablosunu okur.
Ağaç yapısı: Joint Load Cap -> table
Kolonlar: FBearingX, FBearingY, NX Bypass, NY Bypass, NXY Bypass
"""

import logging
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class H5Reader:
    """H5 dosyasından Joint Load Cap verilerini okur."""

    # Joint Load Cap tablosunda beklenen kolon isimleri
    EXPECTED_COLUMNS = [
        "FBearingX",
        "FBearingY",
        "NX Bypass",
        "NY Bypass",
        "NXY Bypass",
    ]

    # Olası ID kolon isimleri
    ID_COLUMN_CANDIDATES = [
        "Bar EID",
        "BarEID",
        "Bar_EID",
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
        """
        H5 dosyasını oku ve Joint Load Cap tablosunu yükle.

        Parameters
        ----------
        group_path : str, optional
            Joint Load Cap tablosunun yolu. None ise otomatik aranır.
        """
        logger.info("H5 dosyasi okunuyor: %s", self.h5_path)

        with h5py.File(self.h5_path, "r") as f:
            # Ağaç yapısını keşfet
            self._h5_structure = self._explore_structure(f)
            logger.info("H5 agac yapisi:\n%s", self._format_structure(self._h5_structure))

            # Joint Load Cap tablosunu bul
            if group_path:
                table_path = group_path
            else:
                table_path = self._find_joint_load_cap_table(f)

            if table_path is None:
                raise ValueError(
                    "H5 dosyasinda Joint Load Cap tablosu bulunamadi. "
                    "Lutfen group_path parametresini belirtin."
                )

            logger.info("Joint Load Cap tablosu bulundu: %s", table_path)
            self._joint_load_cap_df = self._read_table(f, table_path)

        logger.info(
            "Joint Load Cap tablosu okundu: %d satir, %d kolon",
            len(self._joint_load_cap_df),
            len(self._joint_load_cap_df.columns),
        )
        logger.info("Kolonlar: %s", list(self._joint_load_cap_df.columns))

    @property
    def joint_load_cap(self) -> pd.DataFrame:
        """Joint Load Cap tablosu."""
        if self._joint_load_cap_df is None:
            raise RuntimeError("Henuz H5 dosyasi okunmadi. Once read() cagiriniz.")
        return self._joint_load_cap_df

    @property
    def structure(self) -> Dict:
        """H5 dosyasının ağaç yapısı."""
        return self._h5_structure

    def get_joint_loads_for_bar(self, bar_eid: int) -> Optional[pd.DataFrame]:
        """
        Belirli bir bar element için Joint Load Cap verilerini döndür.

        Parameters
        ----------
        bar_eid : int
            Bar element ID.

        Returns
        -------
        Optional[pd.DataFrame]
            Bar element için Joint Load Cap verileri veya None.
        """
        df = self.joint_load_cap
        id_col = self._find_id_column(df)

        if id_col is None:
            logger.warning("Joint Load Cap tablosunda ID kolonu bulunamadi")
            return None

        mask = df[id_col] == bar_eid
        if not mask.any():
            return None

        return df[mask].copy()

    def get_all_bar_eids(self) -> List[int]:
        """H5 tablosundaki tüm bar element ID'lerini döndür."""
        df = self.joint_load_cap
        id_col = self._find_id_column(df)
        if id_col is None:
            return []
        return sorted(df[id_col].unique().tolist())

    def _find_joint_load_cap_table(self, f: h5py.File) -> Optional[str]:
        """H5 dosyasında Joint Load Cap tablosunu otomatik bul."""
        candidates = []

        def _visitor(name, obj):
            name_lower = name.lower()
            if "joint" in name_lower and "load" in name_lower and "cap" in name_lower:
                if isinstance(obj, h5py.Dataset):
                    candidates.append(name)
                elif isinstance(obj, h5py.Group):
                    # Grubun altında 'table' var mı?
                    if "table" in obj:
                        candidates.append(f"{name}/table")

        f.visititems(_visitor)

        if not candidates:
            # Daha geniş arama: sadece 'table' aranır
            def _visitor_table(name, obj):
                if name.endswith("/table") and isinstance(obj, h5py.Dataset):
                    candidates.append(name)
            f.visititems(_visitor_table)

        if candidates:
            # En iyi eşleşmeyi seç (Joint Load Cap > table tercih edilir)
            for c in candidates:
                if "cap" in c.lower() and "table" in c.lower():
                    return c
            return candidates[0]

        return None

    def _read_table(self, f: h5py.File, path: str) -> pd.DataFrame:
        """H5 dataset'ini DataFrame'e dönüştür."""
        dataset = f[path]

        if dataset.dtype.names:
            # Structured array (compound dataset)
            data = {}
            for col_name in dataset.dtype.names:
                col_data = dataset[col_name]
                # Byte stringleri decode et
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
            # Regular array
            return pd.DataFrame(dataset[:])

    def _find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """DataFrame'de ID kolonunu bul."""
        columns_lower = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

        for candidate in self.ID_COLUMN_CANDIDATES:
            candidate_normalized = candidate.lower().replace(" ", "").replace("_", "")
            if candidate_normalized in columns_lower:
                return columns_lower[candidate_normalized]

        # İlk integer kolonu ID olarak kabul et
        for col in df.columns:
            if df[col].dtype in [np.int32, np.int64, np.uint32, np.uint64]:
                logger.info("ID kolonu olarak '%s' kullaniliyor", col)
                return col

        return None

    def _explore_structure(self, f: h5py.File, max_depth: int = 5) -> Dict:
        """H5 dosyasının ağaç yapısını keşfet."""
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
        """Ağaç yapısını string formatında döndür."""
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
