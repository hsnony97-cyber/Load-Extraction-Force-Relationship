"""
BDF Parser Module

BDF dosyasını okuyarak bar elementleri ve bunlara node'lar üzerinden
bağlı olan CQUAD4/CTRIA3 elementlerini tespit eder.
"""

import logging
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from pyNastran.bdf.bdf import BDF

logger = logging.getLogger(__name__)


@dataclass
class BarElementInfo:
    """Bir bar element ve bağlı elemanlarının bilgisi."""
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
    """BDF dosyasını parse eder ve element bağlantılarını çıkarır."""

    # Desteklenen bar element tipleri
    BAR_TYPES = {"CBAR", "CBEAM"}
    # Desteklenen quad element tipleri
    QUAD_TYPES = {"CQUAD4", "CQUAD8", "CQUADR"}
    # Desteklenen tria element tipleri
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
        """BDF dosyasını oku ve element verilerini indeksle."""
        logger.info("BDF dosyasi okunuyor: %s", self.bdf_path)

        bdf_to_read = self.bdf_path
        tmp_path = None

        # Dosya UTF-8 degilse, latin-1 ile okuyup UTF-8 gecici dosyaya yaz
        try:
            with open(self.bdf_path, "r", encoding="utf-8") as f:
                f.read()
        except UnicodeDecodeError:
            logger.warning("BDF dosyasi UTF-8 degil, latin-1 olarak yeniden kodlaniyor...")
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bdf")
            os.close(tmp_fd)
            with open(self.bdf_path, "r", encoding="latin-1") as src:
                with open(tmp_path, "w", encoding="utf-8") as dst:
                    shutil.copyfileobj(src, dst)
            bdf_to_read = tmp_path

        try:
            self.model = BDF(debug=False)
            self.model.read_bdf(bdf_to_read, xref=True, encoding="utf-8")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        logger.info(
            "BDF okundu: %d element, %d node",
            len(self.model.elements),
            len(self.model.nodes),
        )
        self._build_node_element_maps()

    def _build_node_element_maps(self) -> None:
        """Node-to-element haritalarını oluştur."""
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
        """
        Verilen bar element ID'leri için bağlı QUAD/TRIA elementlerini bul.

        Her bar elementin 2 node'u var (GA, GB).
        Bu node'lardan herhangi birine bağlı olan CQUAD4/CTRIA3
        elementleri tespit edilir.

        Parameters
        ----------
        bar_element_ids : List[int]
            İşlenecek bar element ID listesi.

        Returns
        -------
        Dict[int, BarElementInfo]
            Bar element ID -> BarElementInfo mapping.
        """
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
                    eid,
                    elem_type,
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
                eid=eid,
                pid=pid,
                node_a=node_a,
                node_b=node_b,
            )

            # Bar node'larına bağlı QUAD elementlerini bul
            connected_quad_eids = set()
            for nid in [node_a, node_b]:
                connected_quad_eids.update(self._node_to_quads.get(nid, set()))

            for qeid in sorted(connected_quad_eids):
                bar_info.connected_quads[qeid] = self._element_nodes[qeid]

            # Bar node'larına bağlı TRIA elementlerini bul
            connected_tria_eids = set()
            for nid in [node_a, node_b]:
                connected_tria_eids.update(self._node_to_trias.get(nid, set()))

            for teid in sorted(connected_tria_eids):
                bar_info.connected_trias[teid] = self._element_nodes[teid]

            result[eid] = bar_info
            logger.info(
                "Bar %d: NodeA=%d, NodeB=%d, %d QUAD, %d TRIA bagli",
                eid,
                node_a,
                node_b,
                len(bar_info.connected_quads),
                len(bar_info.connected_trias),
            )

        if missing_eids:
            logger.warning(
                "BDF'de bulunamayan %d bar element var: %s",
                len(missing_eids),
                missing_eids[:10],
            )

        return result

    def get_element_type(self, eid: int) -> str:
        """Element tipini döndür."""
        return self._element_types.get(eid, "UNKNOWN")

    def get_element_nodes(self, eid: int) -> List[int]:
        """Element node ID'lerini döndür."""
        return self._element_nodes.get(eid, [])

    def get_element_pid(self, eid: int) -> int:
        """Element property ID'sini döndür."""
        return self._element_pids.get(eid, 0)
