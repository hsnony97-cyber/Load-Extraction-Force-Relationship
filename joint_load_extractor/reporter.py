"""
Report Generator Module

Sonuçları Excel formatında raporlar.

Oluşturulan sheet'ler:
1. Bar Element Set    - Input bar element listesi
2. Element Connectivity - Bar elementler ve bağlı QUAD/TRIA
3. OP2 Bar Forces     - Bar element axial force ve diğer kuvvetler
4. OP2 Shell Fluxes   - Shell element membrane fluxları
5. H5 Joint Loads     - H5'ten okunan Joint Load Cap verileri
6. Correlation Summary - Korelasyon matrisi ve regresyon sonuçları
7. Bar_XXXX (per bar)  - Her bar element için detay sayfası
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from joint_load_extractor.bdf_parser import BarElementInfo
from joint_load_extractor.correlation import JointCorrelationResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Excel rapor oluşturucu."""

    # Hücre format sabitleri
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
        correlation_results: List[JointCorrelationResult],
        correlation_summary_df: pd.DataFrame,
    ) -> str:
        """
        Tam raporu oluştur.

        Returns
        -------
        str
            Oluşturulan dosya yolu.
        """
        logger.info("Excel raporu olusturuluyor: %s", self.output_path)

        with pd.ExcelWriter(self.output_path, engine="xlsxwriter") as writer:
            self.writer = writer
            workbook = writer.book

            # Format tanımları
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

            # 1. Bar Element Set
            self._write_bar_element_set(
                writer, workbook, bar_element_ids, header_fmt, int_fmt
            )

            # 2. Element Connectivity
            self._write_connectivity(
                writer, workbook, connectivity, header_fmt, bar_fmt, quad_fmt, tria_fmt, int_fmt, border_fmt
            )

            # 3. OP2 Bar Forces
            if not bar_forces_df.empty:
                self._write_dataframe(
                    writer, workbook, "OP2 Bar Forces", bar_forces_df,
                    header_fmt, number_fmt, int_fmt,
                )

            # 4. OP2 Shell Fluxes
            if not shell_forces_df.empty:
                self._write_dataframe(
                    writer, workbook, "OP2 Shell Fluxes", shell_forces_df,
                    header_fmt, number_fmt, int_fmt,
                )

            # 5. H5 Joint Loads
            if not h5_joint_loads.empty:
                self._write_dataframe(
                    writer, workbook, "H5 Joint Loads", h5_joint_loads,
                    header_fmt, number_fmt, int_fmt,
                )

            # 6. Correlation Summary
            if not correlation_summary_df.empty:
                self._write_dataframe(
                    writer, workbook, "Correlation Summary", correlation_summary_df,
                    header_fmt, number_fmt, int_fmt,
                )

            # 7. Per-bar detail sheets
            self._write_per_bar_details(
                writer, workbook, connectivity, correlation_results,
                header_fmt, number_fmt, int_fmt, bar_fmt, quad_fmt, tria_fmt, border_fmt,
            )

        logger.info("Rapor olusturuldu: %s", self.output_path)
        return self.output_path

    def _write_bar_element_set(self, writer, workbook, bar_eids, header_fmt, int_fmt):
        """Bar Element Set sheet'i."""
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
        """Element Connectivity sheet'i."""
        sheet_name = "Element Connectivity"

        rows = []
        for bar_eid, info in sorted(connectivity.items()):
            # Bar element satırı
            rows.append({
                "Bar_EID": bar_eid,
                "Node_A": info.node_a,
                "Node_B": info.node_b,
                "Connected_Type": "BAR",
                "Connected_EID": bar_eid,
                "Connected_Nodes": f"{info.node_a}, {info.node_b}",
                "PID": info.pid,
            })

            # Bağlı QUAD elementleri
            for qeid, qnodes in info.connected_quads.items():
                rows.append({
                    "Bar_EID": bar_eid,
                    "Node_A": info.node_a,
                    "Node_B": info.node_b,
                    "Connected_Type": "CQUAD4",
                    "Connected_EID": qeid,
                    "Connected_Nodes": ", ".join(str(n) for n in qnodes if n is not None),
                    "PID": "",
                })

            # Bağlı TRIA elementleri
            for teid, tnodes in info.connected_trias.items():
                rows.append({
                    "Bar_EID": bar_eid,
                    "Node_A": info.node_a,
                    "Node_B": info.node_b,
                    "Connected_Type": "CTRIA3",
                    "Connected_EID": teid,
                    "Connected_Nodes": ", ".join(str(n) for n in tnodes if n is not None),
                    "PID": "",
                })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)

        ws = writer.sheets[sheet_name]
        columns = list(df.columns)
        for col_idx, col_name in enumerate(columns):
            ws.write(0, col_idx, col_name, header_fmt)

        ws.set_column(0, 0, 12)  # Bar_EID
        ws.set_column(1, 2, 10)  # Node_A, Node_B
        ws.set_column(3, 3, 14)  # Connected_Type
        ws.set_column(4, 4, 14)  # Connected_EID
        ws.set_column(5, 5, 30)  # Connected_Nodes
        ws.set_column(6, 6, 10)  # PID

        # Renklendirme
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
        """Genel DataFrame yazma."""
        # Sheet ismi max 31 karakter
        sheet_name = sheet_name[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)

        ws = writer.sheets[sheet_name]
        for col_idx, col_name in enumerate(df.columns):
            ws.write(0, col_idx, col_name, header_fmt)
            # Kolon genişliği
            max_len = max(len(str(col_name)), df[col_name].astype(str).str.len().max())
            ws.set_column(col_idx, col_idx, min(max_len + 2, 25))

    def _write_per_bar_details(
        self, writer, workbook, connectivity, correlation_results,
        header_fmt, number_fmt, int_fmt, bar_fmt, quad_fmt, tria_fmt, border_fmt,
    ):
        """Her bar element icin detay sheet'i - regresyon denklemleri."""
        # Korelasyon sonuclarini bar_eid'ye gore grupla
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

            # --- Baglanti Bilgisi ---
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

            # --- Regresyon Denklemleri ---
            for res in results:
                ws.write(row, 0, f"SC {res.subcase_id} ET {res.element_type}", header_fmt)
                ws.merge_range(
                    row, 0, row, 6,
                    f"Multiple Regression (SC {res.subcase_id}, Element Type {res.element_type}, {res.n_connected_shells} shells)",
                    header_fmt,
                )
                row += 1

                if not res.equations:
                    ws.write(row, 0, "Denklem hesaplanamadi", border_fmt)
                    row += 2
                    continue

                # Tablo basliklari
                eq_headers = [
                    "Target", "Coeff Bar_Axial", "Coeff Avg_Nx",
                    "Coeff Avg_Ny", "Coeff Avg_Nxy", "Intercept", "R²",
                ]
                for j, h in enumerate(eq_headers):
                    ws.write(row, j, h, header_fmt)
                row += 1

                for eq in res.equations:
                    ws.write(row, 0, eq.target_name, border_fmt)
                    for ci, coeff in enumerate(eq.coefficients):
                        ws.write(row, ci + 1, float(coeff), number_fmt)
                    ws.write(row, 5, float(eq.intercept), number_fmt)
                    r2_fmt = good_r2_fmt if eq.r_squared >= 0.7 else number_fmt
                    ws.write(row, 6, eq.r_squared, r2_fmt)
                    row += 1

                row += 1

                # Denklem string olarak
                ws.write(row, 0, "Equations", header_fmt)
                ws.merge_range(row, 0, row, 6, "Equations", header_fmt)
                row += 1
                for eq in res.equations:
                    ws.merge_range(row, 0, row, 6, eq.equation_str(), eq_fmt)
                    row += 1
                row += 1

            ws.set_column(0, 0, 18)
            ws.set_column(1, 5, 16)
            ws.set_column(6, 6, 12)
