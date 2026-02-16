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
        per_shell_results: Optional[List[JointCorrelationResult]] = None,  # deprecated, artik kullanilmiyor
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

            # 7. Predicted vs Actual
            self._write_predicted_vs_actual(
                writer, workbook, correlation_results,
                header_fmt, number_fmt, int_fmt, border_fmt,
            )

            # 8. Total Summary - tum bar elementler icin tek tablo
            self._write_total_summary(
                writer, workbook, connectivity,
                correlation_results,
                header_fmt, number_fmt, int_fmt, border_fmt,
            )

            # 9. Per-bar detail sheets
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

    def _write_total_summary(
        self, writer, workbook, connectivity, correlation_results,
        header_fmt, number_fmt, int_fmt, border_fmt,
    ):
        """
        Total Summary sheet'i - long format.

        Her hedef (target) icin ayri satir.
        Kolonlar:
          Bar_EID | Element_Type | Subcase_ID | Shell_EID | Shell_Type |
          Target | Coeff_Bar_Axial | Coeff_Shell_Nx | Coeff_Shell_Ny |
          Coeff_Shell_Nxy | Intercept | R2
        """
        rows = []
        for res in correlation_results:
            shell_eids_str = ",".join(str(s) for s in res.shell_eids_used) if res.shell_eids_used else ""
            base = {
                "Bar_EID": res.bar_eid,
                "Element_Type": res.element_type,
                "Subcase_ID": res.subcase_id,
                "N_Shells": res.n_connected_shells,
                "Shell_EIDs": shell_eids_str,
            }

            for eq in res.equations:
                row = dict(base)
                row["Target"] = eq.target_name
                for pname, coeff in zip(eq.predictor_names, eq.coefficients):
                    row[f"Coeff_{pname}"] = float(coeff)
                row["Intercept"] = float(eq.intercept)
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

        # Kolon genislikleri
        col_widths = {
            "Bar_EID": 12,
            "Element_Type": 12,
            "Subcase_ID": 12,
            "Shell_EID": 12,
            "Shell_Type": 10,
            "Target": 16,
            "Coeff_Bar_Axial": 16,
            "Coeff_Shell_Nx": 16,
            "Coeff_Shell_Ny": 16,
            "Coeff_Shell_Nxy": 16,
            "Intercept": 14,
            "R2": 10,
        }
        for col_idx, col_name in enumerate(df.columns):
            width = col_widths.get(col_name, 14)
            ws.set_column(col_idx, col_idx, width)

        # R² renklendirme
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
        """Predicted vs Actual sheet'i - per-shell predictor'lar ile."""
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

                predicted_arr = X @ eq.coefficients + eq.intercept

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
                    for col_idx_p, pname in enumerate(pred_names):
                        row[pname] = float(X[i, col_idx_p])

                    row["Actual"] = actual_val
                    row["Predicted"] = pred_val
                    row["Error_%"] = error_pct
                    rows.append(row)

        if not rows:
            logger.info("Predicted vs Actual: veri yok, sheet atlanıyor")
            return

        df = pd.DataFrame(rows)
        sheet_name = "Predicted vs Actual"
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
        ws = writer.sheets[sheet_name]

        for col_idx, col_name in enumerate(df.columns):
            ws.write(0, col_idx, col_name, header_fmt)

        # Kolon genislikleri
        n_cols = len(df.columns)
        ws.set_column(0, 0, 10)   # Bar_EID
        ws.set_column(1, 1, 12)   # Element_Type
        ws.set_column(2, 2, 10)   # Subcase_ID
        ws.set_column(3, 3, 14)   # Target
        # Per-shell predictor kolonlari (degisken sayida)
        pred_end = 4 + len([c for c in df.columns if c.startswith("Bar_") or c.startswith("Shell_")])
        ws.set_column(4, pred_end, 16)
        # Actual, Predicted, Error_%
        ws.set_column(n_cols - 3, n_cols - 2, 16)
        ws.set_column(n_cols - 1, n_cols - 1, 10)

        # Error % renklendirme
        good_err_fmt = workbook.add_format({
            "num_format": "0.00", "border": 1, "bg_color": self.CORR_POS_COLOR,
        })
        bad_err_fmt = workbook.add_format({
            "num_format": "0.00", "border": 1, "bg_color": self.CORR_NEG_COLOR,
        })
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
                    "Target", "Coeff Bar_Axial", "Coeff Shell_Nx",
                    "Coeff Shell_Ny", "Coeff Shell_Nxy", "Intercept", "R²",
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
