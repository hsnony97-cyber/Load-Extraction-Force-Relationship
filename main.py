#!/usr/bin/env python3
"""
Joint Load Extraction Force Relationship Tool
==============================================

BDF dosyasından bar elementler ve bağlı CQUAD4/CTRIA3 elementleri çıkarır,
OP2'den kuvvetleri okur, H5'ten Joint Load Cap verilerini alır ve
aralarındaki korelasyonu hesaplar.

Kullanım:
    python main.py --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx --output output.xlsx
    python main.py --bdf model.bdf --op2 file1.op2 file2.op2 file3.op2 --h5 joint_loads.h5 --excel input.xlsx

Gerekli Input Dosyaları:
    - BDF: Nastran bulk data dosyası (element bağlantıları için)
    - OP2: Nastran output dosyası (kuvvetler/fluxlar için)
    - H5:  Joint Load Extraction dosyası (Joint Load Cap tablosu)
    - Excel: "Bar Element Set" sheet'i olan Excel dosyası (bar element listesi)

Çıktı:
    - Excel dosyası: Bağlantı, kuvvetler, korelasyon sonuçları
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from joint_load_extractor.bdf_parser import BDFParser, BarElementInfo
from joint_load_extractor.op2_reader import OP2Reader, BarForceResult, ShellForceResult
from joint_load_extractor.h5_reader import H5Reader
from joint_load_extractor.correlation import CorrelationEngine, JointCorrelationResult
from joint_load_extractor.reporter import ReportGenerator
from joint_load_extractor.predictor import predict_from_results, predict_from_h5, predict_from_excel_and_h5


def setup_logging(verbose: bool = False) -> None:
    """Logging konfigürasyonu."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_bar_element_set(excel_path: str) -> List[int]:
    """
    Excel dosyasından bar element listesini oku.

    "Bar Element Set" adlı sheet'ten ilk kolon okunur.

    Parameters
    ----------
    excel_path : str
        Excel dosyası yolu.

    Returns
    -------
    List[int]
        Bar element ID listesi.
    """
    logger = logging.getLogger(__name__)
    logger.info("Excel dosyasi okunuyor: %s", excel_path)

    try:
        df = pd.read_excel(excel_path, sheet_name="Bar Element Set")
    except ValueError:
        # Sheet ismi biraz farklı olabilir
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        logger.info("Mevcut sheet'ler: %s", sheet_names)

        # "bar" ve "element" içeren ilk sheet'i bul
        target_sheet = None
        for name in sheet_names:
            if "bar" in name.lower() and "element" in name.lower():
                target_sheet = name
                break

        if target_sheet is None:
            # İlk sheet'i kullan
            target_sheet = sheet_names[0]
            logger.warning(
                "'Bar Element Set' bulunamadi, '%s' kullaniliyor", target_sheet
            )

        df = pd.read_excel(excel_path, sheet_name=target_sheet)

    # İlk kolondaki değerleri al
    first_col = df.columns[0]
    bar_eids = df[first_col].dropna().astype(int).tolist()
    logger.info("%d bar element okundu", len(bar_eids))
    return bar_eids


def build_bar_forces_dataframe(
    bar_forces: Dict[Tuple[int, int], BarForceResult],
) -> pd.DataFrame:
    """Bar kuvvet sonuçlarını DataFrame'e dönüştür."""
    rows = []
    for (sc_id, eid), result in sorted(bar_forces.items()):
        # Her zaman adımı için bir satır
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
    """Shell flux sonuçlarını DataFrame'e dönüştür."""
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
    veri toplayarak coklu regresyon denklemi olusturur:
      Target = a1*Bar_Axial + a2*Avg_Nx + a3*Avg_Ny + a4*Avg_Nxy + b
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

        # OP2 verilerini TUM subcase'ler uzerinden topla
        # Her subcase'ten ortalama deger alinir (statik icin ntimes=1)
        collected_sc_ids = []
        collected_axial = []
        collected_avg_nx = []
        collected_avg_ny = []
        collected_avg_nxy = []

        for sc_id_key, _ in sc_keys:
            if subcase_id is not None and sc_id_key != subcase_id:
                continue

            bar_result = bar_forces.get((sc_id_key, bar_eid))
            if bar_result is None:
                continue

            shell_nx, shell_ny, shell_nxy = [], [], []
            for seid in shell_eids:
                sf = shell_forces.get((sc_id_key, seid))
                if sf is not None:
                    shell_nx.append(sf.membrane_x)
                    shell_ny.append(sf.membrane_y)
                    shell_nxy.append(sf.membrane_xy)

            if not shell_nx:
                continue

            # Her subcase icin ortalama (ntimes boyunca mean)
            axial_mean = np.mean(bar_result.axial_force)
            avg_nx_mean = np.mean(np.mean(shell_nx, axis=0))
            avg_ny_mean = np.mean(np.mean(shell_ny, axis=0))
            avg_nxy_mean = np.mean(np.mean(shell_nxy, axis=0))

            collected_sc_ids.append(sc_id_key)
            collected_axial.append(axial_mean)
            collected_avg_nx.append(avg_nx_mean)
            collected_avg_ny.append(avg_ny_mean)
            collected_avg_nxy.append(avg_nxy_mean)

        if not collected_sc_ids:
            logger.warning("Bar %d: Hicbir subcase icin veri toplanamadi", bar_eid)
            continue

        n_shells = len([s for s in shell_eids
                        if shell_forces.get((collected_sc_ids[0], s)) is not None])

        logger.info("  Bar %d: %d subcase, %d shell uzerinden veri toplandi",
                     bar_eid, len(collected_sc_ids), n_shells)

        # OP2 verilerini subcase_id'ye gore dict'e koy (eslestirme icin)
        op2_by_sc = {}
        for i, sc_id in enumerate(collected_sc_ids):
            op2_by_sc[sc_id] = (
                collected_axial[i],
                collected_avg_nx[i],
                collected_avg_ny[i],
                collected_avg_nxy[i],
            )

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
                # H5 satirlarini OP2 subcase ID'leri ile esleştir
                matched_axial = []
                matched_nx = []
                matched_ny = []
                matched_nxy = []
                matched_h5_indices = []
                matched_sc_ids = []

                for idx, row in h5_subset.iterrows():
                    h5_sc = int(row[sc_col])
                    if h5_sc in op2_by_sc:
                        ax, nx, ny, nxy = op2_by_sc[h5_sc]
                        matched_axial.append(ax)
                        matched_nx.append(nx)
                        matched_ny.append(ny)
                        matched_nxy.append(nxy)
                        matched_h5_indices.append(idx)
                        matched_sc_ids.append(h5_sc)

                n_matched = len(matched_h5_indices)
                n_h5_total = len(h5_subset)
                n_op2_total = len(collected_sc_ids)

                logger.info(
                    "  Bar %d, ET %s: %d/%d H5 satir OP2 ile eslesti "
                    "(H5: %d satir, OP2: %d subcase)",
                    bar_eid, et, n_matched, n_h5_total, n_h5_total, n_op2_total,
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
                pred_nx = np.array(matched_nx)
                pred_ny = np.array(matched_ny)
                pred_nxy = np.array(matched_nxy)
            else:
                # Subcase kolonu yoksa eski davranis (sirali eslestirme)
                logger.warning(
                    "  Bar %d: H5'te Subcase ID kolonu bulunamadi, "
                    "sirali eslestirme yapiliyor",
                    bar_eid,
                )
                h5_matched = h5_subset
                pred_axial = np.array(collected_axial)
                pred_nx = np.array(collected_avg_nx)
                pred_ny = np.array(collected_avg_ny)
                pred_nxy = np.array(collected_avg_nxy)
                matched_sc_ids = list(collected_sc_ids)

            logger.info("  Bar %d, ElementType %s, %d eslesen veri noktasi",
                        bar_eid, et, len(pred_axial))

            result = engine.compute_joint_correlation(
                bar_eid=bar_eid,
                subcase_id=0,  # tum subcase'ler birlesitirildi
                element_type=int(et),
                bar_axial=pred_axial,
                avg_shell_nx=pred_nx,
                avg_shell_ny=pred_ny,
                avg_shell_nxy=pred_nxy,
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

    Ortalama almak yerine, her shell element icin bireysel regresyon:
      Target = a1*Bar_Axial + a2*Shell_Nx + a3*Shell_Ny + a4*Shell_Nxy + b

    Her (bar_eid, shell_eid, element_type) kombinasyonu icin ayri sonuc uretir.
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

        # Shell EID -> (type_str) eslesmesi
        shell_info = {}
        for qeid in info.connected_quads:
            shell_info[qeid] = "CQUAD4"
        for teid in info.connected_trias:
            shell_info[teid] = "CTRIA3"

        # Her subcase icin bar axial toplama
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

        # Element Type ve Subcase kolonlarini bul
        et_col = _find_element_type_col(h5_data)
        sc_col = _find_subcase_col(h5_data)
        element_types = sorted(h5_data[et_col].unique()) if et_col else [0]

        # Her shell element icin ayri regresyon
        for shell_eid, shell_type_str in shell_info.items():
            # Bu shell icin subcase bazli veri topla
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

            # OP2 by subcase dict
            op2_by_sc = {}
            for i, sc_id in enumerate(per_shell_sc_ids):
                op2_by_sc[sc_id] = (
                    per_shell_axial[i],
                    per_shell_nx[i],
                    per_shell_ny[i],
                    per_shell_nxy[i],
                )

            n_shells_total = len(shell_info)

            for et in element_types:
                h5_subset = h5_data[h5_data[et_col] == et].copy() if et_col else h5_data

                if h5_subset.empty:
                    continue

                # Subcase eslestirmesi
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
                            ax, nx, ny, nxy = op2_by_sc[h5_sc]
                            matched_axial.append(ax)
                            matched_nx.append(nx)
                            matched_ny.append(ny)
                            matched_nxy.append(nxy)
                            matched_h5_indices.append(idx)
                            matched_sc_ids.append(h5_sc)

                    if not matched_h5_indices:
                        continue

                    h5_matched = h5_subset.loc[matched_h5_indices].reset_index(drop=True)
                    pred_axial = np.array(matched_axial)
                    pred_nx = np.array(matched_nx)
                    pred_ny = np.array(matched_ny)
                    pred_nxy = np.array(matched_nxy)
                else:
                    h5_matched = h5_subset
                    pred_axial = np.array(per_shell_axial)
                    pred_nx = np.array(per_shell_nx)
                    pred_ny = np.array(per_shell_ny)
                    pred_nxy = np.array(per_shell_nxy)
                    matched_sc_ids = list(per_shell_sc_ids)

                result = per_shell_engine.compute_joint_correlation(
                    bar_eid=bar_eid,
                    subcase_id=0,
                    element_type=int(et),
                    bar_axial=pred_axial,
                    avg_shell_nx=pred_nx,
                    avg_shell_ny=pred_ny,
                    avg_shell_nxy=pred_nxy,
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


def main():
    """Ana giriş noktası."""
    parser = argparse.ArgumentParser(
        description="Joint Load Extraction Force Relationship Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornek kullanim:
  python main.py --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx
  python main.py --bdf model.bdf --op2 file1.op2 file2.op2 file3.op2 --h5 joint_loads.h5 --excel input.xlsx
  python main.py --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx --output results.xlsx --subcase 1
  python main.py --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx --h5-group "JOINT_LOADS_CAP/table"
        """,
    )

    parser.add_argument(
        "--gui", action="store_true",
        help="Tkinter grafik arayuzu ile baslat",
    )
    parser.add_argument(
        "--bdf", required=False, default=None,
        help="Nastran BDF dosyasi yolu",
    )
    parser.add_argument(
        "--op2", required=False, default=None, nargs="+",
        help="Nastran OP2 dosyasi yolu (birden fazla dosya verilebilir)",
    )
    parser.add_argument(
        "--h5", required=False, default=None,
        help="Joint Load Extraction H5 dosyasi yolu",
    )
    parser.add_argument(
        "--excel", required=False, default=None,
        help="Bar Element Set iceren Excel dosyasi yolu",
    )
    parser.add_argument(
        "--output", default="joint_load_correlation_output.xlsx",
        help="Cikti Excel dosyasi yolu (varsayilan: joint_load_correlation_output.xlsx)",
    )
    parser.add_argument(
        "--subcase", type=int, default=None,
        help="Belirli bir subcase ID. Belirtilmezse tum subcaseler islenir",
    )
    parser.add_argument(
        "--h5-group", default=None,
        help="H5 dosyasindaki Joint Load Cap tablosunun yolu (ornek: 'JOINT_LOADS_CAP/table')",
    )
    parser.add_argument(
        "--prediction-h5", default=None,
        help="Tahmin icin ayri H5 dosyasi (bar element combined + shell force combined)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Detayli cikti",
    )

    args = parser.parse_args()

    # GUI modu
    if args.gui:
        from gui import Application
        app = Application()
        app.mainloop()
        return

    # CLI modu: dosya parametreleri zorunlu
    missing = []
    for param in ["bdf", "op2", "h5", "excel"]:
        if getattr(args, param) is None:
            missing.append(f"--{param}")
    if missing:
        parser.error(f"CLI modunda su parametreler zorunludur: {', '.join(missing)}\n"
                     f"Grafik arayuz icin: python main.py --gui")

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Joint Load Extraction Force Relationship Tool")
    logger.info("=" * 60)

    # Input dosya kontrolü
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

    # ============================================================
    # ADIM 1: Excel'den bar element listesini oku
    # ============================================================
    logger.info("-" * 40)
    logger.info("ADIM 1: Bar element listesi okunuyor...")
    bar_element_ids = read_bar_element_set(args.excel)
    logger.info("  %d bar element okundu", len(bar_element_ids))

    # ============================================================
    # ADIM 2: BDF'yi parse et ve bağlantıları bul
    # ============================================================
    logger.info("-" * 40)
    logger.info("ADIM 2: BDF parse ediliyor...")
    bdf_parser = BDFParser(args.bdf)
    bdf_parser.parse()

    connectivity = bdf_parser.get_bar_connectivity(bar_element_ids)
    logger.info("  %d bar element icin baglanti bulundu", len(connectivity))

    # Bağlı shell element listesi
    all_shell_eids = set()
    for info in connectivity.values():
        all_shell_eids.update(info.connected_quads.keys())
        all_shell_eids.update(info.connected_trias.keys())
    logger.info("  Toplam %d bagli shell element", len(all_shell_eids))

    # ============================================================
    # ADIM 3: OP2'den kuvvetleri oku (toplu dosya destegi)
    # ============================================================
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

    # ============================================================
    # ADIM 4: H5'ten Joint Load Cap verilerini oku
    # ============================================================
    logger.info("-" * 40)
    logger.info("ADIM 4: H5 dosyasi okunuyor...")
    h5_reader = H5Reader(args.h5)
    h5_reader.read(group_path=args.h5_group)
    logger.info("  Joint Load Cap: %d satir", len(h5_reader.joint_load_cap))

    # ============================================================
    # ADIM 5: Korelasyon analizi
    # ============================================================
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

    # ============================================================
    # ADIM 5b: Per-shell korelasyon analizi
    # ============================================================
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

    # ============================================================
    # ADIM 6: Rapor oluştur
    # ============================================================
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

    # ============================================================
    # ADIM 7: Tahmin (Prediction) CSV olustur
    # ============================================================
    logger.info("-" * 40)
    logger.info("ADIM 7: Tahmin CSV olusturuluyor...")

    csv_path = str(Path(args.output).with_suffix(".csv"))

    prediction_h5 = getattr(args, "prediction_h5", None)
    if prediction_h5:
        # Katsayilari output Excel'den, kuvvetleri Prediction H5'ten oku
        logger.info("  Prediction H5: %s", prediction_h5)
        logger.info("  Katsayilar:    %s (Correlation Summary sheet)", output_path)
        pred_df = predict_from_h5(
            coefficients_excel=output_path,
            prediction_h5_path=prediction_h5,
            output_csv=csv_path,
            connectivity=connectivity,
        )
    else:
        # Mevcut OP2 verileriyle tahmin (correlation_results - ortalama shell)
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

    # Ozet bilgi yazdir
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


if __name__ == "__main__":
    main()
