#!/usr/bin/env python3
"""
Joint Load Extraction Force Relationship Tool
==============================================

BDF dosyasindan bar elementler ve bagli CQUAD4/CTRIA3 elementleri cikarir,
Main H5'ten kuvvetleri okur, H5'ten Joint Load Cap verilerini alir ve
aralarindaki korelasyonu hesaplar.

Kullanim:
    python main.py --bdf model.bdf --main-h5 forces.h5 --h5 joint_loads.h5 --excel input.xlsx --output output.xlsx
    python main.py --bdf model.bdf --main-h5 file1.h5 file2.h5 file3.h5 --h5 joint_loads.h5 --excel input.xlsx

Gerekli Input Dosyalari:
    - BDF: Nastran bulk data dosyasi (element baglantilari icin)
    - Main H5: Kuvvet/flux verileri (prediction H5 formatinda: ELFORCE_BAR_COMBINED, ELFORCE_SHELL_COMBINED)
    - H5:  Joint Load Extraction dosyasi (Joint Load Cap tablosu)
    - Excel: "Bar Element Set" sheet'i olan Excel dosyasi (bar element listesi)

Cikti:
    - Excel dosyasi: Baglanti, kuvvetler, korelasyon sonuclari
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
    """Logging konfigurasyonu."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_bar_element_set(excel_path: str) -> List[int]:
    """
    Excel dosyasindan bar element listesini oku.

    "Bar Element Set" adli sheet'ten ilk kolon okunur.

    Parameters
    ----------
    excel_path : str
        Excel dosyasi yolu.

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
        # Sheet ismi biraz farkli olabilir
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        logger.info("Mevcut sheet'ler: %s", sheet_names)

        # "bar" ve "element" iceren ilk sheet'i bul
        target_sheet = None
        for name in sheet_names:
            if "bar" in name.lower() and "element" in name.lower():
                target_sheet = name
                break

        if target_sheet is None:
            # Ilk sheet'i kullan
            target_sheet = sheet_names[0]
            logger.warning(
                "'Bar Element Set' bulunamadi, '%s' kullaniliyor", target_sheet
            )

        df = pd.read_excel(excel_path, sheet_name=target_sheet)

    # Ilk kolondaki degerleri al
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
        # Her zaman adimi icin bir satir
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
            logger.warning("Bar %d icin Main H5 kuvvet verisi yok", bar_eid)
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

        # H5 verileri subcase_id'ye gore dict'e koy (H5 eslestirme icin)
        h5_by_sc = {}
        for i, sc_id in enumerate(collected_sc_ids):
            sc_data = {"axial": collected_axial[i], "shells": {}}
            for seid in shells_with_full_data:
                sc_data["shells"][seid] = {
                    "nx": collected_shell[seid]["nx"][i],
                    "ny": collected_shell[seid]["ny"][i],
                    "nxy": collected_shell[seid]["nxy"][i],
                }
            h5_by_sc[sc_id] = sc_data

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
                    if h5_sc in h5_by_sc:
                        sc_data = h5_by_sc[h5_sc]
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
                    "  Bar %d, ET %s: %d/%d H5 satir Main H5 ile eslesti",
                    bar_eid, et, n_matched, n_h5_total,
                )

                if not matched_h5_indices:
                    logger.warning(
                        "  Bar %d, ET %s: Hicbir subcase eslesmiyor! "
                        "Main H5 SC: %s, H5 SC: %s",
                        bar_eid, et,
                        sorted(h5_by_sc.keys()),
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


def main():
    """Ana giris noktasi."""
    parser = argparse.ArgumentParser(
        description="Joint Load Extraction Force Relationship Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornek kullanim:
  python main.py --bdf model.bdf --main-h5 forces.h5 --h5 joint_loads.h5 --excel input.xlsx
  python main.py --bdf model.bdf --main-h5 file1.h5 file2.h5 --h5 joint_loads.h5 --excel input.xlsx
  python main.py --bdf model.bdf --main-h5 forces.h5 --h5 joint_loads.h5 --excel input.xlsx --output results.xlsx --subcase 1
  python main.py --bdf model.bdf --main-h5 forces.h5 --h5 joint_loads.h5 --excel input.xlsx --h5-group "JOINT_LOADS_CAP/table"
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
        "--main-h5", required=False, default=None, nargs="+",
        dest="main_h5",
        help="Main H5 dosyasi yolu - bar/shell kuvvetleri (birden fazla dosya verilebilir)",
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
    for param, param_name in [
        ("bdf", "--bdf"),
        ("main_h5", "--main-h5"),
        ("h5", "--h5"),
        ("excel", "--excel"),
    ]:
        if getattr(args, param) is None:
            missing.append(param_name)
    if missing:
        parser.error(f"CLI modunda su parametreler zorunludur: {', '.join(missing)}\n"
                     f"Grafik arayuz icin: python main.py --gui")

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Joint Load Extraction Force Relationship Tool")
    logger.info("=" * 60)

    # Input dosya kontrolu
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

    # Main H5 dosyalarini kontrol et
    main_h5_paths = args.main_h5
    for h5_path in main_h5_paths:
        p = Path(h5_path)
        if not p.exists():
            logger.error("Main H5 dosyasi bulunamadi: %s", h5_path)
            sys.exit(1)
        logger.info("Main H5: %s", p.resolve())
    logger.info("Toplam %d Main H5 dosyasi", len(main_h5_paths))

    # ============================================================
    # ADIM 1: Excel'den bar element listesini oku
    # ============================================================
    logger.info("-" * 40)
    logger.info("ADIM 1: Bar element listesi okunuyor...")
    bar_element_ids = read_bar_element_set(args.excel)
    logger.info("  %d bar element okundu", len(bar_element_ids))

    # ============================================================
    # ADIM 2: BDF'yi parse et ve baglantilari bul
    # ============================================================
    logger.info("-" * 40)
    logger.info("ADIM 2: BDF parse ediliyor...")
    bdf_parser = BDFParser(args.bdf)
    bdf_parser.parse()

    connectivity = bdf_parser.get_bar_connectivity(bar_element_ids)
    logger.info("  %d bar element icin baglanti bulundu", len(connectivity))

    # Bagli shell element listesi
    all_shell_eids = set()
    for info in connectivity.values():
        all_shell_eids.update(info.connected_quads.keys())
        all_shell_eids.update(info.connected_trias.keys())
    logger.info("  Toplam %d bagli shell element", len(all_shell_eids))

    # ============================================================
    # ADIM 3: Main H5'ten kuvvetleri oku
    # ============================================================
    logger.info("-" * 40)
    logger.info("ADIM 3: Main H5 okunuyor (%d dosya)...", len(main_h5_paths))

    bar_forces, shell_forces, h5_subcases = OP2Reader.read_multiple(
        op2_paths=main_h5_paths,
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
    # ADIM 6: Rapor olustur
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
        # Mevcut Main H5 verileriyle tahmin (correlation_results)
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
    print(f"  Main H5 dosya sayisi:  {len(main_h5_paths)}")
    print(f"  H5 bar kuvvet:         {len(bar_forces)}")
    print(f"  H5 shell flux:         {len(shell_forces)}")
    print(f"  H5 Joint Load satir:   {len(h5_reader.joint_load_cap)}")
    print(f"  Korelasyon sonucu:     {len(correlation_results)}")
    print(f"  Tahmin satiri:         {len(pred_df)}")
    print(f"\n  Excel cikti: {output_path}")
    print(f"  Tahmin CSV:  {csv_path}")


if __name__ == "__main__":
    main()
