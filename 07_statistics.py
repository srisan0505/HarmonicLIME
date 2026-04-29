# ==============================================================================
# File Name: 07_statistics.py
# Version: 1.0
# Description:
#   Step 7 of the HarmonicLIME pipeline (new script).
#   Loads per-clip CSVs produced by v2 scripts and computes:
#     1. Genre-level means ± SD for Tables I and III (GTZAN)
#     2. Genre-level means ± SD for Tables III and IV (FMA-small)
#     3. Wilcoxon signed-rank tests (HarmonicLIME vs Mel-LIME, HL vs SHAP)
#        per genre and overall, for both datasets.
#   Output is formatted for direct insertion into LaTeX tables.
#
#   Addresses R2 Major Issue 4: statistical rigor.
#
#   Run AFTER:
#     05_deletion_auc_v2.py  -> gtzan_hl_auc_perclip.csv
#     06_baseline_auc_v2.py  -> gtzan_baseline_auc_perclip.csv
#     FMA_03_evaluate_v2.py  -> fma_perclip_metrics.csv
# ==============================================================================
# GPU Not needed
# Input: the three CSVs generated in 05, 06 and FMA_03  (all v2)
# Output: printed tables (mean ± SD) and Wilcoxon results — copy-paste into LaTeX

import os
import csv
import numpy as np
from scipy import stats
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = '/content/drive/MyDrive/HarmonicLIME'
GTZAN_HL_CSV   = os.path.join(DRIVE_BASE, 'gtzan_hl_auc_perclip.csv')
GTZAN_BASE_CSV = os.path.join(DRIVE_BASE, 'gtzan_baseline_auc_perclip.csv')
FMA_CSV        = os.path.join(DRIVE_BASE, 'fma_perclip_metrics.csv')

GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']
FMA_GENRES   = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
                'Instrumental', 'International', 'Pop', 'Rock']

def load_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def wilcoxon_str(a, b):
    """Return formatted Wilcoxon result string, or 'n/a' if too few samples."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 5:
        return "n/a (n<5)"
    try:
        stat, p = stats.wilcoxon(a, b, alternative='less')  # HL < baseline (lower is better)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        return f"W={stat:.1f}, p={p:.4f} {sig}"
    except Exception as e:
        return f"err: {e}"

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def analyze_gtzan():
    print_section("GTZAN — Deletion AUC with SD (Table I revision)")

    hl_rows  = load_csv(GTZAN_HL_CSV)
    b_rows   = load_csv(GTZAN_BASE_CSV)

    # Index by clip name
    hl_by_clip  = {r['clip']: float(r['hl_auc'])       for r in hl_rows}
    ml_by_clip  = {r['clip']: float(r['mel_lime_auc']) for r in b_rows}
    sh_by_clip  = {r['clip']: float(r['shap_auc'])     for r in b_rows}

    # Merge on shared clips
    shared_clips = sorted(set(hl_by_clip) & set(ml_by_clip))
    if not shared_clips:
        print("ERROR: No shared clips between HL and baseline CSVs. "
              "Check that v2 scripts ran over the same file set.")
        return

    genre_data = {g: {'hl': [], 'ml': [], 'sh': []} for g in GTZAN_GENRES}
    for clip in shared_clips:
        g = clip.split('.')[0]
        if g not in genre_data:
            continue
        genre_data[g]['hl'].append(hl_by_clip[clip])
        genre_data[g]['ml'].append(ml_by_clip[clip])
        genre_data[g]['sh'].append(sh_by_clip[clip])

    # --- Table I: Means ± SD ---
    print(f"\n{'Genre':<12} | {'Mel-LIME':>16} | {'SHAP':>16} | {'HarmonicLIME':>16}")
    print("-" * 68)

    all_hl, all_ml, all_sh = [], [], []
    all_hl_ex, all_ml_ex, all_sh_ex = [], [], []

    for g in GTZAN_GENRES:
        hl = genre_data[g]['hl']
        ml = genre_data[g]['ml']
        sh = genre_data[g]['sh']
        if not hl:
            continue
        m_hl = np.mean(hl); s_hl = np.std(hl, ddof=1)
        m_ml = np.mean(ml); s_ml = np.std(ml, ddof=1)
        m_sh = np.mean(sh); s_sh = np.std(sh, ddof=1)
        print(f"{g:<12} | {m_ml:>7.4f}±{s_ml:<6.4f} | {m_sh:>7.4f}±{s_sh:<6.4f} | "
              f"{m_hl:>7.4f}±{s_hl:<6.4f}")
        all_hl.extend(hl); all_ml.extend(ml); all_sh.extend(sh)
        if g != 'classical':
            all_hl_ex.extend(hl); all_ml_ex.extend(ml); all_sh_ex.extend(sh)

    print("-" * 68)
    print(f"{'Mean (all)':<12} | {np.mean(all_ml):>7.4f}±{np.std(all_ml,ddof=1):<6.4f} | "
          f"{np.mean(all_sh):>7.4f}±{np.std(all_sh,ddof=1):<6.4f} | "
          f"{np.mean(all_hl):>7.4f}±{np.std(all_hl,ddof=1):<6.4f}")
    print(f"{'Mean (ex.Cl.)':<12} | {np.mean(all_ml_ex):>7.4f}±{np.std(all_ml_ex,ddof=1):<6.4f} | "
          f"{np.mean(all_sh_ex):>7.4f}±{np.std(all_sh_ex,ddof=1):<6.4f} | "
          f"{np.mean(all_hl_ex):>7.4f}±{np.std(all_hl_ex,ddof=1):<6.4f}")

    # --- Wilcoxon tests ---
    print_section("GTZAN — Wilcoxon Signed-Rank Tests (HL vs baselines, alternative: HL < baseline)")
    print("Note: 'less' tests whether HarmonicLIME produces significantly lower AUC.")
    print()

    for g in GTZAN_GENRES:
        hl = genre_data[g]['hl']
        ml = genre_data[g]['ml']
        sh = genre_data[g]['sh']
        if not hl:
            continue
        print(f"  {g:<12}  HL vs Mel-LIME: {wilcoxon_str(hl, ml)}")
        print(f"  {g:<12}  HL vs SHAP:     {wilcoxon_str(hl, sh)}")

    print()
    print(f"  {'Overall (ex.Cl.)':<16}  HL vs Mel-LIME: {wilcoxon_str(all_hl_ex, all_ml_ex)}")
    print(f"  {'Overall (ex.Cl.)':<16}  HL vs SHAP:     {wilcoxon_str(all_hl_ex, all_sh_ex)}")
    print(f"  {'Overall (all)':<16}     HL vs Mel-LIME: {wilcoxon_str(all_hl, all_ml)}")
    print(f"  {'Overall (all)':<16}     HL vs SHAP:     {wilcoxon_str(all_hl, all_sh)}")

    # --- Count wins correctly ---
    wins_vs_ml = sum(1 for g in GTZAN_GENRES
                     if genre_data[g]['hl'] and
                     np.mean(genre_data[g]['hl']) < np.mean(genre_data[g]['ml']))
    wins_vs_sh = sum(1 for g in GTZAN_GENRES
                     if genre_data[g]['hl'] and
                     np.mean(genre_data[g]['hl']) < np.mean(genre_data[g]['sh']))
    print(f"\n  Genre wins vs Mel-LIME: {wins_vs_ml}/10")
    print(f"  Genre wins vs SHAP:     {wins_vs_sh}/10")
    print("  (Use these corrected counts in abstract and Table I caption)")


def analyze_fma():
    print_section("FMA-small — Deletion AUC with SD (Table III revision)")

    rows = load_csv(FMA_CSV)
    genre_data = {g: {'hl': [], 'ml': [], 'sh': []} for g in FMA_GENRES}

    for r in rows:
        g = r['genre']
        if g not in genre_data:
            continue
        genre_data[g]['hl'].append(float(r['hl_auc']))
        genre_data[g]['ml'].append(float(r['ml_auc']))
        genre_data[g]['sh'].append(float(r['sh_auc']))

    print(f"\n{'Genre':<15} | {'Mel-LIME':>16} | {'SHAP':>16} | {'HarmonicLIME':>16}")
    print("-" * 72)

    all_hl, all_ml, all_sh = [], [], []

    for g in FMA_GENRES:
        hl = genre_data[g]['hl']
        ml = genre_data[g]['ml']
        sh = genre_data[g]['sh']
        if not hl:
            continue
        m_hl = np.mean(hl); s_hl = np.std(hl, ddof=1)
        m_ml = np.mean(ml); s_ml = np.std(ml, ddof=1)
        m_sh = np.mean(sh); s_sh = np.std(sh, ddof=1)
        print(f"{g:<15} | {m_ml:>7.4f}±{s_ml:<6.4f} | {m_sh:>7.4f}±{s_sh:<6.4f} | "
              f"{m_hl:>7.4f}±{s_hl:<6.4f}")
        all_hl.extend(hl); all_ml.extend(ml); all_sh.extend(sh)

    print("-" * 72)
    print(f"{'Mean':<15} | {np.mean(all_ml):>7.4f}±{np.std(all_ml,ddof=1):<6.4f} | "
          f"{np.mean(all_sh):>7.4f}±{np.std(all_sh,ddof=1):<6.4f} | "
          f"{np.mean(all_hl):>7.4f}±{np.std(all_hl,ddof=1):<6.4f}")

    print_section("FMA-small — Wilcoxon Signed-Rank Tests")
    for g in FMA_GENRES:
        hl = genre_data[g]['hl']
        ml = genre_data[g]['ml']
        sh = genre_data[g]['sh']
        if not hl:
            continue
        print(f"  {g:<15}  HL vs Mel-LIME: {wilcoxon_str(hl, ml)}")
        print(f"  {g:<15}  HL vs SHAP:     {wilcoxon_str(hl, sh)}")

    print()
    print(f"  {'Overall':<15}  HL vs Mel-LIME: {wilcoxon_str(all_hl, all_ml)}")
    print(f"  {'Overall':<15}  HL vs SHAP:     {wilcoxon_str(all_hl, all_sh)}")

    wins_vs_ml = sum(1 for g in FMA_GENRES
                     if genre_data[g]['hl'] and
                     np.mean(genre_data[g]['hl']) < np.mean(genre_data[g]['ml']))
    print(f"\n  Genre wins vs Mel-LIME: {wins_vs_ml}/8")


def analyze_fma_hlr():
    print_section("FMA-small — HLR/PLR/RLR with SD (Table IV revision)")

    rows = load_csv(FMA_CSV)
    genre_data = {g: {'hlr': [], 'plr': [], 'rlr': []} for g in FMA_GENRES}
    for r in rows:
        g = r['genre']
        if g not in genre_data:
            continue
        genre_data[g]['hlr'].append(float(r['hlr']))
        genre_data[g]['plr'].append(float(r['plr']))
        genre_data[g]['rlr'].append(float(r['rlr']))

    print(f"\n{'Genre':<15} | {'HLR (%)':>14} | {'PLR (%)':>14} | {'RLR (%)':>14}")
    print("-" * 64)
    for g in FMA_GENRES:
        hlr = genre_data[g]['hlr']
        plr = genre_data[g]['plr']
        rlr = genre_data[g]['rlr']
        if not hlr:
            continue
        print(f"{g:<15} | {np.mean(hlr)*100:>6.1f}±{np.std(hlr,ddof=1)*100:<5.1f} | "
              f"{np.mean(plr)*100:>6.1f}±{np.std(plr,ddof=1)*100:<5.1f} | "
              f"{np.mean(rlr)*100:>6.1f}±{np.std(rlr,ddof=1)*100:<5.1f}")


if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running statistical analysis...")
    analyze_gtzan()
    analyze_fma()
    analyze_fma_hlr()
    print("\nDone. Copy SD values and Wilcoxon results into LaTeX tables.")
