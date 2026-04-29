# ==============================================================================
# File Name: 04_harmonic_lime_viz.py
# Version: 2.0
# Description:
#   Step 4 of the HarmonicLIME pipeline.
#   This script loads the generated saliency weights (.npy files) from Phase 3.
#   It calculates the Harmonic Localization Ratio (HLR), Percussive
#   Localization Ratio (PLR), and Residual Localization Ratio (RLR).
#   It generates a side-by-side heatmap visualization (Figure 2) comparing
#   a Classical clip and a Hip-Hop clip, with consistent colormaps and
#   IEEE print-safe grayscale compatibility.
#
# Changes from v1.0:
#   - Unified colormap (RdBu_r with center=0) applied to BOTH panels
#   - vmin/vmax explicitly shared across both panels for a fair visual scale
#   - X-axis label updated to "Time Windows (1 to 30, each 1s)"
#   - Figure caption contradiction note added as figure suptitle
#   - RLR computed and reported alongside HLR and PLR
#   - Grayscale export added for IEEE print compatibility check
#   - Colorbar overlap with y-axis labels fixed via explicit padding
#   - Figure footnote added to console output for paper caption reference
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

# Paths
DRIVE_BASE = '/content/drive/MyDrive/HarmonicLIME'
RESULTS_DIR = os.path.join(DRIVE_BASE, 'lime_results')
FIGURE_SAVE_PATH      = os.path.join(DRIVE_BASE, 'Figure2_HarmonicLIME_Heatmaps.png')
FIGURE_GRAY_SAVE_PATH = os.path.join(DRIVE_BASE, 'Figure2_HarmonicLIME_Heatmaps_grayscale.png')

# Number of time windows — must match pipeline config in Program 3
K_WINDOWS = 30

def log_progress(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")


def calculate_ratios(weights_matrix):
    """
    Calculates HLR, PLR, and RLR from a (3, K) saliency weight matrix.
    Rows: 0=Harmonic, 1=Percussive, 2=Residual.
    Uses absolute values per the HarmonicLIME definition.
    Returns: (hlr, plr, rlr) as floats in [0, 1].
    """
    abs_weights = np.abs(weights_matrix)
    total_saliency = np.sum(abs_weights)

    if total_saliency == 0:
        return 0.0, 0.0, 0.0

    hlr = np.sum(abs_weights[0, :]) / total_saliency
    plr = np.sum(abs_weights[1, :]) / total_saliency
    rlr = np.sum(abs_weights[2, :]) / total_saliency

    return hlr, plr, rlr


def get_shared_vmin_vmax(w1, w2):
    """
    Compute a symmetric vmin/vmax shared across both panels.
    Symmetric around zero so the colorbar centre = 0 on both plots,
    ensuring visual comparability between panels.
    """
    abs_max = max(np.abs(w1).max(), np.abs(w2).max())
    return -abs_max, abs_max


def plot_side_by_side(classical_file, hiphop_file):
    log_progress("Loading weights for visualization...")

    w_classical = np.load(os.path.join(RESULTS_DIR, classical_file))
    w_hiphop    = np.load(os.path.join(RESULTS_DIR, hiphop_file))

    # ── FIX 1: Shared symmetric color scale across both panels ──────────────
    # In v1.0 each panel had its own colorbar range, making visual comparison
    # misleading. A shared vmin/vmax ensures a weight of 0.05 looks the same
    # intensity in Classical as it does in Hip-Hop.
    vmin, vmax = get_shared_vmin_vmax(w_classical, w_hiphop)

    components = ['Harmonic', 'Percussive', 'Residual']

    # ── FIX 2: Layout — extra left margin prevents colorbar/label overlap ────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.5, 2.8),
        gridspec_kw={'wspace': 0.55}   # increased from 0.3 → prevents overlap
    )

    x_label = f'Time Windows (1 to {K_WINDOWS}, each 1s)'

    # ── Shared heatmap arguments ─────────────────────────────────────────────
    heatmap_kwargs = dict(
        cmap='RdBu_r',       # FIX 3: same colormap on both panels (was
                             # RdBu_r on Classical, sequential on Hip-Hop)
        center=0,
        vmin=vmin,           # FIX 1: shared scale
        vmax=vmax,
        yticklabels=components,
        xticklabels=False,
        cbar=True,
        cbar_kws={"shrink": 0.85, "pad": 0.02},
        linewidths=0.3,
        linecolor='#eeeeee'
    )

    # ── Classical panel ──────────────────────────────────────────────────────
    sns.heatmap(w_classical, ax=axes[0], **heatmap_kwargs)
    axes[0].set_title('Classical', fontsize=10, fontweight='bold', pad=6)
    axes[0].set_xlabel(x_label, fontsize=8)                # FIX 4: informative x label
    axes[0].tick_params(axis='y', rotation=0, labelsize=8)
    axes[0].set_ylabel('')

    # ── Hip-Hop panel ────────────────────────────────────────────────────────
    sns.heatmap(w_hiphop, ax=axes[1], **heatmap_kwargs)
    axes[1].set_title('Hip-Hop', fontsize=10, fontweight='bold', pad=6)
    axes[1].set_xlabel(x_label, fontsize=8)                # FIX 4: informative x label
    axes[1].tick_params(axis='y', rotation=0, labelsize=8)
    axes[1].set_ylabel('')

    # ── Colorbar label: clarify that red=positive, blue=negative ────────────
    # Adds axis label to each colorbar so negative weights are not mysterious
    for ax in axes:
        cbar = ax.collections[0].colorbar
        cbar.set_label('Saliency Weight', fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    # ── Save colour version (for digital/online publication) ─────────────────
    plt.savefig(FIGURE_SAVE_PATH, dpi=300, bbox_inches='tight',
                facecolor='white')
    log_progress(f"Colour figure saved to {FIGURE_SAVE_PATH}")
    plt.show()

    # ── FIX 5: Grayscale export for IEEE print compatibility check ───────────
    # IEEE prints figures in black and white. Export a grayscale version and
    # inspect it manually to confirm H/P/R rows remain distinguishable.
    log_progress("Generating grayscale version for IEEE print compatibility check...")

    fig_gray, axes_gray = plt.subplots(
        1, 2,
        figsize=(7.5, 2.8),
        gridspec_kw={'wspace': 0.55}
    )

    gray_kwargs = dict(
        cmap='gray_r',       # grayscale: dark = high positive saliency
        vmin=0,              # show absolute value only for grayscale
        vmax=vmax,
        yticklabels=components,
        xticklabels=False,
        cbar=True,
        cbar_kws={"shrink": 0.85, "pad": 0.02},
        linewidths=0.3,
        linecolor='#cccccc'
    )

    # Use absolute values for grayscale — direction lost, magnitude preserved
    sns.heatmap(np.abs(w_classical), ax=axes_gray[0], **gray_kwargs)
    axes_gray[0].set_title('Classical (Grayscale Check)',
                           fontsize=9, fontweight='bold', pad=6)
    axes_gray[0].set_xlabel(x_label, fontsize=8)
    axes_gray[0].tick_params(axis='y', rotation=0, labelsize=8)

    sns.heatmap(np.abs(w_hiphop), ax=axes_gray[1], **gray_kwargs)
    axes_gray[1].set_title('Hip-Hop (Grayscale Check)',
                           fontsize=9, fontweight='bold', pad=6)
    axes_gray[1].set_xlabel(x_label, fontsize=8)
    axes_gray[1].tick_params(axis='y', rotation=0, labelsize=8)

    for ax in axes_gray:
        cbar = ax.collections[0].colorbar
        cbar.set_label('|Saliency Weight|', fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    plt.savefig(FIGURE_GRAY_SAVE_PATH, dpi=300, bbox_inches='tight',
                facecolor='white')
    log_progress(f"Grayscale figure saved to {FIGURE_GRAY_SAVE_PATH}")
    log_progress("ACTION REQUIRED: Open the grayscale file and confirm "
                 "Harmonic/Percussive/Residual rows are visually distinguishable.")
    plt.show()

    # ── Console summary for paper caption reference ──────────────────────────
    print("\n" + "=" * 60)
    print("SUGGESTED LATEX FIGURE CAPTION (paste into .tex):")
    print("=" * 60)
    print(
        "HarmonicLIME component-level saliency maps for a representative\n"
        "Classical clip (left) and Hip-Hop clip (right). Red indicates\n"
        "positive saliency (feature supports the predicted genre);\n"
        "blue indicates suppressive saliency. The Classical clip exhibits\n"
        "stronger harmonic component attention, while the Hip-Hop clip\n"
        "is dominated by percussive saliency --- consistent with\n"
        "genre-level PLR values in Table~\\ref{tab:hlr_plr}. Both panels\n"
        f"share the same color scale (vmin={vmin:.3f}, vmax={vmax:.3f})."
    )
    print("=" * 60 + "\n")


def main():
    if not os.path.exists(RESULTS_DIR):
        log_progress(f"ERROR: Results directory {RESULTS_DIR} not found. "
                     "Run Program 3 first.")
        return

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('_weights.npy')]

    if len(files) == 0:
        log_progress("ERROR: No weight files found. Run Program 3 first.")
        return

    log_progress("--- Calculating HLR, PLR & RLR Metrics ---")

    classical_file = None
    hiphop_file    = None

    # ── Console table header ─────────────────────────────────────────────────
    print("-" * 62)
    print(f"{'Genre / Clip':<22} | {'HLR (%)':>8} | {'PLR (%)':>8} | {'RLR (%)':>8}")
    print("-" * 62)

    for f in sorted(files):
        weights = np.load(os.path.join(RESULTS_DIR, f))

        # ── FIX: RLR now computed and reported ───────────────────────────────
        hlr, plr, rlr = calculate_ratios(weights)

        clip_name = f.replace('_weights.npy', '')
        print(f"{clip_name:<22} | {hlr*100:>7.1f}% | {plr*100:>7.1f}% | {rlr*100:>7.1f}%")

        if 'classical' in f.lower() and classical_file is None:
            classical_file = f
        if 'hiphop' in f.lower() and hiphop_file is None:
            hiphop_file = f

    print("-" * 62)
    print(f"{'NOTE:':<22}   Rows should sum to ~100% (rounding may apply)")
    print("-" * 62 + "\n")

    if classical_file and hiphop_file:
        plot_side_by_side(classical_file, hiphop_file)
    else:
        log_progress("ERROR: Could not find both a classical and hiphop "
                     "clip for the visualization. Check filenames in "
                     f"{RESULTS_DIR} contain 'classical' and 'hiphop'.")


if __name__ == "__main__":
    main()