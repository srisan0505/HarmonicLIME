# ==============================================================================
# File Name: FMA_03_evaluate_v2.py
# Version: 2.0 (Per-Clip CSV Saving for Statistical Analysis)
# Description:
#   Computes HLR/PLR/RLR and Deletion AUC for HarmonicLIME, Mel-LIME, and SHAP
#   across the FMA-small zero-shot dataset.
#   CHANGE FROM v1.1: Per-clip metrics are now saved to
#   'fma_perclip_metrics.csv' on Drive for downstream statistical analysis
#   (SD computation, Wilcoxon signed-rank tests per R2 Major Issue 4).
#   run !pip install resampy before running this program
# ==============================================================================
# GPU needed
# Input: fma_lime_results/*_weights.npy (from FMA_02), fma_processed/ wav files, vggish_best_model.pth
# Output: fma_perclip_metrics.csv



import os
import csv
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
from sklearn.linear_model import Ridge
from sklearn.metrics import auc, pairwise_distances
from datetime import datetime
import warnings

NUM_CLASSES = 10
SAMPLE_RATE = 16000
TARGET_LENGTH = int(9.6 * SAMPLE_RATE)
K_WINDOWS = 10
GRID_H, GRID_W = 8, 8
ALPHA = 0.01

DRIVE_BASE = '/content/drive/MyDrive/HarmonicLIME'
FMA_PROCESSED_DIR = os.path.join(DRIVE_BASE, 'fma_processed')
BEST_MODEL_PATH = os.path.join(DRIVE_BASE, 'vggish_best_model.pth')
FMA_RESULTS_DIR = os.path.join(DRIVE_BASE, 'fma_lime_results')
# OUTPUT: per-clip CSV for statistics script
PERCLIP_CSV = os.path.join(DRIVE_BASE, 'fma_perclip_metrics.csv')

FMA_GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
              'Instrumental', 'International', 'Pop', 'Rock']

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

class VGGishGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish',
                                     preprocess=False, postprocess=False)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.vggish(x))

def apply_grid_mask(mel_tensor, binary_mask):
    masked_mel = mel_tensor.clone()
    fp, mp = 96 // GRID_H, 64 // GRID_W
    for i in range(GRID_H):
        for j in range(GRID_W):
            if binary_mask[i * GRID_W + j] == 0:
                masked_mel[:, :, i*fp:(i+1)*fp, j*mp:(j+1)*mp] = 0.0
    return masked_mel

def run_fma_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGishGenreClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    preprocessor = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=True)
    preprocessor.eval()

    files = [f for f in os.listdir(FMA_RESULTS_DIR) if f.endswith('_weights.npy')]
    log(f"Found {len(files)} weight files.")

    # ---- NEW: CSV accumulator ----
    csv_rows = []

    log("Starting FMA evaluation...")

    for f in sorted(files):
        genre = f.split('_')[0]
        clip_name = f.replace(f"{genre}_", "").replace('_weights.npy', '.wav')

        weights = np.load(os.path.join(FMA_RESULTS_DIR, f))
        abs_w = np.abs(weights)
        tot = np.sum(abs_w)

        hlr = np.sum(abs_w[0, :]) / tot if tot > 0 else 0
        plr = np.sum(abs_w[1, :]) / tot if tot > 0 else 0
        rlr = np.sum(abs_w[2, :]) / tot if tot > 0 else 0

        wav_path = os.path.join(FMA_PROCESSED_DIR, genre, clip_name)
        if not os.path.exists(wav_path):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = torchaudio.load(wav_path)

        y = y.squeeze(0).numpy()

        with torch.no_grad():
            base_mel = preprocessor._preprocess(y, SAMPLE_RATE).view(-1, 1, 96, 64).to(device)
            target_idx = torch.argmax(
                model(base_mel).mean(dim=0, keepdim=True), dim=1).item()

        # ---- HarmonicLIME Deletion AUC ----
        y_h, y_p = librosa.effects.hpss(y, margin=1.0)
        y_r = y - y_h - y_p
        ranked_hl = np.argsort(abs_w.flatten())[::-1]
        ws = TARGET_LENGTH // K_WINDOWS

        hl_curve = []
        with torch.no_grad():
            for k in range(31):
                z_mask = np.ones((3, K_WINDOWS))
                for i in range(k):
                    idx = ranked_hl[i]
                    z_mask[idx // K_WINDOWS, idx % K_WINDOWS] = 0
                y_tilde = np.zeros(TARGET_LENGTH)
                for w in range(K_WINDOWS):
                    start, end = w * ws, (w + 1) * ws
                    y_tilde[start:end] = (z_mask[0, w] * y_h[start:end] +
                                          z_mask[1, w] * y_p[start:end] +
                                          z_mask[2, w] * y_r[start:end])
                mel = preprocessor._preprocess(y_tilde, SAMPLE_RATE).view(
                    -1, 1, 96, 64).to(device)
                prob = torch.softmax(
                    model(mel).mean(dim=0, keepdim=True), dim=1)[0, target_idx].item()
                hl_curve.append(prob)
        hl_auc = auc(np.linspace(0, 1, 31), hl_curve)

        # ---- Mel-LIME ----
        num_patches = 64
        z_vecs = np.random.binomial(1, 0.5, size=(100, num_patches))
        z_vecs[0, :] = 1
        p_probs = []
        with torch.no_grad():
            for i in range(100):
                m_mel = apply_grid_mask(base_mel, z_vecs[i])
                p_probs.append(torch.softmax(
                    model(m_mel).mean(dim=0, keepdim=True), dim=1)[0, target_idx].item())
        solv = Ridge(alpha=ALPHA).fit(z_vecs, p_probs)
        ranked_ml = np.argsort(np.abs(solv.coef_))[::-1]

        # ---- SHAP ----
        base_mel_grad = base_mel.detach().clone().requires_grad_(True)
        model.zero_grad()
        model(base_mel_grad).mean(dim=0, keepdim=True)[0, target_idx].backward()
        grads = base_mel_grad.grad.detach().abs()
        shap_w = np.zeros(num_patches)
        for i in range(GRID_H):
            for j in range(GRID_W):
                shap_w[i * GRID_W + j] = grads[
                    :, :, i*12:(i+1)*12, j*8:(j+1)*8].sum().item()
        ranked_sh = np.argsort(shap_w)[::-1]

        def calc_baseline_auc(ranked):
            curve = []
            with torch.no_grad():
                for k in range(65):
                    zm = np.ones(64)
                    for i in range(k):
                        zm[ranked[i]] = 0
                    m_mel = apply_grid_mask(base_mel, zm)
                    curve.append(torch.softmax(
                        model(m_mel).mean(dim=0, keepdim=True), dim=1)[0, target_idx].item())
            return auc(np.linspace(0, 1, 65), curve)

        ml_auc = calc_baseline_auc(ranked_ml)
        sh_auc = calc_baseline_auc(ranked_sh)

        clip_id = clip_name.replace('.wav', '')
        csv_rows.append({
            'clip': clip_id,
            'genre': genre,
            'hlr': hlr,
            'plr': plr,
            'rlr': rlr,
            'hl_auc': hl_auc,
            'ml_auc': ml_auc,
            'sh_auc': sh_auc
        })

    # ---- NEW: save CSV ----
    fieldnames = ['clip', 'genre', 'hlr', 'plr', 'rlr', 'hl_auc', 'ml_auc', 'sh_auc']
    with open(PERCLIP_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        os.sync()
    log(f"Per-clip metrics saved to: {PERCLIP_CSV}")

    # Genre-level summary table
    print("\n--- FMA-Small Generalization Metrics ---")
    print(f"{'Genre':<15} | {'HLR%':>5} | {'PLR%':>5} | {'RLR%':>5} || "
          f"{'HL-AUC':>7} | {'HL-SD':>7} | {'ML-AUC':>7} | {'ML-SD':>7} | "
          f"{'SH-AUC':>7} | {'SH-SD':>7}")
    print("-" * 95)

    for g in FMA_GENRES:
        rows = [r for r in csv_rows if r['genre'] == g]
        if not rows:
            continue
        hlr_vals = [r['hlr'] for r in rows]
        plr_vals = [r['plr'] for r in rows]
        rlr_vals = [r['rlr'] for r in rows]
        hl_vals = [r['hl_auc'] for r in rows]
        ml_vals = [r['ml_auc'] for r in rows]
        sh_vals = [r['sh_auc'] for r in rows]
        print(f"{g:<15} | {np.mean(hlr_vals)*100:>5.1f} | {np.mean(plr_vals)*100:>5.1f} | "
              f"{np.mean(rlr_vals)*100:>5.1f} || "
              f"{np.mean(hl_vals):>7.4f} | {np.std(hl_vals, ddof=1):>7.4f} | "
              f"{np.mean(ml_vals):>7.4f} | {np.std(ml_vals, ddof=1):>7.4f} | "
              f"{np.mean(sh_vals):>7.4f} | {np.std(sh_vals, ddof=1):>7.4f}")

if __name__ == "__main__":
    run_fma_eval()
