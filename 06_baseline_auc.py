# ==============================================================================
# File Name: 06_baseline_auc_v2.py
# Version: 2.0 (Per-Clip CSV Saving for Statistical Analysis)
# Description:
#   Step 6 of the HarmonicLIME pipeline.
#   Computes Deletion AUC for Mel-LIME and SHAP baselines.
#   CHANGE FROM v1.0: Per-clip AUC values are now saved to
#   'gtzan_baseline_auc_perclip.csv' on Drive for downstream statistical
#   analysis (SD computation, Wilcoxon signed-rank tests per R2 Major Issue 4).
# ==============================================================================
# GPU needed
# Input: lime_results/*_weights.npy (used only to identify which clips to process), GTZAN wav files, vggish_best_model.pth
# Output: gtzan_baseline_auc_perclip.csv


import os
import csv
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import auc, pairwise_distances
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

# Hyperparameters
NUM_CLASSES = 10
SAMPLE_RATE = 16000
TARGET_LENGTH = int(9.6 * SAMPLE_RATE)
N_SAMPLES = 300
ALPHA = 0.01
GRID_H = 8
GRID_W = 8

# Paths
DRIVE_BASE = '/content/drive/MyDrive/HarmonicLIME'
LOCAL_BASE = '/content/gtzan_local/Data/genres_original'
BEST_MODEL_PATH = os.path.join(DRIVE_BASE, 'vggish_best_model.pth')
RESULTS_DIR = '/content/lime_results_local'
# OUTPUT: per-clip CSV for statistics script
PERCLIP_CSV = os.path.join(DRIVE_BASE, 'gtzan_baseline_auc_perclip.csv')

if not os.path.exists(LOCAL_EXTRACT_ROOT):
    print(f"ERROR: Folder '{LOCAL_EXTRACT_ROOT}' does not exist.")
    print("Please run '03_harmonic_lime_v3.py' to restore the local files to be displayed.")
    sys.exit(1)  # if in a script
else:
    print(f"Folder '{LOCAL_EXTRACT_ROOT}' exists. Proceeding...")


GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

def log_progress(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

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
    frames_per_patch = 96 // GRID_H
    mels_per_patch = 64 // GRID_W
    for i in range(GRID_H):
        for j in range(GRID_W):
            patch_idx = i * GRID_W + j
            if binary_mask[patch_idx] == 0:
                sf = i * frames_per_patch
                ef = sf + frames_per_patch
                sm = j * mels_per_patch
                em = sm + mels_per_patch
                masked_mel[:, :, sf:ef, sm:em] = 0.0
    return masked_mel

def load_and_preprocess_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    if waveform.shape[1] > TARGET_LENGTH:
        start = (waveform.shape[1] - TARGET_LENGTH) // 2
        waveform = waveform[:, start:start + TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    return waveform.squeeze(0).numpy()

def run_baselines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress(f"Using device: {device}")

    model = VGGishGenreClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    preprocessor = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=True)
    preprocessor.eval()

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('_weights.npy')]
    if not files:
        log_progress("No clips found. Run Phase 3 first.")
        return

    # ---- NEW: CSV accumulator ----
    csv_rows = []

    print("-" * 65)
    print(f"{'Clip':<20} | {'Genre':<10} | {'Mel-LIME':>8} | {'SHAP':>8}")
    print("-" * 65)

    num_patches = GRID_H * GRID_W

    for f in sorted(files):
        clip_name = f.replace('_weights.npy', '')
        target_genre = clip_name.split('.')[0]
        target_idx = GENRE_TO_IDX[target_genre]

        rel_path = os.path.join(target_genre, f"{clip_name}.wav")
        file_path = os.path.join(LOCAL_BASE, rel_path)

        try:
            y = load_and_preprocess_audio(file_path)
        except Exception:
            continue

        with torch.no_grad():
            base_mel = preprocessor._preprocess(y, SAMPLE_RATE)
            base_mel = base_mel.view(-1, 1, 96, 64).to(device)

        # ---- Mel-LIME surrogate ----
        model.eval()
        z_vectors = np.random.binomial(1, 0.5, size=(N_SAMPLES, num_patches))
        z_vectors[0, :] = 1

        perturbed_probs = []
        with torch.no_grad():
            for i in range(N_SAMPLES):
                masked_mel = apply_grid_mask(base_mel, z_vectors[i])
                outputs = model(masked_mel).mean(dim=0, keepdim=True)
                prob = torch.softmax(outputs, dim=1)[0, target_idx].item()
                perturbed_probs.append(prob)

        distances = pairwise_distances(z_vectors, z_vectors[0].reshape(1, -1),
                                       metric='cosine').ravel()
        weights = np.sqrt(np.exp(-(distances ** 2) / 0.25 ** 2))
        solver = Ridge(alpha=ALPHA, fit_intercept=True)
        solver.fit(z_vectors, perturbed_probs, sample_weight=weights)
        mel_lime_ranked = np.argsort(np.abs(solver.coef_))[::-1]

        # ---- SHAP (Gradient x Input) ----
        model.eval()
        base_mel_grad = base_mel.detach().clone().requires_grad_(True)
        outputs = model(base_mel_grad).mean(dim=0, keepdim=True)
        model.zero_grad()
        outputs[0, target_idx].backward()
        gradients = base_mel_grad.grad.detach().abs()
        shap_weights = np.zeros(num_patches)
        fpp = 96 // GRID_H
        mpp = 64 // GRID_W
        for i in range(GRID_H):
            for j in range(GRID_W):
                shap_weights[i * GRID_W + j] = gradients[
                    :, :, i*fpp:(i+1)*fpp, j*mpp:(j+1)*mpp].sum().item()
        shap_ranked = np.argsort(shap_weights)[::-1]

        # ---- Deletion AUC for both ----
        def compute_auc_for_ranking(ranked_indices):
            probs_curve = []
            with torch.no_grad():
                for k in range(num_patches + 1):
                    z_mask = np.ones(num_patches)
                    for i in range(k):
                        z_mask[ranked_indices[i]] = 0
                    masked_mel = apply_grid_mask(base_mel, z_mask)
                    outputs = model(masked_mel).mean(dim=0, keepdim=True)
                    prob = torch.softmax(outputs, dim=1)[0, target_idx].item()
                    probs_curve.append(prob)
            x_axis = np.linspace(0, 1, len(probs_curve))
            return auc(x_axis, probs_curve)

        ml_auc = compute_auc_for_ranking(mel_lime_ranked)
        sh_auc = compute_auc_for_ranking(shap_ranked)

        print(f"{clip_name:<20} | {target_genre:<10} | {ml_auc:>8.4f} | {sh_auc:>8.4f}")
        csv_rows.append({
            'clip': clip_name,
            'genre': target_genre,
            'mel_lime_auc': ml_auc,
            'shap_auc': sh_auc
        })

    print("-" * 65)

    # ---- NEW: save CSV ----
    with open(PERCLIP_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['clip', 'genre', 'mel_lime_auc', 'shap_auc'])
        writer.writeheader()
        writer.writerows(csv_rows)
        os.sync()
    log_progress(f"Per-clip baseline AUC saved to: {PERCLIP_CSV}")

    # Genre-level summary
    ml_by_genre = {}
    sh_by_genre = {}
    for row in csv_rows:
        ml_by_genre.setdefault(row['genre'], []).append(row['mel_lime_auc'])
        sh_by_genre.setdefault(row['genre'], []).append(row['shap_auc'])

    print("\nGenre-level summary:")
    print(f"{'Genre':<12} | {'ML Mean':>8} | {'ML SD':>8} | {'SH Mean':>8} | {'SH SD':>8}")
    print("-" * 56)
    for g in GENRES:
        ml = ml_by_genre.get(g, [])
        sh = sh_by_genre.get(g, [])
        if ml:
            print(f"{g:<12} | {np.mean(ml):>8.4f} | {np.std(ml, ddof=1):>8.4f} | "
                  f"{np.mean(sh):>8.4f} | {np.std(sh, ddof=1):>8.4f}")

if __name__ == "__main__":
    run_baselines()
