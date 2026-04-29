# ==============================================================================
# File Name: 05_deletion_auc_v2.py
# Version: 2.0 (Per-Clip CSV Saving for Statistical Analysis)
# Description:
#   Step 5 of the HarmonicLIME pipeline.
#   Calculates the Deletion AUC faithfulness metric for HarmonicLIME.
#   CHANGE FROM v1.1: Per-clip AUC values are now saved to
#   'gtzan_hl_auc_perclip.csv' on Drive for downstream statistical analysis
#   (SD computation, Wilcoxon signed-rank tests per R2 Major Issue 4).
# ==============================================================================
# GPU needed
# Input: lime_results/*_weights.npy (from script 03), GTZAN wav files, vggish_best_model.pth
# Output: gtzan_hl_auc_perclip.csv


pip_install('resampy')

import os
import sys
import csv
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
from sklearn.metrics import auc
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

# Hyperparameters
NUM_CLASSES = 10
SAMPLE_RATE = 16000
TARGET_LENGTH = int(9.6 * SAMPLE_RATE)
K_WINDOWS = 10

# Paths
LOCAL_EXTRACT_ROOT= '/content/gtzan_local'
DRIVE_BASE = '/content/drive/MyDrive/HarmonicLIME'
LOCAL_BASE = '/content/gtzan_local/Data/genres_original'
BEST_MODEL_PATH = os.path.join(DRIVE_BASE, 'vggish_best_model.pth')
RESULTS_DIR = '/content/lime_results_local'
# OUTPUT: per-clip CSV for statistics script
PERCLIP_CSV = os.path.join(DRIVE_BASE, 'gtzan_hl_auc_perclip.csv')

if not os.path.exists(LOCAL_EXTRACT_ROOT):
    print(f"ERROR: Folder '{LOCAL_EXTRACT_ROOT}' does not exist.")
    print("Please run '03_harmonic_lime_v3.py' first to restore the local files.")
    raise FileNotFoundError(f"Folder '{LOCAL_EXTRACT_ROOT}' not found")
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

def hpr_decomposition(y):
    y_h, y_p = librosa.effects.hpss(y, margin=1.0)
    y_r = y - y_h - y_p
    return y_h, y_p, y_r

def calculate_deletion_auc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress(f"Using device: {device}")

    model = VGGishGenreClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    preprocessor = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=True)
    preprocessor.eval()

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('_weights.npy')]
    if not files:
        log_progress("No weights found. Run Program 3 first.")
        return

    # ---- NEW: open CSV writer ----
    csv_rows = []  # list of dicts: {clip, genre, hl_auc}

    print("-" * 50)
    print(f"{'Clip':<20} | {'Genre':<12} | {'HL AUC':<10}")
    print("-" * 50)

    for f in sorted(files):
        clip_name = f.replace('_weights.npy', '')
        target_genre = clip_name.split('.')[0]
        target_idx = GENRE_TO_IDX[target_genre]

        weights = np.load(os.path.join(RESULTS_DIR, f))
        flat_weights = np.abs(weights.flatten())
        ranked_indices = np.argsort(flat_weights)[::-1]

        rel_path = os.path.join(target_genre, f"{clip_name}.wav")
        file_path = os.path.join(LOCAL_BASE, rel_path)

        try:
            y = load_and_preprocess_audio(file_path)
        except Exception:
            continue

        y_h, y_p, y_r = hpr_decomposition(y)
        window_samples = TARGET_LENGTH // K_WINDOWS
        probs_curve = []

        with torch.no_grad():
            for k in range(len(ranked_indices) + 1):
                z_mask = np.ones((3, K_WINDOWS))
                for i in range(k):
                    idx = ranked_indices[i]
                    z_mask[idx // K_WINDOWS, idx % K_WINDOWS] = 0

                y_tilde = np.zeros(TARGET_LENGTH)
                for w in range(K_WINDOWS):
                    start = w * window_samples
                    end = start + window_samples
                    y_tilde[start:end] = (z_mask[0, w] * y_h[start:end] +
                                          z_mask[1, w] * y_p[start:end] +
                                          z_mask[2, w] * y_r[start:end])

                mel_tensor = preprocessor._preprocess(y_tilde, SAMPLE_RATE)
                mel_tensor = mel_tensor.view(-1, 1, 96, 64).to(device)
                outputs = model(mel_tensor)
                outputs = outputs.mean(dim=0, keepdim=True)
                probs = torch.softmax(outputs, dim=1)[0]
                probs_curve.append(probs[target_idx].item())

        x_axis = np.linspace(0, 1, len(probs_curve))
        clip_auc = auc(x_axis, probs_curve)

        print(f"{clip_name:<20} | {target_genre:<12} | {clip_auc:.4f}")
        csv_rows.append({'clip': clip_name, 'genre': target_genre, 'hl_auc': clip_auc})

    print("-" * 50)

    # ---- NEW: save CSV ----
    with open(PERCLIP_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['clip', 'genre', 'hl_auc'])
        writer.writeheader()
        writer.writerows(csv_rows)
        os.sync()
    log_progress(f"Per-clip HarmonicLIME AUC saved to: {PERCLIP_CSV}")

    # Print genre-level summary for verification
    rows_arr = {}
    for row in csv_rows:
        rows_arr.setdefault(row['genre'], []).append(row['hl_auc'])
    print("\nGenre-level summary:")
    print(f"{'Genre':<12} | {'N':>4} | {'Mean':>8} | {'SD':>8}")
    print("-" * 40)
    all_vals = []
    for g in GENRES:
        vals = rows_arr.get(g, [])
        if vals:
            print(f"{g:<12} | {len(vals):>4} | {np.mean(vals):>8.4f} | {np.std(vals, ddof=1):>8.4f}")
            all_vals.extend(vals)
    print("-" * 40)
    print(f"{'Overall mean':<12} | {len(all_vals):>4} | {np.mean(all_vals):>8.4f} | {np.std(all_vals, ddof=1):>8.4f}")

if __name__ == "__main__":
    calculate_deletion_auc()
