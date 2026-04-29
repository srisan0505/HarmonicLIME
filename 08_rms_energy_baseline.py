# ==============================================================================
# File Name: 08_rms_energy_baseline.py
# Version: 1.0
# Description:
#   Step 8 of the HarmonicLIME pipeline.
#   Addresses R2 Minor Issue 2: computes per-component RMS energy ratios
#   (HER/PER/RER) for each GTZAN test clip and compares them against the
#   HarmonicLIME saliency ratios (HLR/PLR/RLR) from the saved weight files.
#   This separates "where signal energy is" from "where saliency is",
#   demonstrating that HarmonicLIME identifies discriminative content
#   rather than merely reflecting energy concentration.
#   Results are printed as a LaTeX-ready table for insertion into the paper.
#
# Hardware Requirements:
#   - CPU only (no GPU needed)
#   - RAM: < 2 GB
#   - Estimated runtime: < 5 minutes for 100 clips
#
# Dependencies:
#   - librosa, numpy, torchaudio
#   - Google Colab + Google Drive mount
#
# Inputs:
#   - /content/drive/MyDrive/HarmonicLIME/lime_results/lime_results.zip
#     (extracted to /content/lime_results_local/)
#   - /content/drive/MyDrive/datasets/GTZAN.zip
#     (extracted to /content/gtzan_local/)
#   - /content/drive/MyDrive/HarmonicLIME/dataset_splits.json
#
# Outputs:
#   - /content/drive/MyDrive/HarmonicLIME/gtzan_rms_vs_saliency.csv
#   - Printed genre-level table (mean HER/PER/RER vs HLR/PLR/RLR)
# ==============================================================================

import os
import json
import csv
import zipfile
import shutil
import numpy as np
import librosa
import torchaudio
import torch
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

# Paths
DRIVE_BASE        = '/content/drive/MyDrive/HarmonicLIME'
GTZAN_ZIP_PATH    = '/content/drive/MyDrive/datasets/GTZAN.zip'
LOCAL_BASE        = '/content/gtzan_local/Data/genres_original'
LOCAL_EXTRACT_ROOT= '/content/gtzan_local'
LOCAL_RESULTS_DIR = '/content/lime_results_local'
DRIVE_ZIP_PATH    = os.path.join(DRIVE_BASE, 'lime_results', 'lime_results.zip')
SPLIT_FILE        = os.path.join(DRIVE_BASE, 'dataset_splits.json')
OUTPUT_CSV        = os.path.join(DRIVE_BASE, 'gtzan_rms_vs_saliency.csv')

SAMPLE_RATE  = 16000
TARGET_LENGTH = int(9.6 * SAMPLE_RATE)
K_WINDOWS    = 10

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def ensure_gtzan_local():
    if os.path.exists(LOCAL_BASE):
        genre_dirs = [d for d in os.listdir(LOCAL_BASE)
                      if os.path.isdir(os.path.join(LOCAL_BASE, d))]
        if len(genre_dirs) >= 10:
            log(f"GTZAN already extracted.")
            return
    log("Extracting GTZAN from Drive...")
    shutil.copy2(GTZAN_ZIP_PATH, '/content/GTZAN.zip')
    os.makedirs(LOCAL_EXTRACT_ROOT, exist_ok=True)
    with zipfile.ZipFile('/content/GTZAN.zip', 'r') as zf:
        zf.extractall(LOCAL_EXTRACT_ROOT)
    os.remove('/content/GTZAN.zip')
    log("GTZAN ready.")

def ensure_weights_local():
    existing = [f for f in os.listdir(LOCAL_RESULTS_DIR)
                if f.endswith('_weights.npy')] if os.path.exists(LOCAL_RESULTS_DIR) else []
    if len(existing) >= 100:
        log(f"Weight files already in local scratch ({len(existing)} files).")
        return
    log("Restoring weight files from Drive zip...")
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as zf:
        zf.extractall(LOCAL_RESULTS_DIR)
    log(f"Restored {len(os.listdir(LOCAL_RESULTS_DIR))} weight files.")

def load_audio(file_path):
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
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.shape[1]))
    return waveform.squeeze(0).numpy()

def rms_ratios(y_h, y_p, y_r):
    """Compute RMS energy fraction for each HPR component."""
    e_h = np.sqrt(np.mean(y_h ** 2))
    e_p = np.sqrt(np.mean(y_p ** 2))
    e_r = np.sqrt(np.mean(y_r ** 2))
    total = e_h + e_p + e_r
    if total == 0:
        return 0, 0, 0
    return e_h / total, e_p / total, e_r / total

def saliency_ratios(weights):
    """Compute HLR/PLR/RLR from saved (3, K) weight array."""
    abs_w = np.abs(weights)
    total = np.sum(abs_w)
    if total == 0:
        return 0, 0, 0
    return (np.sum(abs_w[0]) / total,
            np.sum(abs_w[1]) / total,
            np.sum(abs_w[2]) / total)

def run():
    ensure_gtzan_local()
    ensure_weights_local()

    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)

    csv_rows = []

    for rel_path in splits['test']:
        clip_name = os.path.basename(rel_path).replace('.wav','').replace('.au','')
        genre = rel_path.split('/')[0] if '/' in rel_path else rel_path.split('\\')[0]
        if genre not in GENRES:
            continue

        weight_path = os.path.join(LOCAL_RESULTS_DIR, f"{clip_name}_weights.npy")
        wav_path    = os.path.join(LOCAL_BASE, rel_path)

        if not os.path.exists(weight_path) or not os.path.exists(wav_path):
            log(f"Missing file for {clip_name}, skipping.")
            continue

        try:
            y = load_audio(wav_path)
            y_h, y_p = librosa.effects.hpss(y, margin=1.0)
            y_r = y - y_h - y_p
            her, per, rer = rms_ratios(y_h, y_p, y_r)

            weights = np.load(weight_path)
            hlr, plr, rlr = saliency_ratios(weights)

            csv_rows.append({
                'clip': clip_name, 'genre': genre,
                'her': her, 'per': per, 'rer': rer,
                'hlr': hlr, 'plr': plr, 'rlr': rlr
            })
        except Exception as e:
            log(f"ERROR on {clip_name}: {e}")

    # Save CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['clip','genre',
                                               'her','per','rer',
                                               'hlr','plr','rlr'])
        writer.writeheader()
        writer.writerows(csv_rows)
        os.sync()
    log(f"Per-clip CSV saved to {OUTPUT_CSV}")

    # Genre-level summary
    print("\n--- RMS Energy vs Saliency Ratios by Genre ---")
    print(f"{'Genre':<12} | {'HER%':>6} {'PER%':>6} {'RER%':>6} || "
          f"{'HLR%':>6} {'PLR%':>6} {'RLR%':>6}")
    print("-" * 62)

    for g in GENRES:
        rows = [r for r in csv_rows if r['genre'] == g]
        if not rows:
            continue
        her = np.mean([r['her'] for r in rows]) * 100
        per = np.mean([r['per'] for r in rows]) * 100
        rer = np.mean([r['rer'] for r in rows]) * 100
        hlr = np.mean([r['hlr'] for r in rows]) * 100
        plr = np.mean([r['plr'] for r in rows]) * 100
        rlr = np.mean([r['rlr'] for r in rows]) * 100
        print(f"{g:<12} | {her:>6.1f} {per:>6.1f} {rer:>6.1f} || "
              f"{hlr:>6.1f} {plr:>6.1f} {rlr:>6.1f}")

    print("\nLaTeX snippet for paper (add as footnote or inline comparison):")
    print("Compare HER/PER columns against HLR/PLR in Table II to verify")
    print("saliency diverges from raw energy distribution.")

if __name__ == "__main__":
    run()
