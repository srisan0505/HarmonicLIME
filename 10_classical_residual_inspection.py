# ==============================================================================
# File Name: 10_classical_residual_inspection.py
# Version: 1.0
# Description:
#   Step 10 of the HarmonicLIME pipeline.
#   Addresses R2 Major Issue 5: provides direct spectral evidence for the
#   Clever Hans interpretation of the Classical genre anomaly.
#   For each Classical test clip, this script:
#     1. Decomposes the clip into H/P/R components via HPSS
#     2. Computes the residual spectrogram and measures spectral flatness,
#        high-frequency energy ratio (>4kHz), and temporal variance —
#        all proxies for tape hiss, room reverberation, and recording-era
#        production artifacts
#     3. Compares these residual characteristics against a control genre
#        (Country) which has normal Deletion AUC behaviour
#     4. Generates and saves a figure showing mean residual spectrograms
#        for Classical vs Country (Figure-ready for paper)
#   Results provide the spectral evidence required to substantiate the
#   Clever Hans claim without denoising/dereverberation ablation.
#
# Hardware Requirements:
#   - CPU only (no GPU needed)
#   - RAM: < 2 GB
#   - Estimated runtime: < 3 minutes
#
# Dependencies:
#   - librosa, numpy, matplotlib, scipy, torchaudio
#   - Google Colab + Google Drive mount
#
# Inputs:
#   - /content/drive/MyDrive/datasets/GTZAN.zip
#     (extracted to /content/gtzan_local/)
#   - /content/drive/MyDrive/HarmonicLIME/dataset_splits.json
#
# Outputs:
#   - /content/drive/MyDrive/HarmonicLIME/classical_residual_analysis.csv
#   - /content/drive/MyDrive/HarmonicLIME/classical_residual_figure.pdf
#   - Printed genre-level comparison table
# ==============================================================================

import os
import csv
import json
import zipfile
import shutil
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchaudio
import torch
from scipy.stats import mannwhitneyu
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

# Paths
DRIVE_BASE         = '/content/drive/MyDrive/HarmonicLIME'
GTZAN_ZIP_PATH     = '/content/drive/MyDrive/datasets/GTZAN.zip'
LOCAL_BASE         = '/content/gtzan_local/Data/genres_original'
LOCAL_EXTRACT_ROOT = '/content/gtzan_local'
SPLIT_FILE         = os.path.join(DRIVE_BASE, 'dataset_splits.json')
OUTPUT_CSV         = os.path.join(DRIVE_BASE, 'classical_residual_analysis.csv')
OUTPUT_FIG         = os.path.join(DRIVE_BASE, 'classical_residual_figure.pdf')

SAMPLE_RATE   = 16000
TARGET_LENGTH = int(9.6 * SAMPLE_RATE)
# Control genre: normal Deletion AUC, similar harmonic energy profile to Classical
CONTROL_GENRE = 'country'
TARGET_GENRE  = 'classical'
N_FFT         = 1024
HOP_LENGTH    = 512
HIGH_FREQ_HZ  = 4000  # threshold for high-frequency energy ratio

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def ensure_gtzan_local():
    if os.path.exists(LOCAL_BASE):
        genre_dirs = [d for d in os.listdir(LOCAL_BASE)
                      if os.path.isdir(os.path.join(LOCAL_BASE, d))]
        if len(genre_dirs) >= 10:
            log("GTZAN already extracted.")
            return
    log("Extracting GTZAN from Drive...")
    shutil.copy2(GTZAN_ZIP_PATH, '/content/GTZAN.zip')
    os.makedirs(LOCAL_EXTRACT_ROOT, exist_ok=True)
    with zipfile.ZipFile('/content/GTZAN.zip', 'r') as zf:
        zf.extractall(LOCAL_EXTRACT_ROOT)
    os.remove('/content/GTZAN.zip')
    if not os.path.exists(LOCAL_BASE):
        raise RuntimeError(f"Expected path not found: {LOCAL_BASE}")
    log("GTZAN ready.")

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

def spectral_flatness(S_mag):
    """Geometric mean / arithmetic mean of spectral magnitude (per frame, then averaged)."""
    S = np.abs(S_mag) + 1e-10
    geo_mean = np.exp(np.mean(np.log(S), axis=0))
    ari_mean = np.mean(S, axis=0)
    return np.mean(geo_mean / ari_mean)

def high_freq_energy_ratio(S_mag, sr, n_fft, threshold_hz):
    """Fraction of total energy above threshold_hz."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high_mask = freqs >= threshold_hz
    total = np.sum(S_mag ** 2)
    high  = np.sum(S_mag[high_mask, :] ** 2)
    return high / total if total > 0 else 0

def temporal_variance(S_mag):
    """Mean variance of magnitude across time frames — proxy for stationarity."""
    return np.mean(np.var(S_mag, axis=1))

def analyse_clip(y):
    """Returns dict of residual spectral features for one clip."""
    y_h, y_p = librosa.effects.hpss(y, margin=1.0)
    y_r = y - y_h - y_p

    S_r = np.abs(librosa.stft(y_r, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_full = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))

    return {
        'flatness':   spectral_flatness(S_r),
        'hf_ratio':   high_freq_energy_ratio(S_r, SAMPLE_RATE, N_FFT, HIGH_FREQ_HZ),
        'temp_var':   temporal_variance(S_r),
        'res_energy': np.sqrt(np.mean(y_r ** 2)),          # RMS of residual
        'res_fraction': np.sqrt(np.mean(y_r**2)) /         # residual as fraction of total RMS
                        (np.sqrt(np.mean(y**2)) + 1e-10),
        'S_r': S_r,      # keep for figure
        'S_full': S_full
    }

def run():
    ensure_gtzan_local()

    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)

    results = {TARGET_GENRE: [], CONTROL_GENRE: []}
    spectrograms = {TARGET_GENRE: [], CONTROL_GENRE: []}
    csv_rows = []

    for genre in [TARGET_GENRE, CONTROL_GENRE]:
        clips = [f for f in splits['test']
                 if (f.split('/')[0] if '/' in f else f.split('\\')[0]) == genre]
        log(f"Processing {len(clips)} {genre} clips...")

        for rel_path in clips:
            clip_name = os.path.basename(rel_path).replace('.wav','').replace('.au','')
            wav_path  = os.path.join(LOCAL_BASE, rel_path)
            if not os.path.exists(wav_path):
                log(f"  Missing: {wav_path}")
                continue
            try:
                y = load_audio(wav_path)
                feat = analyse_clip(y)
                results[genre].append(feat)
                spectrograms[genre].append(feat['S_r'])
                csv_rows.append({
                    'clip':        clip_name,
                    'genre':       genre,
                    'flatness':    feat['flatness'],
                    'hf_ratio':    feat['hf_ratio'],
                    'temp_var':    feat['temp_var'],
                    'res_energy':  feat['res_energy'],
                    'res_fraction':feat['res_fraction']
                })
            except Exception as e:
                log(f"  ERROR on {clip_name}: {e}")

    # Save CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'clip','genre','flatness','hf_ratio','temp_var',
            'res_energy','res_fraction'])
        writer.writeheader()
        writer.writerows(csv_rows)
        os.sync()
    log(f"Per-clip CSV saved to {OUTPUT_CSV}")

    # Print comparison table
    print("\n--- Residual Spectral Features: Classical vs Control (Country) ---")
    metrics = ['flatness', 'hf_ratio', 'temp_var', 'res_energy', 'res_fraction']
    labels  = ['Spectral Flatness', 'HF Energy Ratio (>4kHz)',
               'Temporal Variance', 'Residual RMS', 'Residual Fraction']

    print(f"\n{'Metric':<28} | {'Classical':>12} | {'Country':>12} | {'p-value':>10}")
    print("-" * 70)

    for m, label in zip(metrics, labels):
        cl_vals = [r[m] for r in results[TARGET_GENRE]]
        co_vals = [r[m] for r in results[CONTROL_GENRE]]
        if not cl_vals or not co_vals:
            continue
        cl_mean = np.mean(cl_vals)
        co_mean = np.mean(co_vals)
        try:
            _, p = mannwhitneyu(cl_vals, co_vals, alternative='two-sided')
            sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
            p_str = f"{p:.4f} {sig}"
        except Exception:
            p_str = "n/a"
        print(f"{label:<28} | {cl_mean:>12.4f} | {co_mean:>12.4f} | {p_str:>10}")

    # Generate figure: mean residual spectrograms side by side
    log("Generating residual spectrogram figure...")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    for ax, genre, title in zip(axes,
                                 [TARGET_GENRE, CONTROL_GENRE],
                                 ['Classical (anomalous)', 'Country (normal)']):
        if not spectrograms[genre]:
            continue
        mean_S = np.mean(np.stack(spectrograms[genre]), axis=0)
        mean_S_db = librosa.amplitude_to_db(mean_S, ref=np.max)
        img = librosa.display.specshow(mean_S_db, sr=SAMPLE_RATE,
                                       hop_length=HOP_LENGTH,
                                       x_axis='time', y_axis='hz',
                                       ax=ax, cmap='magma')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Frequency (Hz)', fontsize=8)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

    fig.suptitle('Mean Residual ($x_R$) Spectrograms', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Figure saved to {OUTPUT_FIG}")
    log("Done.")
    os.sync()

if __name__ == "__main__":
    run()
