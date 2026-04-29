# ==============================================================================
# File Name: 11_audiolime_comparison.py
# Version: 1.0
# Description:
#   Step 11 of the HarmonicLIME pipeline.
#   Addresses R2 Major Issue 3: empirical comparison against audioLIME.
#   Implements audioLIME using Open-Unmix (UMX) source separation to define
#   perturbation units (vocals, drums, bass, other) instead of HPR components.
#   For each clip, audioLIME generates N=500 perturbations by muting
#   (source, time-window) pairs, fits a Ridge surrogate, and computes
#   Deletion AUC using the same protocol as HarmonicLIME.
#   Runs on the same 50-clip stratified subset (5 per genre) used in the
#   N-sweep sensitivity analysis (09_hyperparam_sensitivity.py).
#   Results are saved as a CSV and printed as two LaTeX table options:
#     Option A: fourth column added to existing Table I
#     Option B: separate compact comparison table
#
# Hardware Requirements:
#   - GPU strongly recommended (T4 or better)
#   - Estimated runtime on T4: 2-3 hours (50 clips x N=500 perturbations)
#
# Dependencies:
#   - resampy, openunmix (installed by this script via pip)
#   - librosa, numpy, torchaudio, torch, sklearn
#   - Google Colab + Google Drive mount
#
# Inputs:
#   - /content/drive/MyDrive/HarmonicLIME/vggish_best_model.pth
#   - /content/drive/MyDrive/HarmonicLIME/dataset_splits.json
#   - /content/drive/MyDrive/datasets/GTZAN.zip
#     (extracted to /content/gtzan_local/)
#
# Outputs:
#   - /content/drive/MyDrive/HarmonicLIME/audiolime_perclip.csv
#   - Printed LaTeX table options A and B
# ==============================================================================

import subprocess
import sys

def pip_install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

pip_install('openunmix')
pip_install('resampy')

import os
import csv
import json
import zipfile
import shutil
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances, auc
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

# Hyperparameters
NUM_CLASSES   = 10
SAMPLE_RATE   = 16000
TARGET_LENGTH = int(9.6 * SAMPLE_RATE)
K_WINDOWS     = 10
N_SAMPLES     = 500
ALPHA         = 0.01
BATCH_SIZE    = 50
CLIPS_PER_GENRE = 5
N_SOURCES     = 4   # Open-Unmix: vocals, drums, bass, other

# Paths
DRIVE_BASE         = '/content/drive/MyDrive/HarmonicLIME'
GTZAN_ZIP_PATH     = '/content/drive/MyDrive/datasets/GTZAN.zip'
LOCAL_BASE         = '/content/gtzan_local/Data/genres_original'
LOCAL_EXTRACT_ROOT = '/content/gtzan_local'
SPLIT_FILE         = os.path.join(DRIVE_BASE, 'dataset_splits.json')
BEST_MODEL_PATH    = os.path.join(DRIVE_BASE, 'vggish_best_model.pth')
OUTPUT_CSV         = os.path.join(DRIVE_BASE, 'audiolime_perclip.csv')

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
SOURCE_NAMES = ['vocals', 'drums', 'bass', 'other']

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

class VGGishGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish',
                                     preprocess=False, postprocess=False)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        return self.classifier(self.vggish(x))

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

def separate_sources(y, separator, device):
    """
    Run Open-Unmix separation on mono waveform y.
    Returns array of shape (N_SOURCES, TARGET_LENGTH).
    Open-Unmix umxl returns a Tensor of shape (n_sources, batch, channels, samples).
    Source order: vocals, drums, bass, other.
    """
    y_tensor = torch.tensor(y, dtype=torch.float32)
    y_stereo = y_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1).to(device)

    with torch.no_grad():
        estimates = separator(y_stereo)
        # estimates shape: (n_sources, batch, channels, samples)

    sources = []
    for src_idx in range(min(N_SOURCES, estimates.shape[0])):
        # (batch=0, mean over channels, all samples)
        src = estimates[src_idx, 0].mean(dim=0).cpu().numpy()
        if len(src) > TARGET_LENGTH:
            src = src[:TARGET_LENGTH]
        elif len(src) < TARGET_LENGTH:
            src = np.pad(src, (0, TARGET_LENGTH - len(src)))
        sources.append(src)

    # Pad with zeros if fewer sources returned than expected
    while len(sources) < N_SOURCES:
        sources.append(np.zeros(TARGET_LENGTH))

    return np.stack(sources)  # (N_SOURCES, TARGET_LENGTH)

def get_predictions(model, preprocessor, audios, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(audios), BATCH_SIZE):
            batch = audios[i:i + BATCH_SIZE]
            mels = torch.stack([preprocessor._preprocess(a, SAMPLE_RATE) for a in batch])
            bs, nf = mels.shape[0], mels.shape[1]
            mels = mels.view(-1, 1, 96, 64).to(device)
            out = model(mels).view(bs, nf, NUM_CLASSES).mean(dim=1)
            all_preds.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.vstack(all_preds)

def run_audiolime(sources, model, preprocessor, device, target_idx):
    """
    sources: (N_SOURCES, TARGET_LENGTH)
    Returns (saliency_map, deletion_auc)
    saliency_map shape: (N_SOURCES, K_WINDOWS)
    """
    ws = TARGET_LENGTH // K_WINDOWS
    n_features = N_SOURCES * K_WINDOWS  # 4 x 10 = 40

    z_vectors = np.random.binomial(1, 0.5, size=(N_SAMPLES, N_SOURCES, K_WINDOWS))
    z_vectors[0, :, :] = 1  # original

    perturbed = []
    for i in range(N_SAMPLES):
        z = z_vectors[i]
        y_tilde = np.zeros(TARGET_LENGTH)
        for k in range(K_WINDOWS):
            s, e = k * ws, (k + 1) * ws
            for src_idx in range(N_SOURCES):
                y_tilde[s:e] += z[src_idx, k] * sources[src_idx, s:e]
        perturbed.append(y_tilde)

    z_flat = z_vectors.reshape(N_SAMPLES, n_features)
    preds = get_predictions(model, preprocessor, perturbed, device)
    target_probs = preds[:, target_idx]

    distances = pairwise_distances(z_flat, z_flat[0:1], metric='cosine').ravel()
    weights = np.sqrt(np.exp(-(distances ** 2) / 0.25 ** 2))
    solver = Ridge(alpha=ALPHA, fit_intercept=True)
    solver.fit(z_flat, target_probs, sample_weight=weights)
    saliency = solver.coef_.reshape(N_SOURCES, K_WINDOWS)

    # Deletion AUC
    ranked = np.argsort(np.abs(saliency.flatten()))[::-1]
    probs_curve = []
    with torch.no_grad():
        for k in range(len(ranked) + 1):
            z_mask = np.ones((N_SOURCES, K_WINDOWS))
            for i in range(k):
                idx = ranked[i]
                z_mask[idx // K_WINDOWS, idx % K_WINDOWS] = 0
            y_tilde = np.zeros(TARGET_LENGTH)
            for w in range(K_WINDOWS):
                s, e = w * ws, (w + 1) * ws
                for src_idx in range(N_SOURCES):
                    y_tilde[s:e] += z_mask[src_idx, w] * sources[src_idx, s:e]
            mel = preprocessor._preprocess(y_tilde, SAMPLE_RATE)
            mel = mel.view(-1, 1, 96, 64).to(device)
            prob = torch.softmax(model(mel).mean(dim=0, keepdim=True),
                                 dim=1)[0, target_idx].item()
            probs_curve.append(prob)

    del_auc = auc(np.linspace(0, 1, len(probs_curve)), probs_curve)
    return saliency, del_auc

def restore_checkpoint():
    """Load previously completed clips directly from Drive CSV."""
    if os.path.exists(OUTPUT_CSV):
        rows = []
        with open(OUTPUT_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if rows:
            log(f"Restored {len(rows)} completed clips from Drive.")
            return {r['clip'] for r in rows}, rows
    log("No checkpoint found on Drive. Starting fresh.")
    return set(), []

def append_to_drive(row):
    """Append one completed row directly to Drive CSV after every clip."""
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['clip', 'genre', 'al_auc'])
        if not file_exists:
            writer.writeheader()
            os.sync()
        writer.writerow(row)
        os.sync()


def print_latex_options(csv_rows):
    """Print two LaTeX table formatting options."""

    # Aggregate by genre
    genre_data = {g: [] for g in GENRES}
    for row in csv_rows:
        genre_data[row['genre']].append(float(row['al_auc']))

    all_vals = [float(r['al_auc']) for r in csv_rows]
    ex_cl    = [float(r['al_auc']) for r in csv_rows if r['genre'] != 'classical']

    print("\n" + "="*70)
    print("OPTION A: Add audioLIME as fourth column to existing Table I")
    print("(Insert this column data alongside existing Mel-LIME/SHAP/HL columns)")
    print("="*70)
    print("\\textbf{audioLIME} column values:")
    for g in GENRES:
        vals = genre_data[g]
        if vals:
            print(f"  {g:<12}: {np.mean(vals):.4f}$\\pm${np.std(vals,ddof=1):.4f}")
    print(f"  {'Mean (all)':<12}: {np.mean(all_vals):.4f}$\\pm${np.std(all_vals,ddof=1):.4f}")
    print(f"  {'Mean (ex.Cl.)':<12}: {np.mean(ex_cl):.4f}$\\pm${np.std(ex_cl,ddof=1):.4f}")

    print("\n" + "="*70)
    print("OPTION B: Separate compact comparison table")
    print("="*70)
    print("\\begin{table}[t]")
    print("\\caption{AudioLIME vs HarmonicLIME: Mean Deletion AUC (GTZAN subset)}")
    print("\\label{tab:audiolime}")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Genre} & \\textbf{audioLIME} & \\textbf{HarmonicLIME} \\\\")
    print("\\midrule")
    for g in GENRES:
        vals = genre_data[g]
        if vals:
            print(f"  {g:<12} & {np.mean(vals):.4f} & (from Table~\\ref{{tab:deletion_auc}}) \\\\")
    print("\\midrule")
    print(f"  Mean (all) & {np.mean(all_vals):.4f} & (from Table~\\ref{{tab:deletion_auc}}) \\\\")
    print(f"  Mean (ex.\\ Classical) & {np.mean(ex_cl):.4f} & (from Table~\\ref{{tab:deletion_auc}}) \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def run():
    ensure_gtzan_local()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if device.type == 'cpu':
        log("WARNING: No GPU detected. Runtime will be very long.")

    # Load Open-Unmix separator
    log("Loading Open-Unmix separator...")
    import openunmix
    separator = openunmix.umxl(targets=SOURCE_NAMES, residual=False)
    separator = separator.to(device)
    separator.eval()
    log("Open-Unmix loaded.")

    # Load VGGish classifier
    model = VGGishGenreClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    preprocessor = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=True)
    preprocessor.eval()

    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)

    # Same stratified subset as script 09
    subset = []
    for g in GENRES:
        clips = [f for f in splits['test']
                 if (f.split('/')[0] if '/' in f else f.split('\\')[0]) == g]
        subset.extend(clips[:CLIPS_PER_GENRE])
    log(f"Subset: {len(subset)} clips ({CLIPS_PER_GENRE} per genre)")

    # Resume: load already-completed clips
    completed_clips, csv_rows = restore_checkpoint()
    log(f"Skipping {len(completed_clips)} already-completed clips.")

    done, failed = 0, 0

    for rel_path in subset:
        genre = rel_path.split('/')[0] if '/' in rel_path else rel_path.split('\\')[0]
        clip_name = os.path.basename(rel_path).replace('.wav','').replace('.au','')
        target_idx = GENRE_TO_IDX[genre]
        wav_path = os.path.join(LOCAL_BASE, rel_path)

        if clip_name in completed_clips:
            continue

        try:
            log(f"Processing {clip_name}...")
            y = load_audio(wav_path)
            sources = separate_sources(y, separator, device)
            _, al_auc = run_audiolime(sources, model, preprocessor, device, target_idx)
            row = {'clip': clip_name, 'genre': genre, 'al_auc': al_auc}
            csv_rows.append(row)
            append_to_drive(row)
            done += 1
            log(f"  audioLIME AUC={al_auc:.4f} [{done} new, {len(csv_rows)} total]")

        except Exception as e:
            log(f"  ERROR on {clip_name}: {e}")
            failed += 1

    # Final save to Drive
    log(f"Drive CSV updated: {len(csv_rows)} clips total.")
    log(f"Done: {done} new | Skipped: {len(completed_clips)} | Failed: {failed}")

    # Print table options
    if csv_rows:
        print_latex_options(csv_rows)

if __name__ == "__main__":
    run()
    os.sync()
