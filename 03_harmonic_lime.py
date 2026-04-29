# ==============================================================================
# File Name: 03_harmonic_lime_v3.py
# Version: 3.0 (Local Write + Zip-to-Drive)
# Description:
#   Step 3 of the HarmonicLIME pipeline.
#   CHANGES FROM v2.0:
#   - All _weights.npy files are written to LOCAL Colab storage first
#     (/content/lime_results_local/) to avoid Drive sync issues with 1000 files.
#   - At the end, all .npy files are zipped into lime_results.zip and copied
#     to /content/drive/MyDrive/HarmonicLIME/lime_results/lime_results.zip
#   - On resume: if lime_results.zip already exists on Drive, it is extracted
#     back to local storage so already-processed clips are skipped correctly.
#   REQUIRES: GPU strongly recommended (T4 or better).
# ==============================================================================

import os
import json
import zipfile
import shutil
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

# Hyperparameters
NUM_CLASSES = 10
SAMPLE_RATE = 16000
TARGET_LENGTH = int(9.6 * SAMPLE_RATE)
K_WINDOWS = 10
N_SAMPLES = 500
ALPHA = 0.01
BATCH_SIZE = 50

# Paths
DRIVE_BASE        = '/content/drive/MyDrive/HarmonicLIME'
GTZAN_ZIP_PATH    = '/content/drive/MyDrive/datasets/GTZAN.zip'
LOCAL_BASE        = '/content/gtzan_local/Data/genres_original'
LOCAL_EXTRACT_ROOT= '/content/gtzan_local'
SPLIT_FILE        = os.path.join(DRIVE_BASE, 'dataset_splits.json')
BEST_MODEL_PATH   = os.path.join(DRIVE_BASE, 'vggish_best_model.pth')

# Local scratch dir — fast I/O, no Drive sync
LOCAL_RESULTS_DIR = '/content/lime_results_local'
# Single zip written to Drive at the end
DRIVE_RESULTS_DIR = os.path.join(DRIVE_BASE, 'lime_results')
DRIVE_ZIP_PATH    = os.path.join(DRIVE_RESULTS_DIR, 'lime_results.zip')

os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# ==============================================================================
# Resume support: restore previously saved .npy files from Drive zip if present
# ==============================================================================
def restore_from_drive_zip():
    if os.path.exists(DRIVE_ZIP_PATH):
        existing = [f for f in os.listdir(LOCAL_RESULTS_DIR) if f.endswith('_weights.npy')]
        if existing:
            log(f"Local scratch already has {len(existing)} files, skipping restore.")
            return
        log(f"Found existing zip on Drive. Restoring to local scratch for resume...")
        with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as zf:
            zf.extractall(LOCAL_RESULTS_DIR)
        restored = len([f for f in os.listdir(LOCAL_RESULTS_DIR) if f.endswith('_weights.npy')])
        log(f"Restored {restored} weight files to {LOCAL_RESULTS_DIR}")
    else:
        log("No existing zip on Drive. Starting fresh.")

# ==============================================================================
# Zip all local .npy files and copy to Drive (called at end and on interruption)
# ==============================================================================
def save_zip_to_drive():
    files = [f for f in os.listdir(LOCAL_RESULTS_DIR) if f.endswith('_weights.npy')]
    if not files:
        log("No weight files to zip.")
        return
    local_zip = '/content/lime_results.zip'
    log(f"Zipping {len(files)} weight files...")
    with zipfile.ZipFile(local_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(os.path.join(LOCAL_RESULTS_DIR, f), arcname=f)
    log(f"Copying zip to Drive: {DRIVE_ZIP_PATH}")
    shutil.copy2(local_zip, DRIVE_ZIP_PATH)
    os.remove(local_zip)
    log("Zip saved to Drive successfully.")

# ==============================================================================
# GTZAN extraction
# ==============================================================================
def ensure_gtzan_local():
    if os.path.exists(LOCAL_BASE):
        genre_dirs = [d for d in os.listdir(LOCAL_BASE)
                      if os.path.isdir(os.path.join(LOCAL_BASE, d))]
        if len(genre_dirs) >= 10:
            log(f"GTZAN already extracted ({len(genre_dirs)} genre folders).")
            return
        shutil.rmtree(LOCAL_EXTRACT_ROOT, ignore_errors=True)

    log("Copying GTZAN.zip from Drive...")
    shutil.copy2(GTZAN_ZIP_PATH, '/content/GTZAN.zip')
    log("Extracting...")
    os.makedirs(LOCAL_EXTRACT_ROOT, exist_ok=True)
    with zipfile.ZipFile('/content/GTZAN.zip', 'r') as zf:
        zf.extractall(LOCAL_EXTRACT_ROOT)
    os.remove('/content/GTZAN.zip')
    if not os.path.exists(LOCAL_BASE):
        raise RuntimeError(
            f"Expected path not found after extraction: {LOCAL_BASE}\n"
            "Check the internal folder structure of GTZAN.zip.")
    log(f"GTZAN ready at {LOCAL_BASE}")

# ==============================================================================
# Model
# ==============================================================================
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
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.shape[1]))
    return waveform.squeeze(0).numpy()

def hpr_decomposition(y):
    y_h, y_p = librosa.effects.hpss(y, margin=1.0)
    return y_h, y_p, y - y_h - y_p

def generate_perturbations(y_h, y_p, y_r):
    window_samples = TARGET_LENGTH // K_WINDOWS
    z_vectors = np.random.binomial(1, 0.5, size=(N_SAMPLES, 3, K_WINDOWS))
    z_vectors[0, :, :] = 1
    perturbed_audios = []
    for i in range(N_SAMPLES):
        z = z_vectors[i]
        y_tilde = np.zeros(TARGET_LENGTH)
        for k in range(K_WINDOWS):
            s, e = k * window_samples, (k + 1) * window_samples
            y_tilde[s:e] = (z[0,k]*y_h[s:e] + z[1,k]*y_p[s:e] + z[2,k]*y_r[s:e])
        perturbed_audios.append(y_tilde)
    return z_vectors.reshape(N_SAMPLES, 3 * K_WINDOWS), perturbed_audios

def get_model_predictions(model, preprocessor, audios, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(audios), BATCH_SIZE):
            batch = audios[i:i + BATCH_SIZE]
            mels = torch.stack([preprocessor._preprocess(y, SAMPLE_RATE) for y in batch])
            bs, nf = mels.shape[0], mels.shape[1]
            mels = mels.view(-1, 1, 96, 64).to(device)
            out = model(mels).view(bs, nf, NUM_CLASSES).mean(dim=1)
            all_preds.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.vstack(all_preds)

# ==============================================================================
# Main
# ==============================================================================
def explain_all_test_clips():
    ensure_gtzan_local()
    restore_from_drive_zip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if device.type == 'cpu':
        log("WARNING: No GPU detected. Runtime will be very long (~10-15x slower).")

    model = VGGishGenreClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    preprocessor = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=True)
    preprocessor.eval()

    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)

    test_files = splits['test']
    log(f"Total test clips: {len(test_files)}")

    done, skipped, failed = 0, 0, 0
    # Checkpoint: zip to Drive every 100 clips
    CHECKPOINT_EVERY = 100

    for i, rel_path in enumerate(test_files):
        clip_name = os.path.basename(rel_path).replace('.wav', '').replace('.au', '')
        save_path = os.path.join(LOCAL_RESULTS_DIR, f"{clip_name}_weights.npy")

        if os.path.exists(save_path):
            skipped += 1
            continue

        target_genre = rel_path.split('/')[0] if '/' in rel_path else rel_path.split('\\')[0]
        if target_genre not in GENRE_TO_IDX:
            log(f"  Unknown genre for {rel_path}, skipping.")
            failed += 1
            continue

        try:
            y = load_and_preprocess_audio(os.path.join(LOCAL_BASE, rel_path))
            y_h, y_p, y_r = hpr_decomposition(y)
            z_flat, perturbed = generate_perturbations(y_h, y_p, y_r)
            preds = get_model_predictions(model, preprocessor, perturbed, device)
            target_probs = preds[:, GENRE_TO_IDX[target_genre]]

            distances = pairwise_distances(z_flat, z_flat[0:1], metric='cosine').ravel()
            weights = np.sqrt(np.exp(-(distances ** 2) / 0.25 ** 2))

            solver = Ridge(alpha=ALPHA, fit_intercept=True)
            solver.fit(z_flat, target_probs, sample_weight=weights)
            np.save(save_path, solver.coef_.reshape(3, K_WINDOWS))
            done += 1
            log(f"[{done+skipped}/{len(test_files)}] {clip_name} done.")

        except Exception as e:
            log(f"  ERROR on {clip_name}: {e}")
            failed += 1

        # Periodic checkpoint to Drive
        if (done > 0) and (done % CHECKPOINT_EVERY == 0):
            log(f"--- Checkpoint: saving zip to Drive ({done} new clips) ---")
            save_zip_to_drive()

    # Final zip
    log(f"\nAll clips processed. Done: {done} | Skipped: {skipped} | Failed: {failed}")
    save_zip_to_drive()
    os.sync()

if __name__ == "__main__":
    explain_all_test_clips()