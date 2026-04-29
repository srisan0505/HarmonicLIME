# ==============================================================================
# File Name: 09_hyperparam_sensitivity.py
# Version: 1.0
# Description:
#   Step 9 of the HarmonicLIME pipeline.
#   Addresses R2 Minor Issue 3: hyperparameter sensitivity analysis.
#   Runs HarmonicLIME with N in {100, 500, 1000, 2000} perturbations on a
#   stratified subset (5 clips per genre, 50 clips total) drawn from the
#   existing test split. For each N, computes the mean Deletion AUC
#   (excluding Classical) and the mean pairwise cosine similarity of
#   saliency maps across runs to assess stability.
#   All other hyperparameters (K=10, alpha=0.01, sigma=0.25) are fixed.
#   Results are printed as a LaTeX-ready row for insertion into the paper.
#
# Hardware Requirements:
#   - GPU strongly recommended (T4 or better)
#   - N=2000 on CPU is impractically slow (~30 min/clip)
#   - Estimated runtime on T4: ~45-60 minutes total
#
# Dependencies:
#   - librosa, numpy, torchaudio, torch, sklearn, resampy
#   - Google Colab + Google Drive mount
#
# Inputs:
#   - /content/drive/MyDrive/HarmonicLIME/vggish_best_model.pth
#   - /content/drive/MyDrive/HarmonicLIME/dataset_splits.json
#   - /content/drive/MyDrive/datasets/GTZAN.zip
#     (extracted to /content/gtzan_local/)
#
# Outputs:
#   - /content/drive/MyDrive/HarmonicLIME/sensitivity_results.csv
#   - Printed sensitivity table (N vs mean AUC vs saliency stability)
# ==============================================================================

pip_install('resampy')
import os
import json
import csv
import zipfile
import shutil
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchaudio
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
ALPHA         = 0.01
BATCH_SIZE    = 50
N_VALUES      = [100, 500, 1000, 2000]
N_REPEATS     = 3       # runs per N to assess stability
CLIPS_PER_GENRE = 5     # stratified subset: 5 per genre = 50 clips total
EXCLUDE_GENRE = 'classical'  # excluded from AUC mean per paper convention

# Paths
DRIVE_BASE        = '/content/drive/MyDrive/HarmonicLIME'
GTZAN_ZIP_PATH    = '/content/drive/MyDrive/datasets/GTZAN.zip'
LOCAL_BASE        = '/content/gtzan_local/Data/genres_original'
LOCAL_EXTRACT_ROOT= '/content/gtzan_local'
SPLIT_FILE        = os.path.join(DRIVE_BASE, 'dataset_splits.json')
BEST_MODEL_PATH   = os.path.join(DRIVE_BASE, 'vggish_best_model.pth')
OUTPUT_CSV        = os.path.join(DRIVE_BASE, 'sensitivity_results.csv')

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

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

def get_predictions(model, preprocessor, audios, device):
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

def run_harmoniclime(y_h, y_p, y_r, model, preprocessor, device,
                     target_idx, n_samples):
    """Single HarmonicLIME run. Returns (saliency_map, deletion_auc)."""
    ws = TARGET_LENGTH // K_WINDOWS
    z_vectors = np.random.binomial(1, 0.5, size=(n_samples, 3, K_WINDOWS))
    z_vectors[0, :, :] = 1

    perturbed = []
    for i in range(n_samples):
        z = z_vectors[i]
        y_tilde = np.zeros(TARGET_LENGTH)
        for k in range(K_WINDOWS):
            s, e = k * ws, (k + 1) * ws
            y_tilde[s:e] = (z[0,k]*y_h[s:e] + z[1,k]*y_p[s:e] + z[2,k]*y_r[s:e])
        perturbed.append(y_tilde)

    z_flat = z_vectors.reshape(n_samples, 3 * K_WINDOWS)
    preds = get_predictions(model, preprocessor, perturbed, device)
    target_probs = preds[:, target_idx]

    distances = pairwise_distances(z_flat, z_flat[0:1], metric='cosine').ravel()
    weights = np.sqrt(np.exp(-(distances ** 2) / 0.25 ** 2))
    solver = Ridge(alpha=ALPHA, fit_intercept=True)
    solver.fit(z_flat, target_probs, sample_weight=weights)
    saliency = solver.coef_.reshape(3, K_WINDOWS)

    # Deletion AUC
    ranked = np.argsort(np.abs(saliency.flatten()))[::-1]
    probs_curve = []
    with torch.no_grad():
        for k in range(len(ranked) + 1):
            z_mask = np.ones((3, K_WINDOWS))
            for i in range(k):
                idx = ranked[i]
                z_mask[idx // K_WINDOWS, idx % K_WINDOWS] = 0
            y_tilde = np.zeros(TARGET_LENGTH)
            for w in range(K_WINDOWS):
                s, e = w * ws, (w + 1) * ws
                y_tilde[s:e] = (z_mask[0,w]*y_h[s:e] + z_mask[1,w]*y_p[s:e] +
                                z_mask[2,w]*y_r[s:e])
            mel = preprocessor._preprocess(y_tilde, SAMPLE_RATE)
            mel = mel.view(-1, 1, 96, 64).to(device)
            prob = torch.softmax(model(mel).mean(dim=0, keepdim=True),
                                 dim=1)[0, target_idx].item()
            probs_curve.append(prob)

    del_auc = auc(np.linspace(0, 1, len(probs_curve)), probs_curve)
    return saliency, del_auc

def run():
    ensure_gtzan_local()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if device.type == 'cpu':
        log("WARNING: No GPU detected. Runtime will be very long.")

    model = VGGishGenreClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    preprocessor = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=True)
    preprocessor.eval()

    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)

    # Stratified subset: first CLIPS_PER_GENRE test clips per genre
    subset = []
    for g in GENRES:
        clips = [f for f in splits['test']
                 if (f.split('/')[0] if '/' in f else f.split('\\')[0]) == g]
        subset.extend(clips[:CLIPS_PER_GENRE])
    log(f"Subset: {len(subset)} clips ({CLIPS_PER_GENRE} per genre)")

    # Results: {N: {clip: [auc_run1, auc_run2, ...], saliency: [map1, map2, ...]}}
    csv_rows = []

    for n_samples in N_VALUES:
        log(f"=== N={n_samples} ===")
        aucs_ex_classical = []
        stabilities = []  # mean pairwise cosine sim across repeats

        for rel_path in subset:
            genre = rel_path.split('/')[0] if '/' in rel_path else rel_path.split('\\')[0]
            clip_name = os.path.basename(rel_path).replace('.wav','').replace('.au','')
            target_idx = GENRE_TO_IDX[genre]
            wav_path = os.path.join(LOCAL_BASE, rel_path)

            try:
                y = load_audio(wav_path)
                y_h, y_p = librosa.effects.hpss(y, margin=1.0)
                y_r = y - y_h - y_p
            except Exception as e:
                log(f"  Load error {clip_name}: {e}")
                continue

            run_aucs = []
            run_maps = []
            for r in range(N_REPEATS):
                saliency, del_auc = run_harmoniclime(
                    y_h, y_p, y_r, model, preprocessor,
                    device, target_idx, n_samples)
                run_aucs.append(del_auc)
                run_maps.append(saliency.flatten())

            if genre != EXCLUDE_GENRE:
                aucs_ex_classical.append(np.mean(run_aucs))

            # Pairwise cosine similarity across N_REPEATS runs
            maps = np.array(run_maps)
            sims = []
            for i in range(N_REPEATS):
                for j in range(i + 1, N_REPEATS):
                    a, b = maps[i], maps[j]
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    sims.append(np.dot(a, b) / denom if denom > 0 else 0)
            stabilities.append(np.mean(sims))

            log(f"  {clip_name}: mean_auc={np.mean(run_aucs):.4f} "
                f"stability={np.mean(sims):.4f}")

        mean_auc = np.mean(aucs_ex_classical)
        mean_stab = np.mean(stabilities)
        log(f"N={n_samples}: Mean AUC (ex.Classical)={mean_auc:.4f}, "
            f"Mean Stability={mean_stab:.4f}")
        csv_rows.append({'N': n_samples,
                         'mean_auc_ex_classical': mean_auc,
                         'mean_stability': mean_stab})

    # Save CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['N', 'mean_auc_ex_classical',
                                               'mean_stability'])
        writer.writeheader()
        writer.writerows(csv_rows)
        os.sync()
    log(f"Results saved to {OUTPUT_CSV}")

    # Print table
    print("\n--- Hyperparameter Sensitivity: N vs AUC and Stability ---")
    print(f"{'N':>6} | {'Mean AUC (ex.Cl.)':>18} | {'Saliency Stability':>18}")
    print("-" * 50)
    for row in csv_rows:
        print(f"{row['N']:>6} | {row['mean_auc_ex_classical']:>18.4f} | "
              f"{row['mean_stability']:>18.4f}")
    print("\nStability = mean pairwise cosine similarity across repeated runs.")
    print("Values near 1.0 indicate low sensitivity to random perturbation sampling.")

if __name__ == "__main__":
    run()
