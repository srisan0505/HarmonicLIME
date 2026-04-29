# ==============================================================================
# File Name: FMA_02_harmoniclime.py
# Description: 
#   Applies HarmonicLIME to the FMA-small processed clips using the FROZEN 
#   GTZAN-fine-tuned VGGish model. Target class for explanation is the model's 
#   argmax prediction (standard zero-shot XAI faithfulness protocol).
#   run pip install resampy before running this program
# ==============================================================================

import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from datetime import datetime
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

NUM_CLASSES = 10
SAMPLE_RATE = 16000 
TARGET_LENGTH = int(9.6 * SAMPLE_RATE) 
K_WINDOWS = 10 
N_SAMPLES = 500
ALPHA = 0.01 
BATCH_SIZE = 50

DRIVE_BASE = '/content/drive/MyDrive/HarmonicLIME'
FMA_PROCESSED_DIR = os.path.join(DRIVE_BASE, 'fma_processed')
BEST_MODEL_PATH = os.path.join(DRIVE_BASE, 'vggish_best_model.pth')
FMA_RESULTS_DIR = os.path.join(DRIVE_BASE, 'fma_lime_results')
os.makedirs(FMA_RESULTS_DIR, exist_ok=True)

class VGGishGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=False, postprocess=False)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        return self.classifier(self.vggish(x))

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_fma_harmoniclime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    model = VGGishGenreClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    preprocessor = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=True)
    preprocessor.eval()
    
    fma_genres = os.listdir(FMA_PROCESSED_DIR)
    
    for genre in fma_genres:
        genre_path = os.path.join(FMA_PROCESSED_DIR, genre)
        if not os.path.isdir(genre_path): continue
        
        log(f"Processing FMA Genre: {genre}")
        clips = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        
        for clip in tqdm(clips):
            save_path = os.path.join(FMA_RESULTS_DIR, f"{genre}_{clip.replace('.wav', '_weights.npy')}")
            if os.path.exists(save_path): continue
            
            y, _ = torchaudio.load(os.path.join(genre_path, clip))
            y = y.squeeze(0).numpy()
            
            # Identify target class (zero-shot predicted class)
            with torch.no_grad():
                mel_tensor = preprocessor._preprocess(y, SAMPLE_RATE).view(-1, 1, 96, 64).to(device)
                outputs = model(mel_tensor).mean(dim=0, keepdim=True)
                target_idx = torch.argmax(outputs, dim=1).item()
            
            # HPSS & Resynthesis
            y_h, y_p = librosa.effects.hpss(y, margin=1.0)
            y_r = y - y_h - y_p
            
            window_samples = TARGET_LENGTH // K_WINDOWS
            z_vectors = np.random.binomial(1, 0.5, size=(N_SAMPLES, 3, K_WINDOWS))
            z_vectors[0, :, :] = 1 
            
            perturbed_audios = []
            for i in range(N_SAMPLES):
                z = z_vectors[i]
                y_tilde = np.zeros(TARGET_LENGTH)
                for k in range(K_WINDOWS):
                    start, end = k * window_samples, (k+1) * window_samples
                    y_tilde[start:end] = (z[0, k] * y_h[start:end] + z[1, k] * y_p[start:end] + z[2, k] * y_r[start:end])
                perturbed_audios.append(y_tilde)
                
            # Score perturbations
            z_flat = z_vectors.reshape(N_SAMPLES, 30)
            target_probs = []
            
            with torch.no_grad():
                for i in range(0, len(perturbed_audios), BATCH_SIZE):
                    batch = perturbed_audios[i:i+BATCH_SIZE]
                    batch_mels = [preprocessor._preprocess(au, SAMPLE_RATE) for au in batch]
                    batch_tensor = torch.stack(batch_mels).view(-1, 1, 96, 64).to(device)
                    outputs = model(batch_tensor).view(len(batch), -1, NUM_CLASSES).mean(dim=1)
                    probs = torch.softmax(outputs, dim=1)[:, target_idx].cpu().numpy()
                    target_probs.extend(probs)
                    
            distances = pairwise_distances(z_flat, z_flat[0].reshape(1, -1), metric='cosine').ravel()
            weights = np.sqrt(np.exp(-(distances ** 2) / 0.25 ** 2))
            
            solver = Ridge(alpha=ALPHA, fit_intercept=True)
            solver.fit(z_flat, target_probs, sample_weight=weights)
            np.save(save_path, solver.coef_.reshape(3, K_WINDOWS))

if __name__ == "__main__":
    run_fma_harmoniclime()