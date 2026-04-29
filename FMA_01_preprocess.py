# ==============================================================================
# File Name: FMA_01_preprocess.py
# Version: 1.1 (Fixed nested extraction path & added error logging)
# Description: 
#   Extracts FMA_small.zip, downloads FMA metadata, samples 100 clips per 
#   genre (seed=42), and safely converts them to GTZAN-equivalent WAV formats.
# ==============================================================================

import os
import zipfile
import urllib.request
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings

from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = '/content/drive/MyDrive/HarmonicLIME'
FMA_ZIP = os.path.join(DRIVE_BASE, 'FMA_small.zip')

# FIXED: Pointing to the nested directory your diagnostic script found
FMA_EXTRACT_DIR = '/content/fma_small/fma_small' 
FMA_PROCESSED_DIR = os.path.join(DRIVE_BASE, 'fma_processed')
META_DIR = '/content/fma_metadata'
META_URL = 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'

SAMPLE_RATE = 16000
TARGET_LENGTH = int(9.6 * SAMPLE_RATE)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def setup_metadata():
    if not os.path.exists(META_DIR):
        log("Downloading FMA metadata...")
        meta_zip = '/content/fma_metadata.zip'
        urllib.request.urlretrieve(META_URL, meta_zip)
        with zipfile.ZipFile(meta_zip, 'r') as z:
            z.extractall('/content/')
    
    log("Parsing tracks.csv...")
    tracks = pd.read_csv(os.path.join(META_DIR, 'tracks.csv'), index_col=0, header=[0, 1])
    small_subset = tracks[tracks[('set', 'subset')] == 'small']
    genres = small_subset[('track', 'genre_top')]
    
    # Sample 100 per genre
    sampled = genres.groupby(genres).sample(n=100, random_state=42)
    return sampled

def extract_audio():
    # Only extract if the root fma_small folder doesn't exist
    if not os.path.exists('/content/fma_small'):
        log(f"Extracting {FMA_ZIP} to local Colab storage...")
        with zipfile.ZipFile(FMA_ZIP, 'r') as z:
            z.extractall('/content/')

def process_audio(sampled_tracks):
    log("Processing audio clips to match GTZAN pipeline...")
    os.makedirs(FMA_PROCESSED_DIR, exist_ok=True)
    
    success_count = 0
    for track_id, genre in tqdm(sampled_tracks.items(), total=len(sampled_tracks)):
        genre_dir = os.path.join(FMA_PROCESSED_DIR, genre)
        os.makedirs(genre_dir, exist_ok=True)
        
        tid_str = f"{track_id:06d}"
        mp3_path = os.path.join(FMA_EXTRACT_DIR, tid_str[:3], f"{tid_str}.mp3")
        wav_path = os.path.join(genre_dir, f"{tid_str}.wav")
        
        # Skip if we already successfully processed this file
        if os.path.exists(wav_path):
            success_count += 1
            continue
            
        if not os.path.exists(mp3_path):
            print(f"\nMissing file: {mp3_path}")
            continue
            
        try:
            # Librosa throws a lot of warnings for MP3s, we catch them to keep output clean
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
            
            # Center Crop to 9.6s
            if len(y) > TARGET_LENGTH:
                start = (len(y) - TARGET_LENGTH) // 2
                y = y[start:start+TARGET_LENGTH]
            else:
                y = np.pad(y, (0, TARGET_LENGTH - len(y)))
                
            sf.write(wav_path, y, SAMPLE_RATE)
            success_count += 1
        except Exception as e:
            print(f"\nError processing {tid_str}.mp3: {e}")
            
    log(f"Successfully processed {success_count} clips into {FMA_PROCESSED_DIR}")

if __name__ == "__main__":
    sampled_tracks = setup_metadata()
    extract_audio()
    process_audio(sampled_tracks)