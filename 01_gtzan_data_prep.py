# ==============================================================================
# File Name: 01_gtzan_data_prep.py
# Version: 1.0
# Description: 
#   Step 1 of the HarmonicLIME pipeline. This script connects to Google Drive,
#   unzips the GTZAN dataset to Colab's local runtime for faster I/O, and 
#   generates a reproducible 8/1/1 (Train/Val/Test) split across the 10 genres.
#   To recover from session breaks, it saves the split metadata (JSON) directly 
#   to the Google Drive 'HarmonicLIME' folder so subsequent scripts consistently 
#   use the same data splits without data leakage.
# ==============================================================================

import os
import zipfile
import json
import random
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

def log_progress(message):
    """Prints a timestamped progress message."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

def setup_environment():
    # 1. Mount Google Drive
    log_progress("Mounting Google Drive...")

    drive_base_path = '/content/drive/MyDrive/HarmonicLIME'
    local_base_path = '/content/gtzan_local'
    zip_path = os.path.join(drive_base_path, 'GTZAN.zip')
    split_file_path = os.path.join(drive_base_path, 'dataset_splits.json')

    # Ensure the drive folder exists
    if not os.path.exists(drive_base_path):
        log_progress(f"Creating base directory in Drive: {drive_base_path}")
        os.makedirs(drive_base_path)

    # 2. Extract Dataset to Local Colab Storage (if not already extracted)
    if not os.path.exists(local_base_path):
        log_progress("Local GTZAN directory not found. Starting extraction...")
        if not os.path.exists(zip_path):
            log_progress(f"ERROR: {zip_path} not found! Please upload GTZAN.zip to your HarmonicLIME Drive folder.")
            return None

        os.makedirs(local_base_path, exist_ok=True)
        log_progress(f"Unzipping {zip_path} to {local_base_path}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_base_path)
        log_progress("Extraction complete.")
    else:
        log_progress("Dataset already extracted in the current session.")

    return local_base_path, split_file_path

def create_and_save_splits(local_data_path, split_file_path):
    # 3. Check for existing splits to recover state
    if os.path.exists(split_file_path):
        log_progress(f"Found existing dataset splits at {split_file_path}. Loading state...")
        with open(split_file_path, 'r') as f:
            splits = json.load(f)
        log_progress("State recovered successfully.")
        return splits

    log_progress("No existing splits found. Generating new 8/1/1 Train/Val/Test splits...")
    
    # 10 genres as per GTZAN spec
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    splits = {'train': [], 'val': [], 'test': []}
    
    # Look for the actual audio folder inside the extracted path
    # (Sometimes zips extract into a subfolder like /content/gtzan_local/Data/genres_original)
    search_path = local_data_path
    for root, dirs, files in os.walk(local_data_path):
        if 'blues' in dirs: # Found the root of the genre folders
            search_path = root
            break

    for genre in genres:
        genre_path = os.path.join(search_path, genre)
        if not os.path.exists(genre_path):
            log_progress(f"WARNING: Directory for genre '{genre}' not found at {genre_path}")
            continue
            
        # Get all audio files for this genre
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav') or f.endswith('.au')]
        files.sort() # Sort for determinism before shuffling
        
        # Set seed for reproducibility
        random.seed(42)
        random.shuffle(files)
        
        # GTZAN has 100 clips per genre. 8/1/1 split means 80 train, 10 val, 10 test per genre
        train_files = files[:80]
        val_files = files[80:90]
        test_files = files[90:]
        
        # Store relative paths
        for f in train_files: splits['train'].append(os.path.join(genre, f))
        for f in val_files: splits['val'].append(os.path.join(genre, f))
        for f in test_files: splits['test'].append(os.path.join(genre, f))

    # Save to Google Drive to ensure survival across session breaks
    log_progress(f"Saving split metadata to {split_file_path}...")
    with open(split_file_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    log_progress("Splits generated and saved safely to Drive.")
    
    # Validation Logging
    log_progress(f"Total Train clips: {len(splits['train'])} (Target: 800)")
    log_progress(f"Total Val clips: {len(splits['val'])} (Target: 100)")
    log_progress(f"Total Test clips: {len(splits['test'])} (Target: 100)")
    
    return splits

if __name__ == "__main__":
    log_progress("--- Starting HarmonicLIME Data Preparation ---")
    paths = setup_environment()
    
    if paths:
        local_base, split_file = paths
        dataset_splits = create_and_save_splits(local_base, split_file)
        log_progress("--- Phase 1 Complete. Ready for VGGish Fine-Tuning. ---")