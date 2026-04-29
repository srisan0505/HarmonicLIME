# ==============================================================================
# File Name: 02_vggish_finetune.py
# Version: 1.3 (Added Corrupted File Handling)
# Description: 
#   Step 2 of the HarmonicLIME pipeline. 
#   Loads pre-trained VGGish, wraps it cleanly to preserve its internal dense
#   layers, and fine-tunes on GTZAN for 10-way genre classification.
#   Added try-except block in Dataset to handle the famously corrupted
#   jazz.00054.wav file natively without crashing.
# ==============================================================================

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from google.colab import drive
drive.mount('/content/drive')

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_CLASSES = 10
SAMPLE_RATE = 16000 
TARGET_LENGTH = int(9.6 * SAMPLE_RATE) 

# Paths
DRIVE_BASE = '/content/drive/MyDrive/HarmonicLIME'
LOCAL_BASE = '/content/gtzan_local/Data/genres_original'
SPLIT_FILE = os.path.join(DRIVE_BASE, 'dataset_splits.json')
CHECKPOINT_PATH = os.path.join(DRIVE_BASE, 'vggish_checkpoint.pth')
BEST_MODEL_PATH = os.path.join(DRIVE_BASE, 'vggish_best_model.pth')

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

def log_progress(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

class GTZANDataset(Dataset):
    def __init__(self, file_list, local_base):
        self.file_list = file_list
        self.local_base = local_base
        self.preprocessor = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=True)
        self.preprocessor.eval()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rel_path = self.file_list[idx]
        file_path = os.path.join(self.local_base, rel_path)
        
        # Robust loading to handle corrupted files (like jazz.00054.wav)
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            # If it fails, load the next file in the list instead
            next_idx = (idx + 1) % len(self.file_list)
            return self.__getitem__(next_idx)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
            
        if waveform.shape[1] > TARGET_LENGTH:
            start = (waveform.shape[1] - TARGET_LENGTH) // 2
            waveform = waveform[:, start:start+TARGET_LENGTH]
        else:
            padding = TARGET_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        waveform_np = waveform.squeeze(0).numpy()
        
        with torch.no_grad():
            mel_tensor = self.preprocessor._preprocess(waveform_np, SAMPLE_RATE)

        genre_label = rel_path.split('/')[0] if '/' in rel_path else rel_path.split('\\')[0]
        label_idx = GENRE_TO_IDX[genre_label]
        
        return mel_tensor, torch.tensor(label_idx, dtype=torch.long)

class VGGishGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', preprocess=False, postprocess=False)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.vggish(x)
        out = self.classifier(features)
        return out

def build_model():
    log_progress("Loading pre-trained VGGish for training...")
    return VGGishGenreClassifier(NUM_CLASSES)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress(f"Using device: {device}")
        
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)
        
    search_path = '/content/gtzan_local'
    for root, dirs, files in os.walk(search_path):
        if 'blues' in dirs:
            local_base = root
            break
            
    train_dataset = GTZANDataset(splits['train'], local_base)
    val_dataset = GTZANDataset(splits['val'], local_base)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start_epoch = 0
    best_val_acc = 0.0
    
    if os.path.exists(CHECKPOINT_PATH):
        log_progress(f"Found checkpoint at {CHECKPOINT_PATH}. Resuming training...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        log_progress(f"Resumed from epoch {start_epoch}. Previous best val accuracy: {best_val_acc:.4f}")
    
    log_progress("Starting Training Loop...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            current_batch_size = inputs.shape[0]
            num_frames = inputs.shape[1]
            
            inputs = inputs.view(-1, 1, 96, 64).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            outputs = outputs.view(current_batch_size, num_frames, NUM_CLASSES).mean(dim=1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                current_batch_size = inputs.shape[0]
 
 if __name__ == "__main__":
    train()
    