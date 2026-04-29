# HarmonicLIME: Frequency-Aware Local Explanations for Deep Music Genre Classification

This repository contains the official implementation for the paper **"HarmonicLIME: Frequency-Aware Local Explanations for Deep Music Genre Classification"**.

HarmonicLIME is a music-aware local explainability framework that perturbs audio directly in the harmonic-percussive-residual (HPR) decomposed space. Unlike standard raw spectrogram-based XAI methods, HarmonicLIME produces component-level saliency maps that align with tonal and rhythmic musical structures, making deep learning decisions interpretable to musicians and domain experts.

---

## Repository Structure

The codebase is divided into three phases: GTZAN training and core evaluation, cross-dataset generalization on FMA-small, and supplementary analysis scripts added during peer review.

### Phase 1: GTZAN Pipeline

* `01_gtzan_data_prep.py` — Downloads and preprocesses the GTZAN dataset (converts to mono, resamples to 16 kHz, applies a 9.6-second center crop, and saves an 80/10/10 train/val/test split JSON to Drive for reproducibility).

* `02_vggish_finetune.py` — Fine-tunes a pre-trained VGGish model on the prepared GTZAN dataset for 10-class genre classification.

* `03_harmonic_lime.py` — Implements the core HarmonicLIME algorithm. Performs HPSS decomposition, generates $3 \times 10$ binary perturbation masks, and fits the Ridge regression surrogate to extract saliency weights for all GTZAN test clips. Automatically extracts GTZAN from Drive if local storage is unavailable (session recovery). Saves all weight files locally and zips to Drive on completion and every 100 clips as a checkpoint.

* `04_harmonic_lime_viz.py` — Generates the $3 \times 10$ time-component heatmaps (Harmonic, Percussive, Residual) for qualitative visual analysis. Produces Figure 1 of the paper.

* `05_deletion_auc.py` — Computes Deletion AUC for HarmonicLIME on the GTZAN test set. Saves per-clip AUC values to `gtzan_hl_auc_perclip.csv` on Drive for downstream statistical analysis.

* `06_baseline_auc.py` — Computes Deletion AUC for Mel-LIME and SHAP baselines on GTZAN. Saves per-clip values to `gtzan_baseline_auc_perclip.csv` on Drive.

* `07_statistics.py` — Loads per-clip CSVs from scripts 05 and 06 (and FMA_03) and computes genre-level mean ± SD, Wilcoxon signed-rank tests (HarmonicLIME vs each baseline), and corrected genre win counts. Outputs LaTeX-ready table rows. **Run after scripts 05, 06, and FMA_03.**

### Phase 2: FMA-small Zero-Shot Generalization

* `FMA_01_preprocess.py` — Downloads FMA metadata, randomly samples 100 clips per genre (seed=42), decodes MP3s, and applies GTZAN-equivalent preprocessing.

* `FMA_02_harmoniclime.py` — Applies HarmonicLIME to FMA-small using the frozen GTZAN-fine-tuned VGGish model (zero-shot transfer).

* `FMA_03_evaluate.py` — Computes zero-shot Deletion AUC and HLR/PLR/RLR metrics across HarmonicLIME, Mel-LIME, and SHAP on FMA-small. Saves per-clip metrics to `fma_perclip_metrics.csv` on Drive.

### Phase 3: Supplementary Analysis (Peer Review)

* `08_rms_energy_baseline.py` — Computes per-component RMS energy ratios (HER/PER/RER) for each GTZAN test clip and compares against HarmonicLIME saliency ratios (HLR/PLR/RLR). Demonstrates that saliency diverges from raw energy distribution, confirming HarmonicLIME identifies discriminative rather than merely energetic content. **CPU only. Runtime: < 5 minutes.**

* `09_hyperparam_sensitivity.py` — Hyperparameter sensitivity analysis over $N \in \{100, 500, 1000\}$ perturbations on a 50-clip stratified subset (5 per genre). Reports saliency stability (mean pairwise cosine similarity across repeated runs) and Deletion AUC at each N, justifying $N=500$ as the operating point. **GPU recommended. Runtime: ~60 minutes.**

* `10_classical_residual_inspection.py` — Spectral inspection of residual component $x_R$ for Classical vs Country test clips. Computes spectral flatness, high-frequency energy ratio, temporal variance, and residual RMS, with Mann-Whitney significance tests. Generates mean residual spectrogram figure. Provides direct evidence for the Clever Hans anomaly interpretation. **CPU only. Runtime: < 3 minutes.**

* `11_audiolime_comparison.py` — Empirical comparison against audioLIME using Open-Unmix source separation (vocals, drums, bass, other) as perturbation units. Runs on a 50-clip stratified subset. Saves per-clip Deletion AUC to `audiolime_perclip.csv` on Drive after every completed clip (resume-safe). Outputs LaTeX table options for paper. **GPU required. Runtime: ~2–3 hours.**

---

## Prerequisites & Installation

Developed and tested in Python 3.10+ on Google Colab with an NVIDIA T4 GPU.

Required libraries:
```bash
pip install torch torchaudio torchvision
pip install resampy librosa soundfile numpy pandas scikit-learn tqdm matplotlib scipy
pip install openunmix   # required for script 11 only
```

Note: The VGGish model is loaded via PyTorch Hub (`harritaylor/torchvggish`). Weights are downloaded automatically on first run (~275 MB).

---

## Usage Instructions

Execute scripts sequentially within each phase. Phase 3 scripts are independent and can be run in any order after Phase 1 is complete.

### 1. Model Training & Baseline Setup
```bash
python 01_gtzan_data_prep.py
python 02_vggish_finetune.py
```

### 2. Generate Explanations & Evaluate (GTZAN)
```bash
python 03_harmonic_lime.py
python 04_harmonic_lime_viz.py
python 05_deletion_auc.py
python 06_baseline_auc.py
python 07_statistics.py
```

### 3. Cross-Dataset Generalization (FMA-small)
```bash
python FMA_01_preprocess.py
python FMA_02_harmoniclime.py
python FMA_03_evaluate.py
python 07_statistics.py   # re-run after FMA_03 to include FMA statistics
```

### 4. Supplementary Analysis
```bash
python 08_rms_energy_baseline.py
python 09_hyperparam_sensitivity.py
python 10_classical_residual_inspection.py
python 11_audiolime_comparison.py
```

---

## Drive Directory Structure

All scripts read from and write to `/content/drive/MyDrive/HarmonicLIME/` unless otherwise noted. Expected layout:

```
MyDrive/
├── datasets/
│   └── GTZAN.zip
├── HarmonicLIME/
│   ├── vggish_best_model.pth
│   ├── dataset_splits.json
│   ├── lime_results/
│   │   └── lime_results.zip
│   ├── fma_lime_results/
│   ├── gtzan_hl_auc_perclip.csv
│   ├── gtzan_baseline_auc_perclip.csv
│   ├── fma_perclip_metrics.csv
│   ├── gtzan_rms_vs_saliency.csv
│   ├── sensitivity_results.csv
│   ├── classical_residual_analysis.csv
│   ├── classical_residual_figure.pdf
│   └── audiolime_perclip.csv
```

---

## Hardware Requirements Summary

| Script | GPU Required | Estimated Runtime |
|--------|-------------|-------------------|
| 01–02  | Recommended | ~30 min           |
| 03     | Required    | ~6–8 hours        |
| 04     | No          | < 5 min           |
| 05–06  | Required    | ~2–3 hours        |
| 07     | No          | < 1 min           |
| FMA_01–03 | Required | ~4–5 hours       |
| 08     | No          | < 5 min           |
| 09     | Required    | ~60 min           |
| 10     | No          | < 3 min           |
| 11     | Required    | ~2–3 hours        |
