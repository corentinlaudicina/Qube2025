# Qube2025 Survival Analysis Challenge 🧬

This repository contains code for modeling and predicting patient survival using clinical and molecular data, as part of the Qube 2025 data science challenge.

## 📁 Project Structure

- `DataStructs.py`: Defines the `Patient` and `SomaticMutation` classes, and functions to load and preprocess data.
- `main.py`: Data loading, feature extraction, preprocessing, modeling pipeline (LightGBM).
- `LightGBM_model.ipynb`: Notebook version of the main workflow with exploration and visualization.
- `train/`, `test/`: Input data directories.

## 🚀 Features Used

- Clinical features: BM_BLAST, HB, PLT, WBC, ANC, MONOCYTES
- Aggregated mutation features: VAF stats, mutation count (in progress)
- Cytogenetic parsing and risk grouping (planned)

## 🧠 Models

- LightGBM regressor
- Evaluation via survival metrics (IPCW concordance index)

## 🛠️ Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
