#!/usr/bin/env bash

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dsi_participant

echo "1) Load raw data"
python src/data.py --input data/raw/diabetes_data_upload.csv --out data/processed/raw_loaded.csv

echo "2) Preprocess"
python src/preprocess.py --input data/processed/raw_loaded.csv --out data/processed/processed.csv

echo "3) Train models"
python src/modeling.py --input data/processed/processed.csv --out models/best_model.joblib --figdir figures

echo "4) Evaluate"
python src/evaluate.py --model models/best_model.joblib --input data/processed/processed.csv --out reports/evaluation.txt

echo "5) SHAP analysis"
python src/shap_analysis.py --model models/best_model.joblib --input data/processed/processed.csv --figdir figures

echo "Done. Check figures/, models/, reports/"

