"""
    Used to quickly summarize the model data present in the specified folder
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import sys, os

MODEL_DATA_PATH = "../data/models"
print("SUMMARY OF DATA:")
for model in os.listdir(MODEL_DATA_PATH):
    model_data = os.listdir(os.path.join(MODEL_DATA_PATH, model, "psl"))
    dates = model_data[0].split("_")[-1].replace(".nc", "")
    print(f"{model}, {len(model_data)} members, {dates}")