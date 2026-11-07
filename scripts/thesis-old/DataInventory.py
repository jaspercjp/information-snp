"""
    Used to quickly summarize the model data present in the specified folder
"""
import sys, os

MODEL_DATA_PATH = "../data/models"
print("SUMMARY OF DATA:")
for model in os.listdir(MODEL_DATA_PATH):
    model_data = os.listdir(os.path.join(MODEL_DATA_PATH, model, "psl"))
    dates = model_data[0].split("_")[-1].replace(".nc", "")
    print(f"{model}, {len(model_data)} members, {dates}")