from pathlib import Path
# --- add project root to sys.path ---
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # parent of /notebooks
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# ------------------------------------

from src.model_training import train_profit_classifier

if __name__ == "__main__":
    CSV_PATH = "data/raw/USSuperstoreData.csv"
    MODEL_OUT = "models/model_profit_clf.joblib"
    Path("models").mkdir(parents=True, exist_ok=True)

    result = train_profit_classifier(CSV_PATH, MODEL_OUT)
    print("Training done ✅")
    print("Metrics:", result["metrics"])
    print("Model saved to:", result["model_path"])
