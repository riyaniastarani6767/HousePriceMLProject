from pathlib import Path
from joblib import dump, load

def save_model(pipeline, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, path)

def load_model(path: str):
    return load(path)
