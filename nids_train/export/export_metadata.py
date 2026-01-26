# export/export_metadata.py
import json

def save_metadata(metadata, path):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)