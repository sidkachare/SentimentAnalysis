import os
import yaml
import pandas as pd
import tarfile
import urllib.request
from pathlib import Path

def read_yaml(filepath):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)

def save_data(df, filepath):
    df.to_csv(filepath, index=False)

def download_file(url, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    urllib.request.urlretrieve(url, filepath)

def extract_tar(filepath, extract_dir):
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_dir)


