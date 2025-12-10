import os
import urllib.request
import gzip
import shutil
from pathlib import Path
import gdown


MODELS_URLS = {
    "https://drive.google.com/uc?id=1aYNZW-ZU8rxksWZVMxSvY89y5_GMPhyW": "tts_model_best.pth",
    "https://drive.google.com/uc?id=1WuGIr8e0opPCSWrZ0flDxBwj6hySx_cH": "8k_tts_model_best.pth",
}


def download_models(path="saved"):
    dir_path = Path(path).absolute().resolve()
    dir_path.mkdir(exist_ok=True, parents=True)

    for url, filepath in MODELS_URLS.items():
        path = str((dir_path / filepath).absolute().resolve())
        gdown.download(url, path)
        print("Model downloaded to", path)


if __name__ == "__main__":
    download_models()