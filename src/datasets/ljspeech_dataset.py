import csv
import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LJSpeechDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        **kwargs,
    ):
        self.data_dir = ROOT_PATH / data_dir
        self.index_path = self.data_dir / f"train_index.json"

        if self.index_path.exists():
            index = read_json(str(self.index_path))
        else:
            index = self._create_index()

        super().__init__(index, **kwargs)

    def _create_index(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)
        index = []
        with open(self.data_dir / "metadata.csv", "r") as metadata_file:
            for file, _, _ in csv.reader(metadata_file, delimiter='|', quotechar='|'):
                file += '.wav'
                audio_path = str(self.data_dir / "wavs" / file)
                index.append({
                    'audio_path': audio_path,
                })
        write_json(index, str(self.index_path))
        return index
