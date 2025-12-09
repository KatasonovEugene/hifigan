from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class TextDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        **kwargs,
    ):
        self.data_dir = ROOT_PATH / data_dir
        self.index_path = self.data_dir / f"inference_index.json"

        if self.index_path.exists():
            index = read_json(str(self.index_path))
        else:
            index = self._create_index()

        super().__init__(index, **kwargs)

    def _create_index(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)
        index = []
        text_dir = self.data_dir / "transcriptions"
        wav_dir = self.data_dir / "wavs"
        for file in text_dir.iterdir():
            text = file.read_text()
            name = file.stem
            wav_name = name + '.wav'
            if wav_dir.exists():
                wav_files = wav_dir.glob(wav_name)
                if len(wav_files) == 0:
                    audio_path = None
                else:
                    audio_path = str(wav_dir / wav_files[0])
            index.append({
                "text": text,
                "text_filename": file.name,
                "audio_path": audio_path,
            })
        write_json(index, str(self.index_path))
        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        text_filename = data_dict["text_filename"]
        text = data_dict["text"]
        audio_path = data_dict["audio_path"]
        audio = self.load_audio(audio_path)

        instance_data = {
            "gt_audio": audio,
            "sample_rate": self.target_sr,
            "text": text,
            "text_filename": text_filename,
        }
        instance_data = self.preprocess_data(instance_data)

        return instance_data
