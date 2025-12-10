from pathlib import Path
import torch
import sys
import hydra
from hydra.utils import instantiate

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import TTS, HiFiGAN
from src.model.hifigan_blocks.tacotron2 import Tacotron2MelGenerator 

@hydra.main(version_base=None, config_path=".", config_name="cfg")
def make_tts(cfg):
    info = torch.load(cfg.path, map_location='cpu', weights_only=False)
    weights = info['state_dict']
    model = TTS(
        acoustic_model=Tacotron2MelGenerator(),
        vocoder=instantiate(cfg.hifigan),
    )
    model.vocoder.load_state_dict(weights)
    rpath = Path(cfg.path)
    new_name = 'tts_' + rpath.name
    torch.save(model.state_dict(), str(rpath.parent / new_name))

if __name__ == "__main__":
    make_tts()