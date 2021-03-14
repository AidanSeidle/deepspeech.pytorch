import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.inference import transcribe

from os import listdir
from os.path import isfile, join
import numpy as np

cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)

DATA_DIR = '/Users/gt/Documents/GitHub/asr/decoding/165_natural_sounds/165_natural_sounds/'

files = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
wav_files = [f for f in files if f.endswith('wav')]


@hydra.main(config_name="config")
def hydra_main(cfg: TranscribeConfig):
    # run across several audio files
    sorted_wav = np.sort(wav_files)

    cfg_all = {}
    for file in wav_files:
        cfg_copy = cfg.copy()
    
        cfg_copy['audio_path'] = DATA_DIR + file
    
        cfg_all[file] = cfg_copy
    
    transcribe(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
