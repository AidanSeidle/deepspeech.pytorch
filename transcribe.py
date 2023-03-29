import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.inference import transcribe

from os import listdir
from os.path import isfile, join
import numpy as np
import random
import torch
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)

DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/'

files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]
wav_files = [f for f in files if f.endswith('wav')]

@hydra.main(config_name="config")
def hydra_main(cfg: TranscribeConfig):
    # run across several audio files

    # generate cfgs
    cfg_all = {}
    for file in wav_files:
        print(file)
        cfg_copy = cfg.copy()
        cfg_copy['audio_path'] = DATADIR + file
        cfg_all[file] = cfg_copy
    
    for file_name, cfg in cfg_all.items():
        transcribe(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
