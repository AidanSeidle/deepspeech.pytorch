import json
from typing import List

import hydra
import torch
from torch.cuda.amp import autocast

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.decoder import Decoder
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.utils import load_decoder, load_model

# import matplotlib.pyplot as plt
import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path
import os
import pickle
from scipy.io import wavfile
import random
import torch
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

RESULTDIR = "C:/Users/ajseidle/Documents/GitHub/auditory_brain_dnn/aud_dnn/full-soundset-actv/DS2/"

def decode_results(decoded_output: List,
                   decoded_offsets: List,
                   cfg: TranscribeConfig):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "path": cfg.model.model_path
            },
            "language_model": {
                "path": cfg.lm.lm_path
            },
            "decoder": {
                "alpha": cfg.lm.alpha,
                "beta": cfg.lm.beta,
                "type": cfg.lm.decoder_type.value,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(cfg.lm.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if cfg.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def transcribe(cfg: TranscribeConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )

    spect_parser = SpectrogramParser(
        audio_conf=model.spect_cfg,
        normalize=True
    )

    decoded_output, decoded_offsets = run_transcribe(
        audio_path=hydra.utils.to_absolute_path(cfg.audio_path),
        spect_parser=spect_parser,
        model=model,
        decoder=decoder,
        device=device,
        precision=cfg.model.precision
    )
    results = decode_results(
        decoded_output=decoded_output,
        decoded_offsets=decoded_offsets,
        cfg=cfg
    )
    print(json.dumps(results))


def run_transcribe(audio_path: str,
                   spect_parser: SpectrogramParser,
                   model: DeepSpeech,
                   decoder: Decoder,
                   device: torch.device,
                   precision: int):
    
    spect = spect_parser.parse_audio(audio_path).contiguous()
    
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    
    # Save outputs
    save_output = SaveOutput()

    hook_handles = []
    layer_names = []
    for idx, layer in enumerate(model.modules()):
        layer_names.append(layer)
        # print(layer)
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            print('Fetching conv handles!')
            handle = layer.register_forward_hook(save_output) # save idx and layer
            hook_handles.append(handle)

        if type(layer) == torch.nn.LSTM:
            print('Fetching rnn handles!')
            handle = layer.register_forward_hook(save_output) # save idx and layer
            hook_handles.append(handle)
            
        if type(layer) == torch.nn.Linear:
            print('Fetching fc handles!')
            handle = layer.register_forward_hook(save_output) # save idx and layer
            hook_handles.append(handle)
            
        if isinstance(layer, torch.nn.modules.BatchNorm2d):
            print('Fetching batch norm handles!')
            handle = layer.register_forward_hook(save_output)  # save idx and layer
            hook_handles.append(handle)
        
        if isinstance(layer, torch.nn.modules.BatchNorm1d):
            print('Fetching batch norm handles!')
            handle = layer.register_forward_hook(save_output)  # save idx and layer
            hook_handles.append(handle)
            
        if isinstance(layer, torch.nn.Hardtanh):
            print('Fetching tanH handles!')
            handle = layer.register_forward_hook(save_output)  # save idx and layer
            hook_handles.append(handle)

    
    print(f'Number of hooks for selected layers: {len(hook_handles)}')

    with autocast(enabled=precision == 16): # forward pass -- getting the outputs
        out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)

    # detach activations
    detached_activations = save_output.detach_activations()
    
    # store and save activations
    # get identifier (sound file name)
    id1 = audio_path.split('/')[-1]
    identifier = id1.split('.')[0]
    
    save_output.store_activations(RESULTDIR=RESULTDIR, identifier=identifier)
    
    return decoded_output, decoded_offsets


class SaveOutput:
    def __init__(self):
        self.outputs = []
        self.activations = {} # create a dict with module name
        self.detached_activations = None
    
    def __call__(self, module, module_in, module_out):
        """
        Module in has the input tensor, module out in after the layer of interest
        """
        self.outputs.append(module_out)
        
        layer_name = self.define_layer_names(module)
        self.activations[layer_name] = module_out
    
    def define_layer_names(self, module):
        layer_name = str(module)
        current_layer_names = list(self.activations.keys())
    
        split_layer_names = [l.split('--') for l in current_layer_names]
    
        num_occurences = 0
        for s in split_layer_names:
            s = s[0]  # base name
        
            if layer_name == s:
                num_occurences += 1
    
        layer_name = str(module) + f'--{num_occurences}'
    
        if layer_name in self.activations:
            warnings.warn('Layer name already exists')
            
        return layer_name
    
    def clear(self):
        self.outputs = []
        self.activations = {}
        
    def get_existing_layer_names(self):
        for k in self.activations.keys():
            print(k)
        
        return list(self.activations.keys())
        
    def return_outputs(self):
        self.outputs.detach().numpy()
        
    def detach_one_activation(self, layer_name):
        return self.activations[layer_name].detach().numpy()
        
    def detach_activations(self, lstm_output='cell'):
        """
        Detach activations (from tensors to numpy)
        
        Arguments:
            lstm_output: for LSTM, can output either the hidden states throughout hidden ('hidden')
                        or the most cell hidden states ('cell')
            
        Returns:
            detached_activations = for each layer, the flattened activations
            packaged_data = for LSTM layers, the packaged data
        """
        detached_activations = {}
        detached_packaged_data = {}

        for k,v in self.activations.items():
            print(f'Detaching activation for layer: {k}')
            if k.startswith('Conv2d') or k.startswith('BatchNorm2d') or k.startswith('Hardtanh'): # all of these are 4d tensors
                activations = v.detach().numpy()
                # squeeze batch dimension
                avg_activations = activations.squeeze()
                # expand (flatten) the channel x kernel dimension:
                avg_activations = avg_activations.reshape(
                    [avg_activations.shape[0] * avg_activations.shape[1], avg_activations.shape[2]])
                # mean over time
                avg_activations = avg_activations.mean(axis=1)
                
                detached_activations[k] = avg_activations
                
            if k.startswith('LSTM'): # packaged data available
                packaged_data = v[0].data.detach().numpy()
                detached_packaged_data[k] = packaged_data
                activations = v[1]
                
                # get both LSTM outputs
                activations_hidden = activations[0].detach().numpy()
                activations_cell = activations[1].detach().numpy()

                # squeeze batch dimension
                avg_activations_hidden = activations_hidden.squeeze()
                avg_activations_cell = activations_cell.squeeze()
                
                # CONCATENATE over the num directions dimension:
                avg_activations_hidden = avg_activations_hidden.reshape(-1)
                avg_activations_cell = avg_activations_cell.reshape(-1)

                detached_activations[f'{k}--hidden'] = avg_activations_hidden
                detached_activations[f'{k}--cell'] = avg_activations_cell

            if k.startswith('Linear') or k.startswith('BatchNorm1d'):
                activations = v.detach().numpy()
                # mean over time dimension
                avg_activations = activations.mean(axis=0)
                detached_activations[k] = avg_activations
        
        self.detached_activations = detached_activations
            
        return detached_activations
    
    def store_activations(self, RESULTDIR, identifier):
        RESULTDIR = (Path(RESULTDIR))

        if not (Path(RESULTDIR)).exists():
            os.makedirs((Path(RESULTDIR)))
        
        # filename = os.path.join(RESULTDIR, f'{identifier}_activations_randnetw.pkl')
        filename = os.path.join(RESULTDIR, f'{identifier}_activations.pkl')

        with open(filename, 'wb') as f:
            pickle.dump(self.detached_activations, f)
    