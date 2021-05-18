import hydra
import torch

from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.model import DeepSpeech
import numpy as np
import os
import pickle
import random
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model(device,
               model_path):
    model = DeepSpeech.load_from_checkpoint(hydra.utils.to_absolute_path(model_path))
    # model = DeepSpeech.load_from_checkpoint(hydra.utils.to_absolute_path(
    #     "/Users/gt/Documents/GitHub/deepspeech.pytorch/data/librispeech_pretrained_v3.ckpt"))
    state_dict = model.state_dict()

    ## The following code was used to generate indices for random permutation ##
    # d_rand_idx = {}  # create dict for storing the indices for random permutation
    # for k, v in state_dict.items():
    #     w = state_dict[k]
    #     idx = torch.randperm(w.nelement())  # create random indices across all dimensions
    #     d_rand_idx[k] = idx
    #
    # with open(os.path.join(os.getcwd(), 'DS2_randnetw_indices.pkl'), 'wb') as f:
    #     pickle.dump(d_rand_idx, f)

    print('OBS! RANDOM NETWORK!')

    for k, v in state_dict.items():
        w = state_dict[k]
        # Load random indices
        print(f'________ Loading random indices from permuted architecture for {k} ________')
        d_rand_idx = pickle.load(open(os.path.join('/Users/gt/Documents/GitHub/deepspeech.pytorch/deepspeech_pytorch', 'DS2_randnetw_indices.pkl'), 'rb'))
        idx = d_rand_idx[k]
        rand_w = w.view(-1)[idx].view(w.size()) # permute, and reshape back to original shape
        state_dict[k] = rand_w
    
    model.load_state_dict(state_dict)   # map_location=torch.device('cpu'))
    model.eval()
    model = model.to(device)
    
    return model


def load_decoder(labels, cfg: LMConfig):
    if cfg.decoder_type == DecoderType.beam:
        from deepspeech_pytorch.decoder import BeamCTCDecoder
        if cfg.lm_path:
            cfg.lm_path = hydra.utils.to_absolute_path(cfg.lm_path)
        decoder = BeamCTCDecoder(labels=labels,
                                 lm_path=cfg.lm_path,
                                 alpha=cfg.alpha,
                                 beta=cfg.beta,
                                 cutoff_top_n=cfg.cutoff_top_n,
                                 cutoff_prob=cfg.cutoff_prob,
                                 beam_width=cfg.beam_width,
                                 num_processes=cfg.lm_workers,
                                 blank_index=labels.index('_'))
    else:
        decoder = GreedyDecoder(labels=labels,
                                blank_index=labels.index('_'))
    return decoder


def remove_parallel_wrapper(model):
    """
    Return the model or extract the model out of the parallel wrapper
    :param model: The training model
    :return: The model without parallel wrapper
    """
    # Take care of distributed/data-parallel wrapper
    model_no_wrapper = model.module if hasattr(model, "module") else model
    return model_no_wrapper
