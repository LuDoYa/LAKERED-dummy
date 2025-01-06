import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.init as init
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat
from argparse import ArgumentParser

def combine(args):
    config_ldm = OmegaConf.load(args.config)

    # Create LDM model and load initial weights
    model = instantiate_from_config(config_ldm.model)
    state_dict = torch.load(args.ldm)['state_dict']

    # Initialize conv_f weights and biases
    conv_f_weight = torch.empty(3, 3, 1, 1)  # Kernel size is 1x1
    conv_f_bias = torch.empty(3)

    # Xavier initialization
    init.xavier_uniform_(conv_f_weight)
    init.zeros_(conv_f_bias)

    conv_b_weight = torch.empty(3, 3, 1, 1)
    conv_b_bias = torch.empty(3)
    init.xavier_uniform_(conv_b_weight)
    init.zeros_(conv_b_bias)

    conv_e_weight = torch.empty(3, 3, 1, 1)
    conv_e_bias = torch.empty(3)
    init.xavier_uniform_(conv_e_weight)
    init.zeros_(conv_e_bias)

    conv_h_weight = torch.empty(3, 3, 1, 1)
    conv_h_bias = torch.empty(3)
    init.xavier_uniform_(conv_h_weight)
    init.zeros_(conv_h_bias)

    state_dict["model.SBG_module.PSF.conv_f.weight"] = conv_f_weight
    state_dict["model.SBG_module.PSF.conv_f.bias"] = conv_f_bias

    state_dict["model.SBG_module.PSF.conv_b.weight"] = conv_b_weight
    state_dict["model.SBG_module.PSF.conv_b.bias"] = conv_b_bias

    state_dict["model.SBG_module.PSF.conv_e.weight"] = conv_e_weight
    state_dict["model.SBG_module.PSF.conv_e.bias"] = conv_e_bias

    state_dict["model.SBG_module.PSF.conv_h.weight"] = conv_h_weight
    state_dict["model.SBG_module.PSF.conv_h.bias"] = conv_h_bias

    state_dict["model_ema.SBG_modulePSFconv_fweight"] = conv_f_weight
    state_dict["model_ema.SBG_modulePSFconv_fbias"] = conv_f_bias

    state_dict["model_ema.SBG_modulePSFconv_bweight"] = conv_b_weight
    state_dict["model_ema.SBG_modulePSFconv_bbias"] = conv_b_bias

    state_dict["model_ema.SBG_modulePSFconv_eweight"] = conv_e_weight
    state_dict["model_ema.SBG_modulePSFconv_ebias"] = conv_e_bias

    state_dict["model_ema.SBG_modulePSFconv_hweight"] = conv_h_weight
    state_dict["model_ema.SBG_modulePSFconv_hbias"] = conv_h_bias

    # Save the updated model
    torch.save({'state_dict': model.state_dict()}, args.savemodel)
    print(f"Updated model saved to {args.savemodel}")

if __name__ == '__main__':
    class Args:
        config = "ldm/models/ldm/inpainting_big/config_LAKERED.yaml"
        ldm = "ldm/models/ldm/inpainting_big/LAKERED_init.ckpt"
        savemodel = "ldm/models/ldm/inpainting_big/LAKERED_final.ckpt"

    combine(Args)
