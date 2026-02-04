import os

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass


def load_hydra_config(cfg_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    cfg = OmegaConf.create(cfg)
    return cfg


def load_config(model_path):
    cfg = omegaconf.OmegaConf.load(os.path.join(model_path, ".hydra/config.yaml"))
    cfg = OmegaConf.create(cfg)
    return cfg


def load_model(model_path, ckpt):
    cfg = load_config(model_path)
    model = hydra.utils.instantiate(cfg.model)
    loadpath = os.path.join(model_path, f"epoch={ckpt}.ckpt")
    checkpoint = torch.load(loadpath, map_location="cuda:0")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda()
    model.eval()
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def enable_gradient(model):
    for param in model.parameters():
        param.requires_grad = True


def load_diffusion_model(
    model_path, ckpt, load_pretrain_weight=True, use_ema=False, **kwargs
):
    model_cfg = load_config(model_path)
    model = hydra.utils.instantiate(model_cfg.model)
    if use_ema:
        print("Loading ema model checkpoints!")
        ckpt_path = os.path.join(model_path, "checkpoints", f"ema_epoch_{ckpt}.ckpt")
    else:
        ckpt_path = os.path.join(model_path, "checkpoints", f"epoch_{ckpt}.ckpt")
    if load_pretrain_weight:
        model.load_state_dict(torch.load(ckpt_path))
    model.to("cuda")
    model.eval()
    noise_scheduler = hydra.utils.instantiate(model_cfg.noise_scheduler)

    return model, noise_scheduler
