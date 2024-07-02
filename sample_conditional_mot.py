import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.utils import make_grid

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import DDPM
from ldm.util import instantiate_from_config


def parse_args():
    # New parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt_path', type=str, default='/media/NAS/nas_32/dongkyu/Workspace/latent-diffusion/logs')
    parser.add_argument('--ckpt', type=str, default='2024-06-19T11-18-25_cspine-ldm-vq-f8-cond-mot')
    parser.add_argument('--ddim_steps', default=200, type=int, help='DDIM steps.')
    parser.add_argument('--ddim_eta', default=0.5, type=float, help='DDIM(0.0) or DDPM(1.0).')
    parser.add_argument('--scale', default=-3.0, type=float, help='node rank for distributed training')
    parser.add_argument('--n_samples', default=-20, type=int, help='node rank for distributed training')
    return parser.parse_args()


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(path):
    config_file = path.split('/')[-1].split('_')[0]
    path_conf = f"{path}/configs/{config_file}-project.yaml"

    path_ckpt_list = os.listdir(f"{path}/checkpoints/")
    path_ckpt_list.sort()
    path_ckpt = path_ckpt_list[-2]

    path_ckpt = f"{path}/checkpoints/{path_ckpt}"
    config = OmegaConf.load(path_conf)
    model = load_model_from_config(config, path_ckpt)
    return model


if __name__ == "__main__":
    params = parse_args()
    name = params.ckpt.split('_')[-1]

    mean_A = Image.open('samples/knuh-cspine-mean/0.png').resize((224, 224), Image.Resampling.BICUBIC).convert('RGB')
    mean_B = Image.open('samples/knuh-cspine-mean/1.png').resize((224, 224), Image.Resampling.BICUBIC).convert('RGB')
    mean_C = Image.open('samples/knuh-cspine-mean/2.png').resize((224, 224), Image.Resampling.BICUBIC).convert('RGB')
    mean_D = Image.open('samples/knuh-cspine-mean/3.png').resize((224, 224), Image.Resampling.BICUBIC).convert('RGB')

    mean_A = np.array(mean_A).astype(np.float32)
    mean_B = np.array(mean_B).astype(np.float32)
    mean_C = np.array(mean_C).astype(np.float32)
    mean_D = np.array(mean_D).astype(np.float32)
    class_mean = [mean_A, mean_B, mean_C, mean_D]

    os.makedirs(f"samples/{name}", exist_ok=True)

    model = get_model(path=f'{params.ckpt_path}/{params.ckpt}')

    print("model is instance of DDPM: ", isinstance(model, DDPM))
    sampler = DDIMSampler(model)

    classes = [0, 1, 2, 3]  # define classes to be sampled here
    n_samples_per_class = params.n_samples

    ddim_steps = params.ddim_steps
    ddim_eta = params.ddim_eta
    scale = params.scale  # for unconditional guidance

    all_samples = list()
    latent_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class * [4]).to(model.device)}
            )

            for class_label in classes:
                print(
                    f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class * [class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                # Mean Offset Inference
                x_T = torch.tensor(n_samples_per_class * class_mean[class_label]).to(model.device)

                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[4, 32, 32],
                                                 x_T=x_T,
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta)

                latent_samples.append(samples_ddim)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class, pad_value=255)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    cv2.imwrite(f"samples/{name}/{name}-step{ddim_steps}-ddim_eta{ddim_eta}-scale{scale}.png", grid)
