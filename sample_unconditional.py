import cv2
import torch
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import DDPM
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    # return {"model": model}, global_step
    return model


def get_model(path):
    config_file = path.split('/')[-1].split('_')[0]
    path_conf = f"{path}/configs/{config_file}-project.yaml"
    path_ckpt = f"{path}/checkpoints/epoch=000334.ckpt"
    config = OmegaConf.load(path_conf)
    model = load_model_from_config(config, path_ckpt)
    return model

model = get_model(path='/media/NAS/nas_32/dongkyu/Workspace/latent-diffusion/logs/2024-05-08T15-34-58_cspine-ldm-vq-f8-uncond')

print("model is instance of DDPM: ", isinstance(model, DDPM))
sampler = DDIMSampler(model)

from einops import rearrange
from torchvision.utils import make_grid

n_samples_per_class = 80
n_samples_per_row = 20

ddim_steps = 200
ddim_eta = 0.0
scale = 1.0   # for unconditional guidance


all_samples = list()
latent_samples = list()

with torch.no_grad():
    with model.ema_scope():
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                         batch_size=n_samples_per_class,
                                         shape=[4, 32, 32],
                                         verbose=False,
                                         eta=ddim_eta)

        latent_samples.append(samples_ddim)
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        all_samples.append(x_samples_ddim)


# display as grid
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=n_samples_per_row, pad_value=255)

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
cv2.imwrite("./latent-unconditional.png", grid)  # vq-f8-cond-vq-default
