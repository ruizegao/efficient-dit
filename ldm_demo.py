import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
sys.path.append('./latent-diffusion')
from taming.models import vqgan
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import torch

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("latent-diffusion/configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "latent-diffusion/models/ldm/cin256-v2/model.ckpt")
    return model


model = get_model()
sampler = DDIMSampler(model)


classes = [25, 187, 448, 992]   # define classes to be sampled here
n_samples_per_class = 6

ddim_steps = 20
ddim_eta = 0.0
scale = 3.0   # for unconditional guidance


all_samples = list()
class_images = {}
with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
            )

        for class_label in classes:
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class*[class_label])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples_per_class,
                                             shape=[3, 64, 64],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,
                                         min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)
            # Convert each image to numpy and store in dictionary
            images = []
            for img_tensor in x_samples_ddim:
                img_np = 255. * rearrange(img_tensor, 'c h w -> h w c').cpu().numpy()
                images.append(img_np.astype(np.uint8))  # Convert to uint8 format

            # Store all images for this class in dictionary
            class_images[f'class_{class_label}'] = images
            np.savez_compressed(f'./LDM_results/class_{class_label}.npz', images=images)


# # display as grid
# grid = torch.stack(all_samples, 0)
# grid = rearrange(grid, 'n b c h w -> (n b) c h w')
# grid = make_grid(grid, nrow=n_samples_per_class)

# # to image
# grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
# image = Image.fromarray(grid.astype(np.uint8))
# image.save("ldm_images.jpg")
