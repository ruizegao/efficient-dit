# install diffusers timm
# run : python dit_demo.py --steps 250 --class_labels '88'
# DiT imports:
import torch
from torchvision.utils import save_image
from DiT.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from DiT.download import find_model
from DiT.models import DiT_XL_2
from PIL import Image
import os
import argparse
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

def run_dit(image_size, model, vae, steps, class_labels):
    # Set user inputs:
    seed = 0 #@param {type:"number"}
    torch.manual_seed(seed)
    num_sampling_steps = steps #@param {type:"slider", min:0, max:1000, step:1}
    cfg_scale = 4 #@param {type:"slider", min:1, max:10, step:0.1}
    class_labels = class_labels #@param {type:"raw"}
    samples_per_row = 4 #@param {type:"number"}
    class_labels = [int(x) for x in class_labels.split(',')]
    # Create diffusion object:
    diffusion = create_diffusion(str(num_sampling_steps))

    # Create sampling noise:
    n = len(class_labels)
    latent_size = int(image_size) // 8
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, 
        model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    folder = "./Dit_results"
    file_path = os.path.join(folder, f"step:{steps},class:{class_labels}.png")

    # Create the folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_image(samples, file_path, nrow=int(samples_per_row), 
            normalize=True, value_range=(-1, 1))
    samples = Image.open(file_path)

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--steps', type = int,  default= 250)

    parser.add_argument('--class_labels', type=str, default='88', help='Comma separated class labels')
    
    args = parser.parse_args()

    image_size = 256 #@param [256, 512]
    vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    latent_size = int(image_size) // 8
    # Load model:
    model = DiT_XL_2(input_size=latent_size).to(device)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval() # important!
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    run_dit(image_size, model, vae, args.steps, args.class_labels)

if __name__ == '__main__':
    main()
    