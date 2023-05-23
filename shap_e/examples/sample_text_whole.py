import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

import imageio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('decoder', device=device) # 'decoder' is the other, lighter option. 'transmitter' is the default.
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

batch_size = 1 # determines how many samples are generated
guidance_scale = 15.0 # is a hyperparameter that changes how well the model works based on context
prompt = "a red plastic bottle"

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

render_mode = 'nerf'  # you can change this to 'stf'
size = 96  # this is the size of the renders; higher values take longer to render. # the default was 64.

cameras = create_pan_cameras(size, device)
for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    images = [image.resize((image.width * 4, image.height * 4)) for image in images] # I modified this. makes the images 4 times bigger
    # display(gif_widget(images))
    imageio.mimsave(f'images/test{i}.gif', images, 'GIF') # images here indicate the output of decode_latent_images
