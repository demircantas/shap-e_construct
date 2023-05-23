import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image

import imageio
from PIL import Image
from PIL import ImageOps
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('decoder', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

model = load_model('image300M', device=device)

# Specify the path to the folder in your Google Drive where the images are located
# folder_path = "/content/drive/MyDrive/thesis/images/stanford_cars"
# folder_path = "/content/drive/MyDrive/thesis/images/shapeNet_generated"
folder_path = "images/input"
output_dir = "images/gifs"

# Get the file paths of all images in the folder
image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]
images_b = []

# Iterate over the image paths and process the images
for image_path in image_paths:
    # Open the image using PIL
    image = Image.open(image_path)

    # Add any additional processing or display logic for the images
    # ...

    border_size = int(image.width / 3)
    print(f'bs: {border_size}')
    image = ImageOps.expand(image, border=(border_size, border_size, border_size, border_size), fill='white')
    # Display the image
    # image.show()
    images_b.append(image)

images_c = images_b
for image in images_c:
  image.show()

batch_size = len(images_c)
# batch_size = 1
guidance_scale = 2.3 # default = 3.0, 2.0 for cow

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    # model_kwargs=dict(images=[images_c[0]] * batch_size),
    model_kwargs=dict(images=images_c[:batch_size]),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

render_mode = 'nerf' # you can change this to 'stf' for mesh rendering
size = 96 # this is the size of the renders; higher values take longer to render.

cameras = create_pan_cameras(size, device)

# ...

# Modify the last line of your code
for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    images = [image.resize((image.width * 3, image.height * 3)) for image in images]
    # display(gif_widget(images))

    # write gif to drive
    gif_path = os.path.join(output_dir, f"image_{i}.gif")
    imageio.mimsave(gif_path, [np.array(image) for image in images], duration=0.2)

# steps of latent interpolation
n = 10

weights = torch.linspace(0, 1, n).unsqueeze(1).to(device)

interpolated_latents = torch.lerp(latents[0], latents[1], weights)

# Reshape the tensor to have shape (n, 1000)
interpolated_latents = interpolated_latents.view(n, -1)

render_mode = 'nerf' # you can change this to 'stf' for mesh rendering
size = 96 # this is the size of the renders; higher values take longer to render.

cameras = create_pan_cameras(size, device)

# ...

# Modify the last line of your code
for i, latent in enumerate(interpolated_latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    images = [image.resize((image.width * 3, image.height * 3)) for image in images]
    # display(gif_widget(images))

    # write gif to drive
    gif_path = os.path.join(output_dir, f"image_interpolated_a{i}.gif")
    imageio.mimsave(gif_path, [np.array(image) for image in images], duration=0.2)
