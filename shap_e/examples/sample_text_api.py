import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

import imageio
from flask import Flask, request
from PIL import Image, ImageOps

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('decoder', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

render_mode = 'nerf'
size = 64 # 64

cameras = create_pan_cameras(size, device)


@app.route('/construct', methods=['POST'])
def generate_images():
    sentence = request.form.get('sentence')
    print(f'\033[94mCONSTRUCT({sentence}) initiated...\033[0m')

    batch_size = 1
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[sentence] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=16, # 64
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    print(f'\033[93mrendering {sentence}...\033[0m')

    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        images = [image.resize((image.width * 4, image.height * 4)) for image in images]
        image_path = f'images/test{i}.gif'
        imageio.mimsave(image_path, images, 'GIF', loop=0)
        # display(Image(filename=image_path))

    print('rendering complete')
    image = Image.open('images/test0.gif')
    image.show()

    return 'Images generated'


if __name__ == '__main__':
    print('Server started listening...')
    app.run()
