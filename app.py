# background_changer_app.py

import os
import shutil
import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
from stylegan2_pytorch import ModelLoader
from google.colab import files
import matplotlib.pyplot as plt

# Ensure the MODNet model is available
if not os.path.exists('MODNet'):
    os.system('git clone https://github.com/ZHKKKe/MODNet')

pretrained_ckpt = 'pretrained/modnet_photographic_portrait_matting.ckpt'
if not os.path.exists(pretrained_ckpt):
    os.system('gdown --id 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz -O pretrained/modnet_photographic_portrait_matting.ckpt')

# Function to load MODNet model for image matting
def load_modnet():
    from demo.image_matting.colab import inference
    model_path = 'pretrained/modnet_photographic_portrait_matting.ckpt'
    return inference.load_model(model_path)

# Function to perform image matting
def perform_image_matting(input_path, output_path):
    os.system(f'python -m demo.image_matting.colab.inference --input-path {input_path} --output-path {output_path} --ckpt-path {pretrained_ckpt}')

# Function to generate a background using StyleGAN2
def generate_background(model, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(1, 512).cuda()
    styles = model.noise_to_styles(noise, trunc_psi=0.7)
    images = model.styles_to_images(styles)
    generated_image = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
    generated_image = generated_image.cpu().numpy().transpose(0, 2, 3, 1)[0]
    return (generated_image * 255).astype(np.uint8)

# Function to replace the background
def replace_background(foreground, alpha_matte, generated_background):
    foreground = np.array(foreground)
    alpha_matte = np.array(alpha_matte) / 255.0
    generated_background = np.array(generated_background)

    generated_background = cv2.resize(generated_background, (foreground.shape[1], foreground.shape[0]))

    alpha = np.dstack([alpha_matte] * 3)
    foreground = cv2.multiply(alpha, foreground.astype(float) / 255.0)
    background = cv2.multiply(1.0 - alpha, generated_background.astype(float) / 255.0)
    result = cv2.add(foreground, background) * 255.0
    return result.astype(np.uint8)

# Load StyleGAN2 model
model = ModelLoader(name='ffhq', base_dir='/content/')

# Streamlit interface
def main():
    st.title("Background Changer with StyleGAN2")
    
    st.sidebar.title("Upload Images")
    uploaded_files = st.sidebar.file_uploader("Choose image files", accept_multiple_files=True, type=["jpg", "png"])

    if uploaded_files:
        st.sidebar.write("Selected files:")
        for uploaded_file in uploaded_files:
            st.sidebar.write(uploaded_file.name)

        if st.sidebar.button('Process Images'):
            input_folder = 'demo/image_matting/colab/input'
            output_folder = 'demo/image_matting/colab/output'

            if os.path.exists(input_folder):
                shutil.rmtree(input_folder)
            os.makedirs(input_folder)

            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder)

            for uploaded_file in uploaded_files:
                file_path = os.path.join(input_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Perform image matting
            perform_image_matting(input_folder, output_folder)

            st.write("Image matting complete. Generating new backgrounds...")

            foreground_images = []
            for uploaded_file in uploaded_files:
                image_path = os.path.join(input_folder, uploaded_file.name)
                matte_path = os.path.join(output_folder, uploaded_file.name.split('.')[0] + '.png')
                foreground_images.append((Image.open(image_path).convert('RGB'), Image.open(matte_path).convert('L')))

            background = generate_background(model)

            for i, (foreground, alpha_matte) in enumerate(foreground_images):
                result_image = replace_background(foreground, alpha_matte, background)
                result_path = f'result_{i}.png'
                cv2.imwrite(result_path, result_image)

                st.image(result_path, caption=f"Processed Image {i + 1}", use_column_width=True)
                st.write(f"Image saved at: {result_path}")

if __name__ == "__main__":
    main()
