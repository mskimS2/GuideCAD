import os
import torch
import random
import numpy as np
from glob import glob
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    model_name = "openai/clip-vit-base-patch32"
    save_base_path = "data/clip/"

    device = "cuda:0"
    print(model_name, save_base_path)

    folder_list = glob("data/cad_image/*")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    for folder_path in tqdm(sorted(folder_list), desc="Processing Folders", ncols=120):
        subfolder_list = glob(folder_path + "/*")

        for subfolder_path in tqdm(subfolder_list, desc="Processing Subfolders", leave=False):
            id_ = subfolder_path.split("/")[-1]
            subfolder = subfolder_path.split("/")[-2]

            file_list = glob(subfolder_path + "/*")
            if not file_list:
                print(f"No files found in {subfolder_path}. Skipping...")
                continue

            images = []
            for img_path in sorted(file_list):
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                except (UnidentifiedImageError, IOError) as e:
                    print(f"Error loading image {img_path}: {e}. Skipping...")
                    continue

            if not images:
                print(f"No valid images in {subfolder_path}. Skipping...")
                continue

            if len(images) != 7:
                print(f"{subfolder_path} No valid images length is not 7.")
                continue

            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_embedding = model.get_image_features(**inputs)

            save_path = os.path.join(save_base_path, subfolder)
            os.makedirs(save_path, exist_ok=True)
            save_file_path = os.path.join(save_path, f"{id_}.npy")
            np.save(save_file_path, image_embedding.cpu().numpy())
            print(f"Saved: {save_file_path}, image_embedding=", image_embedding.shape)
