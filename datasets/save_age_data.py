import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torch

# Folder containing images
data = pd.read_csv("datasets/train.csv")
transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])
image_folder = "datasets/pet_images"
images = []
ages = []
for index, row in tqdm(data.iterrows()):
    photo_id = row['Id']
    age = row['Pawpularity']

    image_path = os.path.join(image_folder, f'{photo_id}.jpg')      
    # Load the image
    if not os.path.exists(image_path):
        continue
    

    # Read and preprocess the imag
    img = Image.open(image_path)
    images.append(transform(img))
    ages.append(age)
X = torch.stack(images).detach().numpy()
y = np.array(ages)

with open("datasets/cuteness.pkl", "wb") as f:
    pickle.dump((X, y), f)