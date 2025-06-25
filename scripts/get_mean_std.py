import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Path to image directory (adjust this)
image_dir = Path('/path/to/madmax/images')

# Image resize size
resize_size = (192, 640)

# Transform: resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor()
])

# Accumulators
mean = torch.zeros(3)
std = torch.zeros(3)
num_pixels = 0

# Loop over images
image_paths = list(image_dir.glob('*.png'))  # or *.jpg depending
for img_path in tqdm(image_paths):
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img)
    num_pixels += tensor.shape[1] * tensor.shape[2]
    mean += tensor.sum((1, 2))
    std += (tensor ** 2).sum((1, 2))

# Final mean and std
mean /= num_pixels
std = (std / num_pixels - mean ** 2) ** 0.5

print(f'Mean: {mean.tolist()}')
print(f'Std:  {std.tolist()}')
