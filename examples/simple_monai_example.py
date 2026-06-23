# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    ToTensor
)
from monai.data import DataLoader, Dataset

# Define a simple transform to load and preprocess the image
transform = [
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    ToTensor()
]

# Define a simple dataset
data_dir = "/monai_sample_images"
images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
print("Images found:", images)
dataset = Dataset(data=images, transform=transform)
print("Number of samples in dataset:", len(dataset))

# Create a data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load and visualize a sample image
for batch_data in data_loader:
    image = batch_data[0][0]
    print("Image shape:", image.shape)
    print("Image data type:", image.dtype)
    print("Image max value:", torch.max(image))
    print("Image min value:", torch.min(image))
    break
