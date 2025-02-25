"""
This module performs the following tasks:
1. Downloads the brain tumor multimodal image dataset from Kaggle.
2. Loads necessary libraries and packages.
3. Loads the dataset and splits it into training, validation, and test sets.
4. Visualizes the dataset.
5. Creates and trains an AlexNet model.
6. Plots the training loss curves.
7. Makes predictions on a random test image and visualizes the results.
"""

import kagglehub
murtozalikhon_brain_tumor_multimodal_image_ct_and_mri_path = kagglehub.dataset_download('murtozalikhon/brain-tumor-multimodal-image-ct-and-mri')

print('Data source import complete.')


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pathlib, random
import torch

from dataset import get_loaders
from visualize import Visualize
from model import (
    AlexnetSotA, AlexnetFromScratch, AlexnetFromScratchV2, ResnetFromScratchV1,
    ResnetFromScratchV2, InceptionModelFromScratch, DenseNetFromScratch)
from utils import create_writer, plot_loss_curves, pred_and_plot_image
from trainer import Trainer

# loading dataset
dataset_folder = os.path.join(murtozalikhon_brain_tumor_multimodal_image_ct_and_mri_path, 'Dataset', 'Brain Tumor CT scan Images')
output_folder = os.path.join(dataset_folder, "split_folder")

# train_loader, val_loader, test_loader, train_transforms, val_transforms = get_loaders(dataset_folder, output_folder=output_folder, transforms=auto_transforms)
train_loader, val_loader, test_loader, train_transforms, val_transforms = get_loaders(dataset_folder, output_folder=output_folder)
class_names = os.listdir(os.path.join(output_folder, "train"))
# model creation
# model, auto_transforms = AlexnetSotA(class_names).build()

# model = AlexnetFromScratchV2(1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

model = DenseNetFromScratch(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# dataset visualization
# vis = Visualize(output_folder)
# vis.plot_random_images_per_class()
# vis.plot_class_distribution_in_pie_chart()
# vis.plot_class_distribution_in_count_chart()
# vis.plot_percentage_split()

if len(class_names) == 2:
    loss_fn = torch.nn.BCEWithLogitsLoss()
else:
    loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

writer = create_writer(
    experiment_name="Animal dataset",
    model_name="Alexnet",
    extra="learning animals dataset from torch api")

trainer = Trainer(model, train_loader, test_loader, optimizer, loss_fn, 10, device, writer)
results = trainer.train()

plot_loss_curves(results)
test_image_folder = os.path.join(output_folder, "test")
images = list(pathlib.Path(test_image_folder).glob("*/*"))
image_path = random.choice(images)
target_image_act_label = os.path.split(os.path.dirname(image_path))[-1]
pred_and_plot_image(model, class_names, image_path, target_image_act_label, device=device, transform=val_transforms)
print('Training complete.')