"""
Main script for training and evaluating the AlexNet model using PyTorch Lightning.
It includes dataset loading, model creation, training, and evaluation.
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
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import matplotlib.pyplot as plt

from dataset import CustomDataModule
from visualize import Visualize
from model import AlexnetSotA, ResnetFromScratch
from callbacks import Callbacks
from utils import plot_loss_curves, pred_and_plot_image

# loading dataset
dataset_folder = os.path.join(murtozalikhon_brain_tumor_multimodal_image_ct_and_mri_path, 'Dataset', 'Brain Tumor CT scan Images')
class_names = os.listdir(dataset_folder)
output_folder = os.path.join(dataset_folder, "split_folder")

# model creation
model = ResnetFromScratch(class_names)
data_module = CustomDataModule(dataset_folder, output_folder=output_folder, transforms=None)

# dataset visualization
# vis = Visualize(output_folder)
# vis.plot_random_images_per_class()
# vis.plot_class_distribution_in_pie_chart()
# vis.plot_class_distribution_in_count_chart()
# vis.plot_percentage_split()

# training
logger = TensorBoardLogger("tb_logs", name="AlexnetSotA")

callbacks = Callbacks().get_callbacks()

trainer = pl.Trainer(
    max_epochs=1,
    logger=logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1 if torch.cuda.is_available() else None,
    callbacks=callbacks
)

# Find the optimal learning rate
tuner = pl.tuner.Tuner(trainer)
lr_finder = tuner.lr_find(model, datamodule=data_module)
new_lr = lr_finder.suggestion()
model.lr = new_lr  # Set the learning rate attribute correctly

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=7, gamma=0.1)
model.scheduler = scheduler

trainer.fit(model, data_module)

# plot loss curves
results = trainer.logged_metrics
plot_loss_curves(results)

# test and plot
test_image_folder = os.path.join(output_folder, "test")
images = list(pathlib.Path(test_image_folder).glob("*/*"))
image_path = random.choice(images)
target_image_act_label = os.path.split(os.path.dirname(image_path))[-1]
pred_and_plot_image(model, class_names, image_path, target_image_act_label, device=model.device, transform=alexnet_model.auto_transforms)
print('Training complete.')