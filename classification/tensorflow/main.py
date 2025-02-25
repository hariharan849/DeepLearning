# -*- coding: utf-8 -*-
"""
Main

This module is the entry point for training, evaluating, and visualizing the deep learning model.
"""

"""Untitled250.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jb_Rgl8fGHZPMxNbOmgy_l6cIH8i1pmz
"""

import kagglehub
murtozalikhon_brain_tumor_multimodal_image_ct_and_mri_path = kagglehub.dataset_download('murtozalikhon/brain-tumor-multimodal-image-ct-and-mri')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(murtozalikhon_brain_tumor_multimodal_image_ct_and_mri_path ):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# imports
import os

from dataset import DatasetLoader
from visualize import Visualize
from callbacks import Callbacks
from classification.tensorflow.models.model import (
    ClassificationModel, train_model, AlexnetFromScratch, ResnetFromScratch,
    DenseNetFromScratch, InceptionModelFromScratch
)

#loading dataset
dataset_folder = os.path.join(murtozalikhon_brain_tumor_multimodal_image_ct_and_mri_path, 'Dataset', 'Brain Tumor CT scan Images')
dataset = DatasetLoader(dataset_folder, img_height=224, img_width=224)
ds_train, ds_val, ds_test, class_names = dataset.load_dataset()
dataframe = dataset.create_dataframe_from_image_folder()

# dataset visualizationn
# vis = Visualize.from_dataframe(dataframe)
# vis.plot_random_images_per_class()
# vis.plot_class_distribution_in_pie_chart()
# vis.plot_class_distribution_in_count_chart()
# vis.plot_percentage_split()

#model creation
# model = AlexnetFromScratch(ds_train.num_classes, input_shape=(227, 227, 3)).build()
# model = ClassificationModel.build("vgg16", 1 if ds_train.num_classes == 2 else ds_train.num_classes, input_shape=(224, 224, 3))
model = DenseNetFromScratch(ds_train.num_classes).build()
model.summary()

#callback creation
callbacks_path = os.path.join("/content/drive/MyDrive", 'callbacks')
if not os.path.exists(callbacks_path):
    os.makedirs(callbacks_path)
callbacks = Callbacks(callbacks_path).get_callbacks()

#model training, evaluation and predict
train_model(model, ds_train, ds_val, ds_test, callbacks, epochs=1, model_name="")
