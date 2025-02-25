# -*- coding: utf-8 -*-
"""
Dataset

This module contains the DatasetLoader class which provides methods to load and preprocess datasets.
"""

import os, pathlib
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import splitfolders

from collections import namedtuple
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator


DATA_HOLDER = namedtuple("DataHolder", ["dataset", "dataloader"])

class DatasetLoader:
    """ Dataset Loader class to load dataset from image folder or tensorflow datasets """
    def __init__(self, image_dataset_folder="", num_classes=10, img_height=224, img_width=224, batch_size=32, validation_split=0.2, image_preprocess=None, label_preprocess=tf.one_hot):
        """ Initialize the DatasetLoader class

            Args:
                image_dataset_folder(str): Folder containing images for training and validation
                num_classes(int): Number of classes
                img_height(int): Image height
                img_width(int): Image width
                batch_size(int): Batch size
                validation_split(float): Validation split ratio
                image_preprocess(function): Image preprocess function
                label_preprocess(function): Label preprocess function
        """
        self._image_dataset_folder = image_dataset_folder
        self._img_height = img_height
        self._img_width = img_width
        self._batch_size = batch_size
        self._validation_split = validation_split
        self._image_preprocess = image_preprocess
        self._label_preprocess = label_preprocess
        self._num_classes = num_classes

    def _input_preprocess_train(self, image, label):
        """ Preprocess the input image and label 

            Args:
                image(tf.Tensor): Input image
                label(tf.Tensor): Input label

            Returns:
                tuple: Preprocessed image and label
        """
        if self._image_preprocess:
            image = self._image_preprocess(image)
        else:
            image = tf.cast(image, tf.float32) / 255.0  # Normalize image to [0, 1]

        label = self._label_preprocess(label, self._num_classes)

        return image, label

    def _dataset_to_dataframe(self, dataset, split):
        """ Convert a dataset split into a DataFrame

            Args:
                dataset(tf.data.Dataset): Dataset split
                split(str): Split name (train, val, test)

            Returns:
                pd.DataFrame: DataFrame containing image data
        """
        data = []
        for image, label in tfds.as_numpy(dataset):
            data.append({
                'image_path': image,
                'labels': int(label),
                'split': split
            })
        
        return pd.DataFrame(data)

    def create_dataframe_from_image_folder(self):
        """ Create a DataFrame from image folder

            Returns:
                df(pd.DataFrame): DataFrame containing image path, split, and class name
        """
        # List to store data
        data = []

        # Walk through train, validation, and test folders
        for split in ['train', 'val', 'test']:
            split_folder = os.path.join(self._image_dataset_folder, split)
            for class_name in os.listdir(split_folder):
                class_folder = os.path.join(split_folder, class_name)
                if os.path.isdir(class_folder):
                    for image_name in os.listdir(class_folder):
                        image_path = os.path.join(class_folder, image_name)
                        if os.path.isfile(image_path):
                            # Append data as (image_path, split, class_name)
                            data.append({'image_path': image_path,
                                         'split': split,
                                         'labels': class_name})
        
        # Create a DataFrame
        df = pd.DataFrame(data)
        return df

    def _input_preprocess_test(self, image, label):
        """ Preprocess the input image and label

            Args:
                image(tf.Tensor): Input image
                label(tf.Tensor): Input label

            Returns:
                tuple: Preprocessed image and label
        """
        label = self._label_preprocess(label, self._num_classes)
        return image, label

    def _get_image_dataset_from_jpg(self, imageFolderPath, shuffle=True, label_mode="categorical"):
        """ Returns dataset

            Args:
                image_folder_path(str) : Folder containing images for training and validation
                shuffle(bool): Whether to shuffle the dataset
                label_mode(str): Label mode (categorical or binary)

            Returns:
                DATA_HOLDER: Dataset and dataloader
        """
        ds = tf.keras.image_dataset_from_directory(
            imageFolderPath,
            labels='inferred',
            label_mode=label_mode,
            class_names=None,
            color_mode='rgb',
            shuffle=shuffle,
            seed=123,
            image_size=(self._img_height, self._img_width),
            batch_size=self._batch_size
        )

        AUTOTUNE = tf.data.AUTOTUNE
        loader = ds.cache().prefetch(buffer_size=AUTOTUNE)
        dataset = DATA_HOLDER(dataset=ds, dataloader=loader)
        return dataset

    @staticmethod
    def _get_img_extension(imageFolderPath):
        """ Returns list of image extensions from image paths folder

            Args:
                image_folder_path(str): Image path directory

            Returns:
                list: List of image extensions
        """
        images = list(pathlib.Path(imageFolderPath).glob("*/*"))
        return [os.path.splitext(img_path)[-1] for img_path in images]

    def _get_image_dataset_from_multiple_ext(self, train_dir, val_dir, test_dir, label_mode='categorical'):
        """ Creates train and validation dataset from image folder hierarchy
        
            Args:
                train_dir(str): Training directory
                val_dir(str): Validation directory
                test_dir(str): Test directory
                label_mode(str): Label mode (categorical or binary)

            Returns:
                tuple: Train, validation, and test generators
        """
        train_datagen = ImageDataGenerator(
            rescale=1/255.0,
            horizontal_flip=True,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode="nearest"
        )

        valid_datagen = ImageDataGenerator(rescale=1/255.0)

        train_gen = train_datagen.flow_from_directory(
            train_dir,
            seed=45,
            batch_size=self._batch_size,
            target_size=(self._img_height, self._img_width),
            color_mode="rgb",
            class_mode=label_mode,
            shuffle=True
        )
        val_gen = valid_datagen.flow_from_directory(
            val_dir,
            seed=45,
            batch_size=self._batch_size,
            target_size=(self._img_height, self._img_width),
            color_mode="rgb",
            class_mode=label_mode,
            shuffle=False
        )
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self._img_height, self._img_width),
            batch_size=32,
            class_mode=label_mode,
            shuffle=False
        )

        return train_gen, val_gen, test_gen

    def load_dataset(self):
        """ Load dataset from image folder or tensorflow datasets

            Returns:
                tuple: Train, validation, and test datasets along with class names
        """
        # Load dataset from image folder
        directories = os.listdir(self._image_dataset_folder)
        # Check if the dataset is already split into train, validation and test
        if "train" in directories:
            train_dir = os.path.join(self._image_dataset_folder, "train")
            val_dir = test_dir = None
            if "test" in directories:
                test_dir = os.path.join(self._image_dataset_folder, "test")
            if "validation" in directories:
                val_dir = os.path.join(self._image_dataset_folder, "validation")
            if "val" in directories:
                val_dir = os.path.join(self._image_dataset_folder, "val")
        else:
            # Split the dataset into train, validation and test
            train_split = 1.0-self._validation_split-0.1
            output_folder = os.path.join(os.path.dirname(self._image_dataset_folder), 'split_folder')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            splitfolders.ratio(
                self._image_dataset_folder, output=output_folder, seed=1337, ratio = (train_split, self._validation_split, 0.1)
            )

            train_dir = os.path.join(output_folder, "train")
            test_dir = os.path.join(output_folder, "test")
            val_dir = os.path.join(output_folder, "val")
            self._image_dataset_folder = output_folder
            print(output_folder)

        label_mode = "categorical"
        if len(os.listdir(train_dir)) == 2:
            label_mode = "binary"
        # Check if the images are in jpg format
        ext = DatasetLoader._get_img_extension(train_dir)
        class_names = os.listdir(train_dir)
        if len(ext) == 1 and ext[0] == "jpg":
            ds_train = self._get_image_dataset_from_jpg(train_dir, label_mode=label_mode)
            ds_val = self._get_image_dataset_from_jpg(val_dir, shuffle=False, label_mode=label_mode)
            ds_test = self._get_image_dataset_from_jpg(test_dir, shuffle=False, label_mode=label_mode)
            return ds_train, ds_val, ds_test, class_names
        else:
            ds_train, ds_val, ds_test = self._get_image_dataset_from_multiple_ext(train_dir, val_dir, test_dir, label_mode=label_mode)
            return ds_train, ds_val, ds_test, class_names