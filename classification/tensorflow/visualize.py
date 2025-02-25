# -*- coding: utf-8 -*-
"""
Visualize

This module contains the Visualize class which provides methods to visualize datasets.
"""

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


class Visualize:
    """ Visualize class to visualize dataset
    """
    def __init__(self, image_dataset_folder=""):
        """ Initialize the Visualize class

            Args:
                image_dataset_folder(str): Folder containing images for training and validation
        """
        self._image_dataset_folder = image_dataset_folder
        if image_dataset_folder:
            self._dataframe = self.create_dataframe_from_image_folder()

    def create_dataframe_from_image_folder(self):
        """ Create a DataFrame from image folder

            Returns:
                df(pd.DataFrame): DataFrame containing image path, split, and class name
        """
        # List to store data
        data = []

        # Walk through train, validation, and test folders
        for split in ['train', 'validation', 'test']:
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

    @classmethod  
    def from_dataframe(cls, dataframe):
        """ Create a Visualize object from a DataFrame

            Args:
                dataframe(pd.DataFrame): DataFrame containing image data

            Returns:
                Visualize: An instance of the Visualize class
        """
        # Create a Visualize object
        visualize = Visualize()
        visualize._dataframe = dataframe
        return visualize

    def plot_class_distribution_in_pie_chart(self):
        """ Plots dataset label distribution in pie chart
        """
        # Assuming your data is in a pandas DataFrame called 'train_df'
        animals_counts = self._dataframe['labels'].value_counts()

        fig, ax = plt.subplots()
        wedges, texts, _ = ax.pie(
            animals_counts.values.astype("float"), startangle=90,
            autopct='%1.1f%%', wedgeprops=dict(width=0.3, edgecolor='black')
        )

        # Add glow effect to each wedge
        for wedge in wedges:
            wedge.set_path_effects([withStroke(linewidth=6, foreground='cyan', alpha=0.4)])

        # Customize chart labels
        plt.legend(self._dataframe.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10, facecolor='#222222')

        # Dark background for the cyberpunk look
        fig.patch.set_facecolor('#2c2c2c')
        ax.set_facecolor('#2c2c2c')

        # Title
        plt.title("Pie Chart", color="white", fontsize=16)

        plt.show()

    def plot_class_distribution_in_count_chart(self):
        """ Plots dataset label distribution in bar chart
        """
        #count Plot

        plt.figure(figsize=(8, 6))
        ax = sns.countplot(self._dataframe, x="labels", palette='pastel')

        # Annotate the count on top of each bar
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom')

        plt.tight_layout()
        # Show the plot
        plt.show()

    def plot_random_images_per_class(self, no_images:int=2):
        """ Plots random images per class

            Args:
                no_images(int): Number of images to plot per class
        """
        class_names = self._dataframe["labels"].unique()
        random_images = self._dataframe.sample(frac=1).sort_values(by="labels").groupby('labels').head(no_images)

        count = 0
        num_classes = len(class_names)

        plt.figure(figsize=(12, num_classes * 4))

        for index, (image_path, _, class_name) in random_images.iterrows():
            if isinstance(image_path, str):
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            else:
                print(image_path.shape)
                image = image_path
            count += 1
            plt.subplot(num_classes, 2, count)
            plt.imshow(image)
            plt.axis('off')
            plt.title(class_name)

    def plot_percentage_split(self):
        """ Plots dataset split distribution in pie chart
        """
        split_distribution = self._dataframe['split'].value_counts()
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    
        def autopct_format(value):
            """Formats the autopct value to display the percentage and count."""
            total = split_distribution.sum()
            percentage = f'{value:.1f}%'
            count = int(value * total / 100)
            return f'{percentage}\n{count}'

        # Create a pie chart
        plt.figure(figsize=(8, 8))
        split_distribution.plot.pie(colors=colors, autopct=autopct_format, startangle=140)
        plt.title('Dataset Split Distribution', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        plt.show()