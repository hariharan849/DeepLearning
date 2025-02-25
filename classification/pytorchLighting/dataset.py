""" Helper functions to create dataset from image folders and apply transforms"""
import cv2, pathlib, os
import pandas as pd
from PIL import Image

import splitfolders

import torch
import pytorch_lightning as L
from torchvision import datasets, transforms
from torchvision.transforms._presets import ImageClassification
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_dataframe_from_image_paths(image_folder_path: str) -> pd.DataFrame:
    """ Get images, labels and converts to dataframe

        Args:
            image_folder_path(str): Folders containing images

        Returns:
            pd.DataFrame: DataFrame containing image paths and labels
    """
    images = list(pathlib.Path(image_folder_path).glob("*/*"))
    labels = [os.path.split(os.path.dirname(img_path))[-1] for img_path in images]
    dataframe = pd.DataFrame(zip(images, labels), columns=["image_path", "labels"])
    dataframe["image_path"] = dataframe["image_path"].astype('str')
    return dataframe

def is_valid_image(image_path):
    """
    Validate if an image file is not empty or corrupted.

    Args:
        image_path (str): Path to the image file

    Returns:
        bool: True if the image is valid, False otherwise
    """
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        return False  # File doesn't exist or is not a valid file
    
    try:
        # Try to open with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return False  # OpenCV couldn't read the image

        # Optionally, validate with PIL
        with Image.open(image_path) as img:
            img.verify()  # Check for image integrity
        return True  # Image is valid
    except (IOError, cv2.error):
        return False  # Corrupted or unreadable image
    
def get_albumentation_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """ Returns train and test transforms

    Args:
        mean (tuple): Mean values for normalization
        std (tuple): Standard deviation values for normalization

    Returns:
        tuple: Train and validation transforms
    """
    train_transforms = A.Compose(
        transforms=[
            A.Resize(height=224, width=224),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=128, width=128),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        transforms=[
            A.Resize(height=224, width=224),
            A.CenterCrop(height=128, width=128),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return train_transforms, val_transform

def get_torch_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """ Returns train and test transforms

    Args:
        mean (tuple): Mean values for normalization
        std (tuple): Standard deviation values for normalization

    Returns:
        tuple: Train and validation transforms
    """
    train_transform = transforms.Compose([
        # Resize the images to 224
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, test_transform

class CustomDataset(Dataset):
    """ Custom Dataset to apply transforms on dataframe

    Args:
        dataframe(pd.Dataframe): Dataset
        transforms(list): List of transforms to apply
        class_to_idx (dict): Mapping from class names to indices
    """
    def __init__(self, dataframe, transforms, class_to_idx):
        super().__init__()
        self._dataframe = dataframe
        self._transforms = transforms
        self._class_to_idx = class_to_idx

    def __len__(self):
        """ Returns length of dataframe

        Returns:
            int: Number of samples in the dataset
        """
        return len(self._dataframe)

    def __getitem__(self, index):
        """ Returns image and labels from specified index

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            tuple: Transformed image and corresponding label index
        """
        image_path, label = self._dataframe.iloc[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)
        if isinstance(self._transforms, transforms.transforms.Compose) or isinstance(self._transforms, ImageClassification):
            transformed_image = self._transforms(image)
        else:
            transformed_image = self._transforms(image=image)["image"]
        return transformed_image, self._class_to_idx[label]
  
def calculate_mean_std_from_dataset(train_df):
    """ Calculate mean and standard deviation from dataset

    Args:
        train_df (pd.DataFrame): DataFrame containing training data

    Returns:
        tuple: Mean and standard deviation of the dataset
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # Converts images to [C, H, W] format with values in [0, 1]
    ])
    # Load the dataset
    class_to_idx = {cls_name: i for i, cls_name in enumerate(train_df["labels"].unique())}
    train_dataset = CustomDataset(train_df, transform, class_to_idx)

    loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Initialize variables to compute mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    # Iterate through the dataset
    for images, _ in loader:  # Only images are needed
        batch_samples = images.size(0)  # Current batch size
        images = images.view(batch_samples, 3, -1)  # Flatten H and W dimensions
        mean += images.mean(dim=2).sum(dim=0)  # Sum mean of each channel
        std += images.std(dim=2).sum(dim=0)    # Sum std of each channel
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples

    print(f"Mean: {mean}, Std: {std}")
    return mean, std

class CustomDataModule(L.LightningDataModule):
    """ Custom Data Module for PyTorch Lightning

    Args:
        image_dataset_folder (str): Path to the image dataset folder
        batch_size (int): Batch size for data loaders
        output_folder (str): Output folder for split data
        validation_split (float): Fraction of data to use for validation
        transforms (callable): Transformations to apply to the data
        albumentation_transforms (bool): Whether to use Albumentations for transformations
    """
    def __init__(self, image_dataset_folder, batch_size=32, output_folder="", validation_split=0.2, transforms=None, albumentation_transforms=False):
        super().__init__()
        self.image_dataset_folder = image_dataset_folder
        self.batch_size = batch_size
        self.output_folder = output_folder
        self.validation_split = validation_split
        self.transforms = transforms
        self.albumentation_transforms = albumentation_transforms

    def prepare_data(self):
        """ Prepare data by splitting into train, validation, and test sets
        """
        directories = os.listdir(self.image_dataset_folder)
        if "train" in directories:
            self.train_dir = os.path.join(self.image_dataset_folder, "train")
            self.val_dir = self.test_dir = None
            if "test" in directories:
                self.test_dir = os.path.join(self.image_dataset_folder, "test")
            if "validation" in directories:
                self.val_dir = os.path.join(self.image_dataset_folder, "validation")
        else:
            train_split = 1.0 - self.validation_split - 0.1
            self.output_folder = os.path.join(self.image_dataset_folder, "split_folder")

            splitfolders.ratio(
                self.image_dataset_folder, output=self.output_folder, seed=1337, ratio=(train_split, self.validation_split, 0.1)
            )
            self.train_dir = os.path.join(self.output_folder, "train")
            self.test_dir = os.path.join(self.output_folder, "test")
            self.val_dir = os.path.join(self.output_folder, "val")

    def setup(self, stage=None):
        """ Setup datasets and transforms

        Args:
            stage (str, optional): Stage of the setup (train, val, test)
        """
        self.train_df = get_dataframe_from_image_paths(self.train_dir)
        self.train_df['is_valid'] = self.train_df['image_path'].apply(is_valid_image)
        self.train_df = self.train_df[self.train_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)

        if self.val_dir:
            self.val_df = get_dataframe_from_image_paths(self.val_dir)
            self.val_df['is_valid'] = self.val_df['image_path'].apply(is_valid_image)
            self.val_df = self.val_df[self.val_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)
        else:
            self.val_df = None

        if self.test_dir:
            self.test_df = get_dataframe_from_image_paths(self.test_dir)
            self.test_df['is_valid'] = self.test_df['image_path'].apply(is_valid_image)
            self.test_df = self.test_df[self.test_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)
        else:
            self.test_df = None

        mean, std = calculate_mean_std_from_dataset(self.train_df)

        if self.albumentation_transforms:
            self.train_transforms, self.val_transforms = get_albumentation_transforms(mean, std)
        elif self.transforms:
            _, self.val_transforms = get_torch_transforms(mean, std)
            self.train_transforms = self.transforms
        else:
            self.train_transforms, self.val_transforms = get_torch_transforms(mean, std)

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.train_df["labels"].unique())}
        self.train_dataset = CustomDataset(self.train_df, self.train_transforms, self.class_to_idx)
        self.val_dataset = CustomDataset(self.val_df, self.val_transforms, self.class_to_idx) if self.val_df is not None else None
        self.test_dataset = CustomDataset(self.test_df, self.val_transforms, self.class_to_idx) if self.test_df is not None else None

    def train_dataloader(self):
        """ Returns the training data loader

        Returns:
            DataLoader: DataLoader for training data
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """ Returns the validation data loader

        Returns:
            DataLoader: DataLoader for validation data
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False) if self.val_dataset is not None else None

    def test_dataloader(self):
        """ Returns the test data loader

        Returns:
            DataLoader: DataLoader for test data
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False) if self.test_dataset is not None else None
