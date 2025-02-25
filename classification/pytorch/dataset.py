"""Helper functions to create dataset from image folders and apply transforms."""
import cv2, pathlib, os
import pandas as pd
from PIL import Image

import splitfolders
import shutil

import torch
from torchvision import datasets, transforms
from torchvision.transforms._presets import ImageClassification
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_dataframe_from_image_paths(image_folder_path: str) -> pd.DataFrame:
    """Get images, labels and converts to dataframe.

    Args:
        image_folder_path (str): Folders containing images.
    
    Returns:
        pd.DataFrame: DataFrame containing image paths and labels.
    """
    images = list(pathlib.Path(image_folder_path).glob("*/*"))
    labels = [os.path.split(os.path.dirname(img_path))[-1] for img_path in images]
    dataframe = pd.DataFrame(zip(images, labels), columns=["image_path", "labels"])
    dataframe["image_path"] = dataframe["image_path"].astype('str')
    return dataframe

def is_valid_image(image_path):
    """Validate if an image file is not empty or corrupted.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        bool: True if the image is valid, False otherwise.
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
    """Returns train and test transforms.
    
    Args:
        mean (tuple): Mean for normalization.
        std (tuple): Standard deviation for normalization.
    
    Returns:
        tuple: Train and validation transforms.
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

def get_torch_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), img_size=224):
    """Returns train and test transforms.
    
    Args:
        mean (tuple): Mean for normalization.
        std (tuple): Standard deviation for normalization.
    
    Returns:
        tuple: Train and validation transforms.
    """
    train_transform = transforms.Compose([
        # Resize the images to 224
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    return train_transform, test_transform

class CustomDataset(Dataset):
    """Custom Dataset to apply transforms on dataframe.

    Args:
        dataframe (pd.DataFrame): Dataset.
        transforms (list): List of transforms to apply.
        class_to_idx (dict): Mapping from class names to indices.
    """
    def __init__(self, dataframe, transforms, class_to_idx):
        super().__init__()
        self._dataframe = dataframe
        self._transforms = transforms
        self._class_to_idx = class_to_idx

    def __len__(self):
        """Returns length of dataframe."""
        return len(self._dataframe)

    def __getitem__(self, index):
        """Returns image and labels from specified index.
        
        Args:
            index (int): Index of the item.
        
        Returns:
            tuple: Transformed image and label.
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
  
def calculate_mean_std_from_dataset(train_df, batch_size=64):
    """Calculates mean and standard deviation from the dataset.
    
    Args:
        train_df (pd.DataFrame): Training dataframe.
        batch_size (int): Batch size for the DataLoader.
    
    Returns:
        tuple: Mean and standard deviation.

    The function performs the following steps:
     1. Defines a transform to resize images to 224x224 and convert them to tensors.
     2. Creates a CustomDataset instance using the provided dataframe and transform.
     3. Loads the dataset into a DataLoader with the specified batch size.
     4. Initializes variables to accumulate the mean and standard deviation of the dataset.
     5. Iterates through the DataLoader, computing the mean and standard deviation for each batch.
     6. Sums the mean and standard deviation of each batch and keeps track of the total number of samples.
     7. Divides the accumulated mean and standard deviation by the total number of samples to get the final values.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # Converts images to [C, H, W] format with values in [0, 1]
    ])
    # Load the dataset
    class_to_idx = {cls_name: i for i, cls_name in enumerate(train_df["labels"].unique())}
    train_dataset = CustomDataset(train_df, transform, class_to_idx)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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

def get_loaders(image_dataset_folder, batch_size=32, output_folder="", validation_split=0.2, img_size=224, transforms=None, albumentation_transforms=False):
    """Returns Torch dataloader for train, val and test.
    
    Args:
        image_dataset_folder (str): Path to the image dataset folder.
        batch_size (int): Batch size for the dataloaders.
        output_folder (str): Output folder for split data.
        validation_split (float): Ratio for validation split.
        transforms (list): List of transforms to apply.
        albumentation_transforms (bool): Whether to use Albumentations transforms.
    
    Returns:
        tuple: Train, validation, and test dataloaders, and transforms.
    """
    parent_directories = os.listdir(image_dataset_folder)
    output_folder = os.path.join(image_dataset_folder, "split_folder")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Use shutil.rmtree to remove non-empty directory
    test_dir = val_dir = None
    if "train" in parent_directories:
        train_dir = os.path.join(image_dataset_folder, "train")
        val_dir = test_dir = None
        if "test" in parent_directories:
            test_dir = os.path.join(image_dataset_folder, "test")
        if "validation" in parent_directories:
            val_dir = os.path.join(image_dataset_folder, "validation")
    else:
        train_split = 1.0-validation_split-0.1
        output_folder = os.path.join(image_dataset_folder, "split_folder")

        splitfolders.ratio(
            image_dataset_folder, output=output_folder, seed=1337, ratio = (train_split, validation_split, 0.1)
        )
        train_dir = os.path.join(output_folder, "train")
        test_dir = os.path.join(output_folder, "test")
        val_dir = os.path.join(output_folder, "val")

    val_df = test_df = None
    train_df = get_dataframe_from_image_paths(train_dir)
    print(f"Original entries: {len(train_df)}")
    # Example: Assuming your DataFrame has a column 'image_path'
    train_df['is_valid'] = train_df['image_path'].apply(is_valid_image)

    # Keep only valid images
    train_df = train_df[train_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)

    
    print(f"Valid entries: {len(train_df)}")

    if val_dir:
        val_df = get_dataframe_from_image_paths(val_dir)
        print(f"Original entries: {len(val_df)}")
        # Example: Assuming your DataFrame has a column 'image_path'
        val_df['is_valid'] = val_df['image_path'].apply(is_valid_image)

        # Keep only valid images
        val_df = val_df[val_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)

        print(f"Valid entries: {len(val_df)}")
    if test_dir:
        test_df = get_dataframe_from_image_paths(test_dir)
        print(f"Original entries: {len(test_df)}")
        # Example: Assuming your DataFrame has a column 'image_path'
        test_df['is_valid'] = test_df['image_path'].apply(is_valid_image)

        # Keep only valid images
        test_df = test_df[test_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)

        print(f"Valid entries: {len(test_df)}")

    mean, std = calculate_mean_std_from_dataset(train_df)
    
    if albumentation_transforms:
        train_transforms, val_transforms = get_albumentation_transforms(mean, std)
    elif transforms:
        _, val_transforms = get_torch_transforms(mean, std, img_size)
        train_transforms = transforms
    else:
        train_transforms, val_transforms = get_torch_transforms(mean, std, img_size)

    class_to_idx = {cls_name: i for i, cls_name in enumerate(train_df["labels"].unique())}
    train_dataset = CustomDataset(train_df, train_transforms, class_to_idx)
    val_dataset = test_dataset = None
    if val_df is not None:
        val_dataset = CustomDataset(val_df, val_transforms, class_to_idx)
    if test_df is not None:
        test_dataset = CustomDataset(test_df, val_transforms, class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = test_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_transforms, val_transforms

