import cv2, pathlib, os
import pandas as pd
from PIL import Image

try:
    import splitfolders
except:
    ! pip install split-folders
    import splitfolders

import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_dataframe_from_image_paths(image_folder_path: str) -> pd.DataFrame:
    """ Get images, labels and converts to dataframe

        Args:
            image_folder_path(str): Folders containing images
    """
    images = list(pathlib.Path(image_folder_path).glob("*/*"))
    labels = [os.path.split(os.path.dirname(img_path))[-1] for img_path in images]
    dataframe = pd.DataFrame(zip(images, labels), columns=["image_path", "labels"])
    dataframe["image_path"] = dataframe["image_path"].astype('str')
    return dataframe

def is_valid_image(image_path):
    """
    Validate if an image file is not empty or corrupted.
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
  """
  def __init__(self, dataframe, transforms, class_to_idx):
    super().__init__()
    self._dataframe = dataframe
    self._transforms = transforms
    self._class_to_idx = class_to_idx

  def __len__(self):
    """ Returns length of dataframe
    """
    return len(self._dataframe)

  def __getitem__(self, index):
    """ Returns image and labels from specified index
    """
    image_path, label = self._dataframe.iloc[index]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert NumPy array to PIL Image
    image = Image.fromarray(image)
    if isinstance(self._transforms, transforms.transforms.Compose):
      transformed_image = self._transforms(image)
    else:
      transformed_image = self._transforms(image=image)["image"]
    return transformed_image, self._class_to_idx[label]
  
def calculate_mean_std_from_dataset(train_df):
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

def get_loaders(image_dataset_folder, batch_size=32, validation_split=0.2, albumentation_transforms=False):
    """ Returns Torch dataloader for train, val and test
    """
    directories = os.listdir(image_dataset_folder)
    test_dir = val_dir = None
    if "train" in directories:
      train_dir = os.path.join(image_dataset_folder, "train")
      val_dir = test_dir = None
      if "test" in directories:
        test_dir = os.path.join(image_dataset_folder, "test")
      if "validation" in directories:
        val_dir = os.path.join(image_dataset_folder, "validation")
    else:
      train_split = 1.0-validation_split-0.1
      output_folder = os.path.dirname(image_dataset_folder)

      train_dir, val_dir, test_dir = splitfolders.ratio(
          image_dataset_folder, output=output_folder, seed=1337, ratio = (train_split, validation_split, 0.1)
      )

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
    
    if not albumentation_transforms:
      train_transforms, val_transforms = get_torch_transforms(mean, std)
    else:
      train_transforms, val_transforms = get_albumentation_transforms(mean, std)

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
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if test_dataset is not None:
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

