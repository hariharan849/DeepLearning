# Brain Tumor Classification

This project aims to classify brain tumor images using deep learning models. The dataset used is the "Brain Tumor Multimodal Image CT and MRI" dataset from Kaggle.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Seaborn
- Pandas
- NumPy
- KaggleHub
- SplitFolders

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/DeepLearning.git
    cd DeepLearning/classification/tensorflow
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset is automatically downloaded from Kaggle using the KaggleHub library. Ensure you have your Kaggle API credentials set up.

## Usage

To train, evaluate, and visualize the model, run the `main.py` script:

```bash
python main.py
```

## Script Overview

### main.py

- Downloads the dataset from Kaggle.
- Loads the dataset using the `DatasetLoader` class.
- Visualizes the dataset using the `Visualize` class.
- Builds the model using the `ClassificationModel` class.
- Creates callbacks using the `Callbacks` class.
- Trains, evaluates, and predicts using the model.

### dataset.py

Contains the `DatasetLoader` class which provides methods to load and preprocess datasets.

### visualize.py

Contains the `Visualize` class which provides methods to visualize datasets.

### model.py

Contains the `ClassificationModel` class which provides methods to build, train, and evaluate deep learning models.

### callbacks.py

Contains the `Callbacks` class which provides methods to create various Keras callbacks for training.

## Results

The results of the training, including accuracy and loss plots, will be saved in the specified output directory.

## License

This project is licensed under the MIT License.
