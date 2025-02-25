# Deep Learning Classification with PyTorch

This repository contains a deep learning classification project using PyTorch.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

## Installation

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage

To train and evaluate the model, run the following command:
```bash
python main.py --epochs <num_epochs> --batch-size <batch_size> --learning-rate <learning_rate>
```

## Arguments

The `main.py` script accepts the following arguments:

- `--epochs`: Number of epochs to train the model (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate for the optimizer (default: 0.001)

## Example

```bash
python main.py --epochs 20 --batch-size 64 --learning-rate 0.0001
```

## Results

The training and validation results will be displayed during the training process. The final model will be saved to the `models` directory.

## License

This project is licensed under the MIT License.
