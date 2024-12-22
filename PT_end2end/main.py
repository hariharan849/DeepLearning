import torch
from .utils import create_writer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .model import Alexnet

model = Alexnet(3)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

writer = create_writer(
    experiment_name="Animal dataset",
    model_name="Alexnet",
    extra="learning animals dataset from torch api")