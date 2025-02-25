import torch, torchvision
from torch import nn
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(self, class_names, device=None, lr=1e-3):
        """ Initialize the base model

        Args:
            class_names (list): List of class names
            device (str, optional): Device to run the model on (default: auto-detect)
            lr (float, optional): Learning rate (default: 1e-3)
        """
        super(BaseModel, self).__init__()
        self.class_names = class_names
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr

    def forward(self, x):
        """ Forward pass through the model

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """ Training step

        Args:
            batch (tuple): Batch of data
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Training loss
        """
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Validation step

        Args:
            batch (tuple): Batch of data
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Validation loss
        """
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        """ Configure optimizers

        Returns:
            torch.optim.Optimizer: Optimizer for training
        """
        optimizer = torch.optim.Adam(self.get_model_parameters(), lr=self.lr)
        return optimizer

    def get_model_parameters(self):
        """ Get model parameters for optimizer

        Returns:
            iterator: Model parameters
        """
        return self.model.parameters()

class AlexnetSotA(BaseModel):
    def __init__(self, class_names, device=None, lr=1e-3):
        """ Initialize the AlexnetSotA model

        Args:
            class_names (list): List of class names
            device (str, optional): Device to run the model on (default: auto-detect)
            lr (float, optional): Learning rate (default: 1e-3)
        """
        super(AlexnetSotA, self).__init__(class_names, device, lr)
        self.model, self.auto_transforms = self.build()

    def build(self):
        """ Build the AlexNet model with custom classifier

        Returns:
            tuple: Model and auto transforms
        """
        weights = torchvision.models.AlexNet_Weights.DEFAULT
        auto_transforms = weights.transforms()
        model = torchvision.models.alexnet(weights=weights)
        
        for param in model.parameters():
            param.requires_grad = False

        output_shape = len(self.class_names)

        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=9216, out_features=512, bias=True),
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=512, out_features=output_shape, bias=True)
        ).to(self._device)

        model = model.to(self._device)
        return model, auto_transforms

class ResnetIdentityBlockV2(nn.Module):
    def __init__(self, cout):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.LazyConv2d(out_channels=cout, kernel_size=3, padding=1),  # Change padding to 1
            torch.nn.LazyBatchNorm2d(),
            torch.nn.ReLU(),
            torch.nn.LazyConv2d(out_channels=cout, kernel_size=3, padding=1),  # Change padding to 1
            torch.nn.LazyBatchNorm2d()
        )
        self.shortcut = torch.nn.LazyConv2d(cout, kernel_size=1, stride=1, padding=0)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x_identity = self.shortcut(x)
        x = self.block(x)
        x = torch.add(x_identity, x)
        return self.act(x)

class ResnetFromScratch(BaseModel):
    def __init__(self, class_names, device=None, lr=1e-3):
        """ Initialize the ResnetFromScratch model

        Args:
            class_names (list): List of class names
            device (str, optional): Device to run the model on (default: auto-detect)
            lr (float, optional): Learning rate (default: 1e-3)
        """
        super(ResnetFromScratch, self).__init__(class_names, device, lr)
        self.model = self.build()

    def build(self):
        """ Build the Resnet model with custom classifier

        Returns:
            nn.Module: Resnet model
        """
        model = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),  # Change padding to 3
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),  # Change padding to 1
            *self._make_layers()
        )
        return model.to(self._device)

    def _make_layers(self):
        layers = []
        block_layers = [3, 4, 6, 3]
        filter_size = [64, 128, 256, 512]

        for i in range(len(block_layers)):
            if i == 0:
                for j in range(block_layers[i]):
                    layers.append(ResnetIdentityBlockV2(filter_size[i]))
            else:
                layers.append(ResnetIdentityBlockV2(filter_size[i]))
                for j in range(block_layers[i] - 1):
                    layers.append(ResnetIdentityBlockV2(filter_size[i]))

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.LazyLinear(len(self.class_names)))
        return layers