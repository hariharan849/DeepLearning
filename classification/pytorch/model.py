import torch, torchvision
from torch import nn

# Alexnet state of the Art model
class AlexnetSotA:
    """A class for building the AlexNet state-of-the-art model."""

    def __init__(self, class_names, device=None):
        """Initializes the AlexnetSotA class.
        
        Args:
            class_names (list): List of class names.
            device (str, optional): Device to run the model on. Defaults to "cuda" if available.
        """
        self.class_names = class_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def build(self):
        """Builds the AlexNet model with pretrained weights.
        
        Returns:
            model (torch.nn.Module): The AlexNet model.
            auto_transforms (torchvision.transforms): The transforms used for the pretrained weights.
        """
        weights = torchvision.models.AlexNet_Weights.DEFAULT

        # Get the transforms used to create our pretrained weights
        auto_transforms = weights.transforms()
        model = torchvision.models.alexnet(weights=weights)
        
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in model.features.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers for fine-tuning
        for param in model.features[-4:].parameters():
            param.requires_grad = True

        # Get the length of class_names (one output unit for each class)
        output_shape = len(self.class_names) if len(self.class_names) != 2 else 1

        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=9216, out_features=512, bias=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=output_shape, bias=True)
        ).to(self.device)

        model = model.to(self.device)
        return model, auto_transforms
    
class AlexnetFromScratch(nn.Module):
    def __init__(self, classes=1):
        super().__init__()

        self.feature1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25),
        )
        self.feature2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25)
         )
        self.feature3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(384)
        )
        self.feature4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(384)
        )
        self.feature5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(20736, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096, classes)
        )

    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.feature5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class AlexnetFromScratchV2(nn.Module):
    def __init__(self, classes=1):
        super().__init__()

        self.feature1 = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25),
        )
        self.feature2 = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25)
         )
        self.feature3 = nn.Sequential(
            nn.LazyConv2d(384, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.LazyBatchNorm2d()
        )
        self.feature4 = nn.Sequential(
            nn.LazyConv2d(384, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.LazyBatchNorm2d()
        )
        self.feature5 = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.LazyBatchNorm2d()
        )
        self.classifier1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(classes)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.feature5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)
        return x



class ResnetIdentityBlockV1(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, padding=1),  # Change padding to 1
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(in_channels=cout, out_channels=cout, kernel_size=3, padding=1),  # Change padding to 1
            nn.BatchNorm2d(cout)
        )
        self.shortcut = nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()

    def forward(self, x):
        x_identity = self.shortcut(x)
        x = self.block(x)
        x = torch.add(x_identity, x)
        return self.act(x)

class ResnetFromScratchV1(nn.Module):
    def __init__(self, classes=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),  # Change padding to 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)  # Change padding to 1
        )

        self.blocks = nn.ModuleList()
        block_layers = [3, 4, 6, 3]
        filter_size = [64, 128, 256, 512]

        for i in range(len(block_layers)):
            if i == 0:
                for j in range(block_layers[i]):
                    self.blocks.append(ResnetIdentityBlockV1(filter_size[i], filter_size[i]))
            else:
                self.blocks.append(ResnetIdentityBlockV1(filter_size[i-1], filter_size[i]))
                for j in range(block_layers[i] - 1):
                    self.blocks.append(ResnetIdentityBlockV1(filter_size[i], filter_size[i]))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filter_size[-1], classes)

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResnetIdentityBlockV2(nn.Module):
    def __init__(self, cout):
        super().__init__()

        self.block = nn.Sequential(
            nn.LazyConv2d(out_channels=cout, kernel_size=3, padding=1),  # Change padding to 1
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=cout, kernel_size=3, padding=1),  # Change padding to 1
            nn.LazyBatchNorm2d()
        )
        self.shortcut = nn.LazyConv2d(cout, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()

    def forward(self, x):
        x_identity = self.shortcut(x)
        x = self.block(x)
        x = torch.add(x_identity, x)
        return self.act(x)

class ResnetFromScratchV2(nn.Module):
    def __init__(self, classes=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),  # Change padding to 3
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)  # Change padding to 1
        )

        self.blocks = nn.ModuleList()
        block_layers = [3, 4, 6, 3]
        filter_size = [64, 128, 256, 512]

        for i in range(len(block_layers)):
            if i == 0:
                for j in range(block_layers[i]):
                    self.blocks.append(ResnetIdentityBlockV2(filter_size[i]))
            else:
                self.blocks.append(ResnetIdentityBlockV2(filter_size[i]))
                for j in range(block_layers[i] - 1):
                    self.blocks.append(ResnetIdentityBlockV2(filter_size[i]))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LazyLinear(classes)

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

class InceptionBlock(nn.Module):
    def __init__(self, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.LazyConv2d(filters_1x1, kernel_size=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.LazyConv2d(filters_3x3_reduce, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(filters_3x3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.LazyConv2d(filters_5x5_reduce, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(filters_5x5, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv2d(filters_pool_proj, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)
    
class InceptionModelFromScratch(nn.Module):
    def __init__(self, classes=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),  # Change padding to 3
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)  # Change padding to 1
        )

        self.blocks = nn.ModuleList()
        block_layers = [2, 2, 2, 2]
        filter_size = [64, 128, 256, 512]

        for i in range(len(block_layers)):
            if i == 0:
                for j in range(block_layers[i]):
                    self.blocks.append(InceptionBlock(filter_size[i], filter_size[i], filter_size[i], filter_size[i], filter_size[i], filter_size[i]))
            else:
                self.blocks.append(InceptionBlock(filter_size[i], filter_size[i], filter_size[i], filter_size[i], filter_size[i], filter_size[i]))
                for j in range(block_layers[i] - 1):
                    self.blocks.append(InceptionBlock(filter_size[i], filter_size[i], filter_size[i], filter_size[i], filter_size[i], filter_size[i]))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LazyLinear(classes)

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

class DenseBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, padding=1),  # Change padding to 1
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )

    def forward(self, x):
        x_skip = x
        x = self.block(x)
        x = torch.cat([x_skip, x], 1)
        return x
    
class TransitionBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.block(x)
        return x

    
class DenseNetFromScratch(nn.Module):
    def __init__(self, classes=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),  # Change padding to 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Change padding to 1
        )

        self.blocks = nn.ModuleList()
        block_layers = [6, 12, 24, 16]
        filter_size = [64, 128, 256, 512]

        for i in range(len(block_layers)):
            if i == 0:
                for j in range(block_layers[i]):
                    self.blocks.append(DenseBlock(filter_size[i], 32))
            else:
                self.blocks.append(TransitionBlock(filter_size[i-1], filter_size[i]))
                for j in range(block_layers[i]):
                    self.blocks.append(DenseBlock(filter_size[i] + 32 * j, 32))
       
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filter_size[-1], classes)

    def forward(self, x):
        x = self.block1(x)
        print(x.shape)
        for block in self.blocks:
            x = block(x)
            print(x.shape)
        x = self.avg_pool(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class MobileNetBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=3, stride=stride, padding=1, groups=cin),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
class MobileNetFromScratch(nn.Module):
    def __init__(self, classes=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        block_layers = [1, 2, 3, 4, 3, 3]
        filter_size = [64, 128, 256, 512, 1024, 2048]
        strides = [1, 2, 2, 2, 1, 2]

        for i in range(len(block_layers)):
            for j in range(block_layers[i]):
                self.blocks.append(MobileNetBlock(filter_size[i], filter_size[i], strides[i]))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filter_size[-1], classes)

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
