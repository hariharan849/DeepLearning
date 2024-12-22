import torch, torchvision

class Alexnet:
    def __init__(self, class_names, device=None):
        self.class_names = class_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def build(self):

        weights = torchvision.models.AlexNet_Weights.DEFAULT

        # Get the transforms used to create our pretrained weights
        auto_transforms = weights.transforms()
        model = torchvision.models.alexnet(weights=weights)

        
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in model.parameters():
            param.requires_grad = False

        # Get the length of class_names (one output unit for each class)
        output_shape = len(self.class_names)

        model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=9216,
                        out_features=512, # same number of output units as our number of classes
                        bias=True),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=512,
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(self.device)

        model = model.to(self.device)
        return model