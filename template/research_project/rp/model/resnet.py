import torch.nn as nn
from torchvision import models

from rp.registry import MODEL


@MODEL.register_module(force=True)
class ResNet34(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        A ResNet-based image classification model.

        :param num_classes: Number of classes for classification.
        :param pretrained: If True, use a ResNet model pre-trained on ImageNet.
        """
        super(ResNet34, self).__init__()

        # Load a pre-trained ResNet model
        self.model = models.resnet34(
            pretrained=pretrained
        )  # Use resnet34 as an example

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass for the model.

        :param x: Input tensor of shape (batch_size, 3, H, W).
        :return: Output tensor of shape (batch_size, num_classes).
        """
        return self.model(x)

    def freeze_backbone(self):
        """
        Freeze the backbone (pre-trained ResNet layers) for feature extraction.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        # Keep the fully connected layer trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreeze the backbone (pre-trained ResNet layers) for fine-tuning.
        """
        for param in self.model.parameters():
            param.requires_grad = True
