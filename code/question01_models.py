import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet34_Weights

class SimpleReIdModel(nn.Module):
    """
    A vehicle re-identification model with pre-trained ResNet50 backbone for
    feature extraction.  
    """

    def __init__(self, embedding_dim):
        super().__init__()

        weights = ResNet34_Weights.DEFAULT
        pretrained_model = models.resnet34(weights=weights)

        # Keep feature extraction layers, but remove classification layer
        self.backbone = nn.Sequential(*list(pretrained_model.children())[:-2])

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Average pool converts [B, 512, H, W] to [B, 512, 1, 1]
        self.avgpool = pretrained_model.avgpool

        # Flatten before sending into embedding head
        self.flatten = nn.Flatten()

        # Get number of features for embedding head input
        # Need to extract directly from pretrained model
        num_features = pretrained_model.fc.in_features
        first_output = num_features // 2  # Reduce by half

        # Custom embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(num_features, first_output), 
            nn.SiLU(),
            nn.Linear(first_output, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.avgpool(features)
        flattened = self.flatten(pooled)
        embedding = self.embedding_head(flattened)
        norm_embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalization

        return norm_embedding