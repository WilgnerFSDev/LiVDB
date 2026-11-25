import torch
import torch.nn as nn


class ReVealMLP(nn.Module):
    def __init__(self, input_dim: int = 200):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(256, 1)

    def forward(self, xg):
        h = self.transform(xg)
        yg = torch.sigmoid(self.classifier(h)).squeeze(-1)
        return h, yg

    def predict(self, xg):
        return self.forward(xg)

    def transform_only(self, xg):
        return self.transform(xg)
