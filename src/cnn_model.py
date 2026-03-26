import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    """
    Simple CNN classifier. Design your own architecture.
    Forward signature MUST be forward(x):
        x: (B, 28, 28)
    """
    def __init__(self, w=28, h=28, num_classes=10): # ⚠️ DO NOT change this line
        super().__init__()
        # Feature extraction: two convolutional blocks
        self.feat_extract = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (B,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (B,32,14,14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (B,64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (B,64,7,7)
        )

        # Compute flattened dimension after convolutions
        # Input: (B, 1, 28, 28) -> after first pool: (B,32,14,14) -> after second pool: (B,64,7,7)
        flat_dim = 64 * 7 * 7

        # Classifier: two linear layers with ReLU
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Add channel dimension if missing
        if x.dim() == 3:
            x = x.unsqueeze(1)            # (B,28,28) -> (B,1,28,28)
        elif x.dim() == 4:
            pass                          # already (B,1,28,28)
        else:
            raise ValueError(f"Expected input of shape (B,28,28) or (B,1,28,28), got {tuple(x.shape)}")

        x = self.feat_extract(x)          # (B,64,7,7)
        x = torch.flatten(x, 1)           # (B,64*7*7)
        logits = self.classifier(x)       # (B,num_classes)
        return logits