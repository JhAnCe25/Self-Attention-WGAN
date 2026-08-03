import torch.nn as nn


def initialize_weights(model):
    """Applies Xavier uniform init to Conv/Linear layers and DCGAN init to BatchNorm."""
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
