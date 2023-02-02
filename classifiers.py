""" Classification heads for convnet backbones. """
import torch
from torch import nn


class AvgPoolClassifier(nn.Module):
    """ Classification head based on global average pooling. """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        """Initialize layer weights.

        Args:
            in_channels (int) : expected number of input channels.
            num_classes (int) : number of target classes.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute classification logits.

        Args:
            inp (torch.Tensor): 2D input feature maps

        Returns:
            torch.Tensor: classification logits.
        """
        return self.layers(inp)

    def __repr__(self) -> str:
        kwargs_str = ', '.join([
            f'in_channels={self.in_channels}',
            f'num_classes={self.num_classes}'
        ])
        return f'AvgPoolClassifier({kwargs_str})'


class MaxPoolClassifier(nn.Module):
    """ Classification head based on global max pooling. """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        """Initialize layer weights.

        Args:
            in_channels (int) : expected number of input channels.
            num_classes (int) : number of target classes.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute classification logits.

        Args:
            inp (torch.Tensor): 2D input feature maps

        Returns:
            torch.Tensor: classification logits.
        """
        return self.layers(inp)

    def __repr__(self):
        kwargs_str = ', '.join([
            f'in_channels={self.in_channels}',
            f'num_classes={self.num_classes}'
        ])
        return f'MaxPoolClassifier({kwargs_str})'
