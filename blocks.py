""" General purpose blocks. """
from typing import Tuple, Union
from torch import nn


class ConvBlock(nn.Sequential):
    """ Standard convolutional neural network block. """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 norm_first: bool = True) -> None:
        """Initialize layer weights.

        Args:
            in_channels (int): number of input neurons/channels
            out_channels (int): number of output neurons/channels
            kernel_size (Union[int, Tuple[int, int]]): kernel height and width
            stride (Union[int, Tuple[int, int]], optional):
                stride height and width. Defaults to 1.
            padding (Union[str, int, Tuple[int, int]], optional):
                row and col pixels to pad. Defaults to 0.
            dilation (Union[int, Tuple[int, int]], optional):
                space between kernel weights. Defaults to 1.
            groups (int, optional): number of input groups. Defaults to 1.
            bias (bool, optional):
                whether conv layer uses bias terms. Defaults to False.
            norm_first (bool, optional):
                whether to batchnorm or relu first. Defaults to True.
        """
        super().__init__()
        if norm_first:
            self.add_module('norm', nn.BatchNorm2d(in_channels))
            self.add_module('relu', nn.ReLU())
        else:
            self.add_module('relu', nn.ReLU())
            self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        )
    # pylint: enable=too-many-arguments
