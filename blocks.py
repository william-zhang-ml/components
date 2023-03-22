""" General purpose blocks. """
from typing import Tuple, Union
import torch
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

    def __repr__(self) -> str:
        argstr = ', '.join([
            f'in_channels={self[-1].in_channels}',
            f'out_channels={self[-1].out_channels}',
            f'kernel_size={self[-1].kernel_size}',
            f'stride={self[-1].stride}',
            f'padding={self[-1].padding}',
            f'dilation={self[-1].dilation}',
            f'groups={self[-1].groups}',
            f'bias={self[-1].bias is not None}',
            f'norm_first={isinstance(self[0], nn.BatchNorm2d)}'
        ])
        return f'ConvBlock({argstr})'


# pylint: disable=too-many-instance-attributes
class SelfAttention2d(nn.Module):
    """ Multihead self-attention layer for images. """
    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 emb_dim: int,
                 enc_norm: int) -> None:
        """Initialize layers.

        Args:
            in_channels (int): expected number of input channels
            num_heads (int): number of attention heads
            emb_dim (int): query, key, and value dimension
            enc_norm (int): row/col positional encoding normalizing factor
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.enc_norm = enc_norm
        self.query_emb = nn.Conv2d(in_channels + 2, num_heads * emb_dim, 1)
        self.key_emb = nn.Conv2d(in_channels + 2, num_heads * emb_dim, 1)
        self.val_emb = nn.Conv2d(in_channels + 2, num_heads * emb_dim, 1)
        self.out_emb = nn.Conv2d(num_heads * emb_dim, in_channels, 1)

    def __repr__(self) -> str:
        arg_str = ', '.join([
            f'in_channels={self.in_channels}',
            f'num_heads={self.num_heads}',
            f'emb_dim={self.emb_dim}',
            f'enc_norm={self.enc_norm}'
        ])
        return f'MultiheadAttn2DSimple({arg_str})'

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute new feature map.

        Args:
            inp (torch.Tensor): input feature map

        Returns:
            (torch.Tensor): output feature map
        """
        # concatenate positional encoding channels
        inp = torch.cat(
            [
                inp,
                self._get_coord_chan(
                    inp.shape[0],
                    inp.shape[2],
                    inp.shape[3],
                ) / self.enc_norm
            ],
            dim=1
        )

        # compute embeddings
        emb_shape = (inp.shape[0], self.num_heads, self.emb_dim, -1)
        query = self.query_emb(inp).view(emb_shape)
        key = self.key_emb(inp).view(emb_shape)
        value = self.val_emb(inp).view(emb_shape)

        # compute new attention-based features
        attention = query.transpose(2, 3) @ key  # (HW,emb_dim) x (emb_dim,HW)
        attention = (attention / (self.emb_dim ** 0.5)).softmax(dim=-2)
        attention = (value @ attention).view(
            inp.shape[0],
            self.num_heads * self.emb_dim,
            inp.shape[2],
            inp.shape[3]
        )

        return self.out_emb(attention)

    @staticmethod
    def _get_coord_chan(nbatch: int,
                        nrows: int,
                        ncols: int) -> torch.Tensor:
        """Make coordinate channels for image feature map.

        Args:
            nbatch (int): batch size
            nrows (int): number of feature map rows
            ncols (int): number of feature map columns

        Returns:
            (torch.Tensor): row and column coordinates
        """
        coords = torch.meshgrid(
            [torch.arange(nrows), torch.arange(ncols)],
            indexing='ij'
        )
        coords = torch.stack(coords)
        return torch.tile(coords, (nbatch, 1, 1, 1))
# pylint: enable=too-many-instance-attributes


class AttentionEncoder2D(nn.Module):
    """ Multihead self-attention block for images. """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 emb_dim: int,
                 enc_norm: int,
                 feedforward_channels: int) -> None:
        """Initialize layers.

        Args:
            in_channels (int): expected number of input channels
            num_heads (int): number of attention heads
            emb_dim (int): query, key, and value dimension
            enc_norm (int): row/col positional encoding normalizing factor
            feedforward_channels (int): number of hidden units in MLP
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            SelfAttention2d(
                 in_channels,
                 num_heads,
                 emb_dim,
                 enc_norm
            ),
        )
        self.feedforward = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, feedforward_channels, 1),
            nn.ReLU(),
            nn.Conv2d(feedforward_channels, in_channels, 1)
        )
    # pylint: enable=too-many-arguments

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute new feature map.

        Args:
            inp (torch.Tensor): input feature map

        Returns:
            (torch.Tensor): output feature map
        """
        features = inp + self.attention(inp)
        return features + self.feedforward(inp)
