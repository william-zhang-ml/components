""" Object detection heads. """
import torch
from torch import nn


class ConvDetector(nn.Module):
    """ Object detection head as a conv layer. """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 anchors: torch.Tensor) -> None:
        """"""
        super().__init__()
        self.in_channels, self.num_classes = in_channels, num_classes
        self.register_buffer('anchors', anchors)
        self.layers = nn.Conv2d(
            in_channels=in_channels,
            out_channels=(5 + num_classes) * len(anchors),
            kernel_size=1
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute object detection logits and activations.

        Args:
            inp: input feature map (B, C, H, W)

        Returns:
            torch.Tensor: logits (if training) or activations (if eval)
        """
        if self.training:
            outp = self.layers(inp).view(
                inp.shape[0],
                5 + self.num_classes,
                len(self.anchors),
                inp.shape[-2],
                inp.shape[-1]
            )
        else:
            with torch.no_grad():
                outp = self.layers(inp).view(
                    inp.shape[0],
                    5 + self.num_classes,
                    len(self.anchors),
                    inp.shape[-2],
                    inp.shape[-1]
                )
                det_score = outp[:, :1].sigmoid()
                center_offset = outp[:, 1:3].sigmoid()
                size_offset = outp[:, 3:5].exp()
                size_offset *= self.anchors.T.view(1, 2, -1, 1, 1)
                cls_score = outp[:, 5:].argmax(dim=1, keepdim=True)
                outp = torch.cat(
                    [det_score, center_offset, size_offset, cls_score],
                    dim=1
                )
        return outp
