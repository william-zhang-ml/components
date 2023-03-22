""" Utility functions. """
from torch import nn


def count_params(module: nn.Module) -> int:
    """Count the number of parameters in a module.
    Args:
        module (nn.Module): module of interest

    Returns:
        int: number of module parameters
    """
    return sum([p.nelement() for p in module.parameters()])


def repeat_layer(num: int, layer: type, kwargs: dict) -> nn.Sequential:
    """Chain a sequence of identical layers.

    Args:
        num (int): number of layers
        layer (type): layer class
        kwargs (dict): layer arguments

    Returns:
        nn.Sequential: sequence of identical layers
    """
    layers = [layer(**kwargs) for _ in range(num)]
    return nn.Sequential(*layers)
