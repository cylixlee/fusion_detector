from typing import *

import torch
from torch import nn

from fusion_detector.misc import console
from fusion_detector.misc.singleton import SingletonConstructor

__all__ = ["DeviceManager", "apply_device"]

_TDeviceApplicable = TypeVar("_TDeviceApplicable", bound=Union[nn.Module, torch.Tensor])


@SingletonConstructor
class _DeviceManager(object):
    """A singleton class holding a device attribute indicating whether to use GPU or CPU.

    It's redundant to determine current device **EVERYWHERE**,
        so let's use a global, singleton class.
    """

    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    def apply(self, target: _TDeviceApplicable) -> _TDeviceApplicable:
        """Receives an object which can be transported to devices.

        Here we just use a `TypeVar(bound=(nn.Module, torch.Tensor))`, so only
            `nn.Module` and `torch.Tensor` is accepted.
        Note that if there's multiple GPUs available, `nn.Module`s will be wrapped by
            `nn.DataParallel` to take advantage.
        """
        result = target.to(self.device)
        if isinstance(target, nn.Module):
            if torch.cuda.device_count() > 1:
                console.message(f"=== Using {torch.cuda.device_count()} GPUs ===")
                result = nn.DataParallel(result)
        return result


DeviceManager = _DeviceManager()
""" A singleton instance holding a device attribute indicating whether to use GPU or CPU.
    
It's redundant to determine current device EVERYWHERE,
    so let's use a global, singleton class.
"""


def apply_device(
    first: _TDeviceApplicable,
    *more: _TDeviceApplicable,
) -> Union[_TDeviceApplicable, Tuple[_TDeviceApplicable, ...]]:
    """A batched shortcut of DeviceManager.apply.

    It receives multiple targets that are device-applicable, and apply
        them to current device.
    Returns a tuple in the same form of inputs, so we can use tuple deconstruction.
    """
    if len(more) > 0:
        return (
            first.to(DeviceManager.device),
            *(target.to(DeviceManager.device) for target in more),
        )
    return first.to(DeviceManager.device)
