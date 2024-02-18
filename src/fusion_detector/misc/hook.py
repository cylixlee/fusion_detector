from typing import *

import torch
from torch import nn

__all__ = [
    "LayerOutputCollectedModuleWrapper",
    "LayerOutputsCollectedModuleWrapper",
]


def layer_of(module: nn.Module, pattern: str) -> nn.Module:
    parts = pattern.split(".")
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


class LayerOutputCollectedModuleWrapper(object):
    def __init__(self, module: nn.Module, layer_pattern: str) -> None:
        self.underlying = module
        self.layer = layer_of(module, layer_pattern)
        self._output: Optional[torch.Tensor] = None

        def capture_output(
            target: nn.Module,
            input_tuple: Tuple[torch.Tensor, ...],
            output_tensor: torch.Tensor,
        ) -> None:
            self._output = output_tensor.clone().detach()

        self.hook = self.layer.register_forward_hook(capture_output)

    def __del__(self) -> None:
        self.hook.remove()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.underlying(x)

    @property
    def intermediate(self) -> torch.Tensor:
        return self._output


class LayerOutputsCollectedModuleWrapper(object):
    def __init__(self, module: nn.Module, layer_pattern: str) -> None:
        self.underlying = module
        self.layer = layer_of(module, layer_pattern)
        self._outputs: List[torch.Tensor] = []

        def capture_outputs(
            target: nn.Module,
            input_tuple: Tuple[torch.Tensor, ...],
            output_tensor: torch.Tensor,
        ) -> None:
            """A hook function matching PyTorch's required form.

            Appends the output to the `values` each time the module is `forward`ed.

            Arguments:
                target (Module): representing the module the hook is registered to,
                    often a layer in a machine-learning module.
                input_tuple (tuple of Tensor): input(s) of this layer, may contain multiple
                    tensors.
                output_tensor (Tensor): the output tensor of this layer.
            """
            self._outputs.append(output_tensor.clone().detach())

        self.hook = self.layer.register_forward_hook(capture_outputs)

    def __del__(self) -> None:
        self.hook.remove()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.underlying(x)

    @property
    def intermediates(self) -> List[torch.Tensor]:
        return self._outputs
