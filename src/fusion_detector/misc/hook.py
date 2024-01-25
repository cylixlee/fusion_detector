from typing import *

import torch
from torch import nn

__all__ = ["LayerOutputValueCollector", "LayerOutputValuesCollector"]

from . import console


class LayerOutputValueCollector(object):
    """A context-managed class for hooking out result of a specified layer of a module.

    Usage::

        import torch
        from torch import nn

        module = nn.Sequential(
            nn.Linear(224, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        with LayerOutputValueCollector(module[0]) as collector:
            module(torch.rand(1, 224))
            print(collector.value)

    You can use this class to collect the output of a specified layer of any PyTorch module.
    """

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        """ References to the layer whose output is being collected. """

        self.value: Optional[torch.Tensor] = None
        """ A cloned and detached Tensor if module is forwarded, None otherwise. """

        def hook(
            target: nn.Module,
            input_tuple: Tuple[torch.Tensor, ...],
            output_tensor: torch.Tensor,
        ) -> None:
            """A hook function matching PyTorch's required form.

            Assigns the `value` attribute with the latest `forward` output.

            Arguments:
                target (Module): representing the module the hook is registered to, often 
                    a layer in a machine-learning module.
                input_tuple (tuple of Tensor): input(s) of this layer, may contain multiple 
                    tensors.
                output_tensor (Tensor): the output tensor of this layer.
            """
            self.value = output_tensor.clone().detach()

        self.hook = hook

    def __enter__(self) -> "LayerOutputValueCollector":
        """Registers the hook when entering `with` block(s)."""
        self._registered_hook = self.layer.register_forward_hook(self.hook)
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        """Removes the hook when exiting `with` block(s). 

        Exception-related arguments are ignored.
        """
        if exception_value is not None:
            console.error(str(exception_value))
        self._registered_hook.remove()


class LayerOutputValuesCollector(object):
    """A context-managed class for hooking out results of a specified layer of a module.

    Note that **all** results of `forward` within a `with` block are collected in the 
        `values` attribute.
    Usage::

        import torch
        from torch import nn

        module = nn.Sequential(
            nn.Linear(224, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        with LayerOutputValueListCollector(module[0]) as collector:
            module(torch.rand(1, 224))
            module(torch.rand(1, 224))
            module(torch.rand(1, 224))
            for value in collector.values:
                print(value)

    You can use this class to collect the outputs of a specified layer of any PyTorch module.
    """

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        """ References to the layer whose output is being collected. """

        self.values: List[torch.Tensor] = []
        """ List of cloned and detached Tensors if module is forwarded, empty otherwise. """

        def hook(
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
            self.values.append(output_tensor.clone().detach())

        self.hook = hook

    def __enter__(self) -> "LayerOutputValuesCollector":
        """Registers the hook when entering `with` block(s)."""
        self._registered_hook = self.layer.register_forward_hook(self.hook)
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        """Removes the hook when exiting `with` block(s). 

        Exception-related arguments are ignored.
        """
        if exception_value is not None:
            console.error(str(exception_value))
        self._registered_hook.remove()
