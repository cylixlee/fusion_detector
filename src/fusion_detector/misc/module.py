import os
from abc import ABC, abstractmethod
from pathlib import PurePath
from typing import *

import torch
import torchsummary
from torch import Tensor, nn, optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange

from .device import apply_device
from .utility import (ListOrElement, ListOrElementProxy, RedirectStream,
                      TensorTransformAction)

__all__ = [
    "OptimizerConstructor",
    "OptimizerConstructive",
    "MODULE_SAVE_SUFFIX",
    "AbstractModuleProxy",
    "ModuleProxy",
    "MultiModuleProxy",
    "AdversarialModuleProxy",
]

# Type aliases
OptimizerConstructor = Callable[[Iterator[nn.Parameter]], optim.Optimizer]
OptimizerConstructive = Union[optim.Optimizer, OptimizerConstructor]

# Constants
MODULE_SAVE_SUFFIX = ".pt"


class AbstractModuleProxy(ABC):
    """Abstract base class for module proxy.

    A module proxy is a wrapper object that handles module training, evaluating and
    testing.

    Abstract methods:
        train: train the managed module(s) on a dataset.
        evaluate: forward the given input(s) and returns output(s).
        test: test module(s) accuracy on a dataset.

    Virtual methods:
        Actually Python does not provide a `virtual` keyword: every method can be
        overridden. The methods below are ones that *wants* override in some
        special cases.

        preprocess_input: aims to preprocess the input tensor before forwarding.
        preprocess_label: aims to preprocess the label tensor before forwarding.
        postprocess_output: when forwarded, postprocess the output tensor to fit with the 
            label. This is important before calculating accuracy.
    """

    @abstractmethod
    def train(self, data: DataLoader, epochs: int) -> ListOrElement[float]:
        """Train the managed module(s) on a dataset.

        Parameters:
            data: a DataLoader object which loads the dataset to train module(s) on.
            epochs: how many times to train.

        Returns:
            a floating number representing the final loss when finish training, or a list
            of floating numbers if the proxy holds multiple modules.
        """
        ...

    @abstractmethod
    def evaluate(self, x: torch.Tensor) -> ListOrElement[torch.Tensor]:
        """Forward the given input(s) and returns output(s).

        Parameters:
            x: the tensor passes to the module to forward.
        """
        ...

    @abstractmethod
    def test(self, data: DataLoader) -> ListOrElement[float]:
        """Test module(s) accuracy on a dataset.

        Parameters:
            data: the DataLoader object which loads the dataset to test module(s) on.
        """
        ...

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess the input before forwarding."""
        return x.float()

    def preprocess_label(self, label: torch.Tensor) -> torch.Tensor:
        """Preprocess the label before forwarding."""
        return label.float()

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """Postprocess the output tensor to fit with the label. 

        This is important before calculating accuracy.
        """
        return torch.argmax(y, dim=1).float()


class ModuleProxy(AbstractModuleProxy):
    """Proxy for a single module.

    Encapsulates the common train/evaluate/test pipeline, and offers some utility methods.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        optimizer: Optional[OptimizerConstructive] = None,
        criteria: Optional[TensorTransformAction] = None,
    ) -> None:
        self.module: nn.Module = apply_device(module)
        self.optimizer: Optional[optim.Optimizer] = None
        if optimizer is not None:
            if isinstance(optimizer, optim.Optimizer):
                self.optimizer = optimizer
            else:
                self.optimizer = optimizer(module.parameters())
        self.criteria: Optional[TensorTransformAction] = None
        if criteria is not None:
            self.criteria = criteria

    def train_once(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        *,
        once_optimizer: Optional[OptimizerConstructive] = None,
        once_criteria: Optional[TensorTransformAction] = None,
    ) -> float:
        """Train the module on a single BATCH.

        Uses temporary optimizer and criteria (loss function) if provided. Otherwise use
            the ones passed in when initializing. If no optimizer or criteria available,
            that will lead to an assertion failure.

        Parameters:
            x: input tensor of the batch.
            label: label tensor of the batch.
            once_optimizer: temporary optimizer, used if provided.
            once_criteria: temporary criteria, used if provided.
        """
        # Configure optimizer and loss function.
        assert self.optimizer is not None or once_optimizer is not None
        assert self.criteria is not None or once_criteria is not None
        optimizer = once_optimizer if once_optimizer is not None else self.optimizer
        criteria = once_criteria if once_criteria is not None else self.criteria
        # Set module training mode.
        self.module.train()
        # Apply device.
        x, label = apply_device(x, label)
        # Preprocess input and label.
        x = self.preprocess_input(x)
        label = self.preprocess_label(label)
        # Zero-grad and forward.
        optimizer.zero_grad()
        y: torch.Tensor = self.module(x)
        # Postprocess output to fit label.
        y = self.postprocess_output(y)
        # Backward and step.
        loss: torch.Tensor = criteria(y, label)
        loss.requires_grad_()  # WTF?
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, data: DataLoader, epochs: int) -> float:
        loss = 0
        with tqdm(total=epochs, desc="Training") as progress:
            for _ in range(epochs):
                loss = 0
                for x, label in data:
                    loss += self.train_once(x, label)
                progress.set_postfix({"loss": loss})
                progress.update()
        return loss

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        self.module.eval()
        with torch.no_grad():
            return self.module(apply_device(x))

    def test(self, data: DataLoader) -> float:
        self.module.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(data), desc="Test") as progress:
                for x, label in data:
                    # Apply device
                    x, label = apply_device(x, label)
                    # Preprocess input and label
                    x = self.preprocess_input(x)
                    label = self.preprocess_label(label)
                    # Forward
                    y = self.module(x)
                    # Postprocess output to fit label
                    y = self.postprocess_output(y)
                    # Collect statistics
                    correct += torch.eq(y, label).sum().item()
                    total += len(label)
                    # Update progress
                    progress.set_postfix({"accuracy": correct / total})
                    progress.update()
            return correct / total

    def save(self, directory: str, name: str) -> None:
        """Saves the model's state_dict to a local file."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(
            self.module.state_dict(),
            os.path.join(directory, name + MODULE_SAVE_SUFFIX),
        )

    def summary(
        self,
        input_shape: Any = (3, 224, 224),
        path: Optional[os.PathLike] = None,
    ) -> None:
        with RedirectStream(path):
            torchsummary.summary(self.module, input_shape)

    def structure(self, path: Optional[os.PathLike] = None) -> None:
        with RedirectStream(path):
            print(self.module)


class MultiModuleProxy(AbstractModuleProxy):
    """Proxy for multiple modules.

    Those modules may be trained on a same dataset, or may be pretrained modules so that
        we can evaluate/test them all in once.
    Note that dataset are iterated early than modules in this class, in order to save 
        CPU time.

    For example, a possible piece of pseudo code shoule looks like::

        for epoch in epochs:
            for x, label in data:
                for module in modules:
                    train_module(module, x, label)

    In this case, dataset will be iterated by epoch times. If we put dataset iteration
        later than module iteration, then it will be iterated by (epoch * modules) times.
        That's a considerable performance loss.
    """

    def __init__(
        self,
        *modules: Union[nn.Module, ModuleProxy],
        optimizer: Optional[ListOrElement[OptimizerConstructive]] = None,
        criteria: Optional[ListOrElement[TensorTransformAction]] = None,
    ) -> None:
        self.proxies: List[ModuleProxy] = []
        for module in modules:
            if isinstance(module, ModuleProxy):
                self.proxies.append(module)
            else:
                self.proxies.append(ModuleProxy(module))
        self.optimizer: Optional[ListOrElementProxy[optim.Optimizer]] = None
        if optimizer is not None:
            self.optimizer = ListOrElementProxy(optimizer)
        self.criteria: Optional[ListOrElementProxy[TensorTransformAction]] = None
        if criteria is not None:
            self.criteria = ListOrElementProxy(criteria)

    def train(self, data: DataLoader, epochs: int) -> List[float]:
        """Train all managed modules on a single dataset.

        Note that even if a ModuleProxy is passed in when initialized, its optimizer and
            criteria are ignored since we just call its train_once method with the over-
            ridden optimizers and loss functions.
        """
        last_losses: List[float] = [0 for _ in range(len(self.proxies))]
        losses: List[float]
        # Iterate epochs.
        with tqdm(total=epochs, desc="Training (Epoch)") as epoch_progress:
            for _ in range(epochs):
                losses = [0 for _ in range(len(self.proxies))]
                # Iterate dataset and train modules.
                # We're not iterating modules first because loading dataset requires
                #     a considerable CPU time. We should iterate the dataset as less
                #     as possible.
                for x, label in tqdm(data, desc="Dataset"):
                    x, label = apply_device(x, label)
                    # Iterate models and train once. Dataset-first iteration's drawback
                    #     is that we cannot see the loss change intuitively.
                    with tqdm(total=len(self.proxies), desc="Module") as module_progress:
                        for index, proxy in enumerate(self.proxies):
                            loss = proxy.train_once(
                                x=x,
                                label=label,
                                once_optimizer=self.optimizer.decide(index),
                                once_criteria=self.criteria.decide(index),
                            )
                            losses[index] += loss
                            module_progress.set_postfix({
                                "module": index,
                                "loss": loss,
                            })
                            module_progress.update()
                # Calculate the maximum convergence of modules per epoch is a compromise
                #     in order to see how's the losses change.
                max_convergence = 0
                for index, loss in enumerate(losses):
                    max_convergence = max(
                        max_convergence,
                        last_losses[index] - loss,
                    )
                epoch_progress.set_postfix({
                    "max convergence": max_convergence,
                })
                epoch_progress.update()
                last_losses = losses
        return losses

    def evaluate(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [proxy.evaluate(x) for proxy in self.proxies]

    def test(self, data: DataLoader) -> List[float]:
        for proxy in self.proxies:
            proxy.module.eval()
        corrects: List[int] = [0 for _ in range(len(self.proxies))]
        total: int = 0
        with torch.no_grad():
            for x, label in trange(data, desc="Test (Dataset)"):
                # Collect statistics `total` because the dataset is only iterated once
                #     in testing.
                total += len(label)
                # Apply device.
                x, label = apply_device(x, label)
                # Preprocess input and label.
                x = self.preprocess_input(x)
                label = self.preprocess_label(label)
                with tqdm(total=len(self.proxies), desc="Module") as module_progress:
                    for index, proxy in enumerate(self.proxies):
                        y = proxy.evaluate(x)
                        # Postprocess output to fit label.
                        y = self.postprocess_output(y)
                        # Collect statistics.
                        corrects[index] += torch.eq(y, label).sum().item()
                        # Update progress.
                        module_progress.set_postfix({
                            "module": index,
                            "accuracy": corrects[index] / total,
                        })
                        module_progress.update()
        return [correct / total for correct in corrects]

    def save(self, directory: str, names: List[str]) -> None:
        for index, proxy in enumerate(self.proxies):
            proxy.save(directory, names[index])

    def summary(
        self,
        input_shape: Any = (3, 224, 224),
        directory: Optional[os.PathLike] = None,
        names: Optional[List[str]] = None,
    ) -> None:
        if directory is not None or names is not None:
            assert directory is not None and names is not None
            for index, proxy in enumerate(self.proxies):
                proxy.summary(
                    input_shape,
                    PurePath(
                        directory,
                        names[index],
                    ),
                )
        else:
            for proxy in self.proxies:
                proxy.summary(input_shape)

    def structure(
        self,
        directory: Optional[os.PathLike] = None,
        names: Optional[List[str]] = None,
    ) -> None:
        if directory is not None or names is not None:
            assert directory is not None and names is not None
            for index, proxy in enumerate(self.proxies):
                proxy.structure(PurePath(directory, names[index]))
        else:
            for proxy in self.proxies:
                proxy.structure()


class AdversarialModuleProxy(ModuleProxy):
    def __init__(
        self,
        module: nn.Module,
        attack: ListOrElement[TensorTransformAction],
        *,
        optimizer: Optional[OptimizerConstructive] = None,
        criteria: Optional[TensorTransformAction] = None,
    ) -> None:
        super().__init__(module, optimizer=optimizer, criteria=criteria)
        self.attack = ListOrElementProxy(attack)

    def preprocess_input(self, x: Tensor) -> Tensor:
        # Generare adversarial examples and append them to the input tensor.
        x = super().preprocess_input(x)
        adversarial_examples = (attack(x) for attack in self.attack.tolist())
        inputs = torch.concat((x, *adversarial_examples))
        # Shuffle the preprocessed input.
        self.shuffle = torch.randperm(inputs.size(0))
        return inputs[self.shuffle, ...]

    def preprocess_label(self, label: Tensor) -> Tensor:
        # Replace categories to zeros and ones.
        # Zeros represents normal examples, ones represents adversarial examples.
        label = super().preprocess_label(label)
        normal_labels = torch.zeros_like(label)
        adversarial_labels = (torch.ones_like(label)
                              for _ in range(len(self.attack)))
        labels = torch.concat((normal_labels, *adversarial_labels))
        # Shuffle accordingly.
        return labels[self.shuffle, ...]

    def test_per_attack(self, data: DataLoader) -> List[float]:
        self.module.eval()
        detected = [0 for _ in range(len(self.attack))]
        total = 0
        with torch.no_grad():
            for x, label in tqdm(data, desc="Test Per Attack"):
                total += len(label)
                # Apply device
                x, label = apply_device(x, label)
                for index, attack in enumerate(self.attack.tolist()):
                    # Preprocess input and label
                    x = attack(x)
                    label = torch.ones_like(label)
                    # Forward
                    y = self.module(x)
                    # Postprocess output to fit label
                    y = self.postprocess_output(y)
                    # Collect statistics
                    detected[index] += torch.eq(y, label).sum().item()
            return [defend / total for defend in detected]
