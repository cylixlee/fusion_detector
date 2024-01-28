import sys
import unittest

import torch
from test_common import PROJECT_DIRECTORY

project_path = str(PROJECT_DIRECTORY)
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import src.fusion_detector.thirdparty.pytorch_cifar10.module as M
from src.fusion_detector import dataset

PRETRAINED_MODULE_CONSTRUCTORS = {
    "vgg19_bn": M.vgg19_bn,
    "resnet50": M.resnet50,
    "densenet121": M.densenet121,
    "mobilenet_v2": M.mobilenet_v2,
    "googlenet": M.googlenet,
    "inception_v3": M.inception_v3,
}


class AdversarialDatasetTestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.batch = dataset.CifarAdversarialBatchDataset().batch

    def test_type_and_shape(self) -> None:
        x, _ = self.batch
        assert isinstance(x, torch.Tensor)
        assert x.shape == torch.Size([32, 3, 32, 32])

    def test_accuracies(self) -> None:
        modules = {}
        for name, constructor in PRETRAINED_MODULE_CONSTRUCTORS.items():
            modules[name] = constructor(pretrained=True)
        x, label = self.batch
        for name, model in modules.items():
            y = model(x)
            predict = torch.argmax(y, dim=1)
            # Collect statistics
            correct = torch.eq(predict, label).sum().item()
            total = len(label)
            print(f"{name} accuracy={correct / total}")


if __name__ == "__main__":
    unittest.main()
