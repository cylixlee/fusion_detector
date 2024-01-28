import sys

from test_common import PROJECT_DIRECTORY

source_path = str(PROJECT_DIRECTORY / "src")
if str(source_path) not in sys.path:
    sys.path.insert(0, str(source_path))

import unittest

import torch

from fusion_detector import dataset, misc
from fusion_detector.thirdparty.pytorch_cifar10 import module as M


class LayerOutputValueCollectorTestCase(unittest.TestCase):
    def test_hook(self) -> None:
        resnet = M.resnet18(pretrained=True)
        x, _ = dataset.NormalizedCifarBatchDataset().batch
        with misc.LayerOutputValueCollector(resnet.avgpool) as collector:
            y: torch.Tensor = resnet(x)
            hookvalue = collector.value
        hooky: torch.Tensor = resnet.fc(hookvalue.view(-1, 512))
        self.assertTrue(torch.equal(y, hooky))


if __name__ == "__main__":
    unittest.main()
