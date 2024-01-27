import sys

import torch
from test_common import PROJECT_DIRECTORY

source_path = str(PROJECT_DIRECTORY / "src")
if str(source_path) not in sys.path:
    sys.path.insert(0, str(source_path))

import unittest

from fusion_detector import dataset
from fusion_detector.extractor.mobileres import MobileResExtractor, ResNetKind


class ExtractorTestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.extractor = MobileResExtractor(ResNetKind.RESNET_50)

    def test_mobileres_requires_grad(self):
        for parameter in self.extractor.resnet.parameters():
            self.assertFalse(parameter.requires_grad)
        for parameter in self.extractor.mobilenet.parameters():
            self.assertFalse(parameter.requires_grad)

    def test_mobileres_extraction_shape(self):
        x, _ = dataset.CifarBatchDataset().batch
        y: torch.Tensor = self.extractor(x)
        self.assertTrue(y.shape, (-1, 2048 + 1280, 1, 1))


if __name__ == "__main__":
    unittest.main()
