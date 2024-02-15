import unittest

import inclconf
import torch

inclconf.configure_includepath()

import src.fusion_detector.thirdparty.pytorch_cifar10.module as M
from src.fusion_detector import datasource, extractor, misc


class HookTestCase(unittest.TestCase):
    def test_intermediate_correctness(self):
        x, _ = datasource.CifarBatchDataSource(8).batch()
        resnet = M.resnet18(pretrained=True)
        wrapper = misc.LayerOutputCollectedModuleWrapper(resnet, "avgpool")

        # Use misc.LayerOutputCollectedModuleWrapper to test
        wrapper(x)
        intermediate = wrapper.intermediate
        intermediate = intermediate.reshape(intermediate.size(0), -1)  # Resnet trick.
        result = resnet(x)
        indirect_result = resnet.fc(intermediate)
        self.assertTrue(torch.equal(result, indirect_result))

        # Use extractor.IntermediateLayerFeatureExtractor, a higher encapsulation of
        # the former one.
        ext = extractor.IntermediateLayerFeatureExtractor(resnet, "avgpool")
        intermediate = ext(x)
        intermediate = intermediate.reshape(intermediate.size(0), -1)  # Resnet trick.
        indirect_result = resnet.fc(intermediate)
        self.assertTrue(torch.equal(result, indirect_result))


if __name__ == "__main__":
    unittest.main()
