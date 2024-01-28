import sys
import unittest

from test_common import PROJECT_DIRECTORY

source_path = str(PROJECT_DIRECTORY / "src")
if str(source_path) not in sys.path:
    sys.path.insert(0, str(source_path))

import fusion_detector.thirdparty.pytorch_cifar10.module as M

PRETRAINED_MODULE_CONSTRUCTORS = {
    "vgg11_bn": M.vgg11_bn,
    "vgg13_bn": M.vgg13_bn,
    "vgg16_bn": M.vgg16_bn,
    "vgg19_bn": M.vgg19_bn,
    "resnet18": M.resnet18,
    "resnet34": M.resnet34,
    "resnet50": M.resnet50,
    "densenet121": M.densenet121,
    "densenet161": M.densenet161,
    "densenet169": M.densenet169,
    "mobilenet_v2": M.mobilenet_v2,
    "googlenet": M.googlenet,
    "inception_v3": M.inception_v3,
}


class PretrainedModulesSanityTestCase(unittest.TestCase):
    def test_pretrained_sanity(self):
        try:
            for constructor in PRETRAINED_MODULE_CONSTRUCTORS.values():
                module = constructor(pretrained=True)
            self.assertTrue(True)
        except:
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
