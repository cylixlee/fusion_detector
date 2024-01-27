import sys
import unittest

import torch
from test_common import PROJECT_DIRECTORY

project_path = str(PROJECT_DIRECTORY)
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import src.fusion_detector.thirdparty.pytorch_cifar10.module as M
from src.fusion_detector import dataset, perturbation

PRETRAINED_MODULE_CONSTRUCTORS = {
    "vgg19_bn": M.vgg19_bn,
    "resnet50": M.resnet50,
    "densenet121": M.densenet121,
    "mobilenet_v2": M.mobilenet_v2,
    "googlenet": M.googlenet,
    "inception_v3": M.inception_v3,
}


def no_attack(x: torch.Tensor) -> torch.Tensor:
    return x


class CleverhansTestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.x, self.labels = dataset.CifarBatchDataset().batch
        self.modules = {}
        for name, constructor in PRETRAINED_MODULE_CONSTRUCTORS.items():
            self.modules[name] = constructor(pretrained=True)

    def test_cross_attacks(self) -> None:
        for victim_name, victim in self.modules.items():
            for reference_name, reference in self.modules.items():
                attacks = [
                    no_attack,
                    perturbation.fgsm(reference, 0.1),
                    perturbation.pgd(reference, 0.1, 0.05, 10),
                    perturbation.sparsel1(reference, 0.1, 0.05, 10),
                ]
                for attack in attacks:
                    adversarial = attack(self.x)
                    y = victim(adversarial)
                    predict = torch.argmax(y, dim=1)
                    # Collect statistics
                    correct = torch.eq(predict, self.labels).sum().item()
                    total = len(self.labels)
                    print(
                        f"reference={reference_name}",
                        f"victim={victim_name}",
                        f"accuracy={correct / total}",
                    )
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
