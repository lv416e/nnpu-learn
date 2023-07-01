"""
Note that this test code does not provide 100% coverage.
"""
import unittest

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.datasets import PositiveUnlabeledMNIST


class TestPositiveUnlabeledMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        pul_dataset = PositiveUnlabeledMNIST(train=True, transform=transform)
        pul_dataloader = DataLoader(dataset=pul_dataset, batch_size=64, shuffle=False)
        cls.dataset = pul_dataset
        cls.samples = next(iter(pul_dataloader))

    def test_features_dimensions(self):
        features_tensor_shape = TestPositiveUnlabeledMNIST.samples[0].shape
        self.assertEqual(len(features_tensor_shape), 3)
        self.assertEqual(features_tensor_shape[0], 64)
        self.assertEqual(features_tensor_shape[1], 28)
        self.assertEqual(features_tensor_shape[2], 28)

    def test_targets_dimensions(self):
        targets_tensor_shape = TestPositiveUnlabeledMNIST.samples[1].shape
        self.assertEqual(len(targets_tensor_shape), 1)
        self.assertEqual(targets_tensor_shape[0], 64)

    def test_num_each_labels(self):
        targets = TestPositiveUnlabeledMNIST.dataset.targets
        num_positive_label = torch.sum(torch.where(targets < 0, 0, 1)).item()
        num_unlabeled_label = torch.sum(torch.where(targets < 0, 1, 0)).item()
        self.assertEqual(num_positive_label, 1000)
        self.assertEqual(num_unlabeled_label, 59000)


if __name__ == '__main__':
    unittest.main()
