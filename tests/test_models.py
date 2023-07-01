import unittest

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.datasets import PositiveUnlabeledMNIST
from models.models import PositiveUnlabeledModel


class TestPositiveUnlabeledModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = PositiveUnlabeledModel(in_features=784, hide_features=300, out_features=1)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        datasets = PositiveUnlabeledMNIST(transform=transform)
        dataloader = DataLoader(datasets, batch_size=4, shuffle=False)
        cls.samples = next(iter(dataloader))

    def test_something(self):
        inputs = TestPositiveUnlabeledModel.samples[0]
        predict = TestPositiveUnlabeledModel.model(inputs)
        self.assertEqual(predict.shape, torch.Size([4, 1]))


if __name__ == '__main__':
    unittest.main()
