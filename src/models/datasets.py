from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class PositiveUnlabeledMNIST(Dataset):
    """ TODO: Add comments. """

    def __init__(
            self,
            root: str = './data',
            train: bool = True,
            download: bool = True,
            transform: Optional[Callable] = None,
            is_elkanoto: bool = False,
            num_labeled: int = 1000,
            num_classes: int = 1) -> None:
        """
        TODO: Add comments.
        :param root:
        :param train:
        :param download:
        :param transform:
        :param is_elkanoto:
        :param num_labeled:
        :param num_classes:
        """
        super(PositiveUnlabeledMNIST, self).__init__()

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        self.transform = transform
        self.dataset = MNIST(root=root, train=train, transform=transform, download=download)
        self.feature = self.dataset.data.float()
        self.targets = self.dataset.targets.float()

        self.positive = np.where(self.targets % 2 == 0)[0]
        self.indices = np.random.permutation(len(self.positive))

        if train:
            self.positive = self.positive[self.indices[:num_labeled]]

        self.targets = -1 * torch.ones_like(self.targets).float()
        if is_elkanoto:
            self.targets = torch.zeros_like(self.targets).float()
        self.targets[self.positive] = torch.tensor(1).float()

        if num_classes > 1:
            ohe = OneHotEncoder(sparse=False, dtype=int)
            self.targets = self.targets.reshape(-1, 1)
            self.targets = ohe.fit_transform(self.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature = self.feature[idx]
        targets = self.targets[idx]

        if not torch.is_tensor(feature):
            feature = self.transform(feature)

        return feature, targets


def get_datasets_as_array(dataloader, x: np.ndarray = None, y: np.ndarray = None):
    for idx, data in enumerate(dataloader):
        feature, targets = data[0].numpy().reshape(-1, 28 * 28), data[1].numpy()
        if not idx:
            x = feature
            y = targets
            continue
        x = np.vstack((x, feature))
        y = np.hstack((y, targets))
    return x, y
