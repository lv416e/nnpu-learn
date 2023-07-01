import unittest

import torch

from models.losses import PositiveUnlabeledLoss


class TestPositiveUnlabeledLoss(unittest.TestCase):
    """ Unit test code: loss function"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = torch.tensor([1, -1, -1, 1, -1])
        cls.beta = 0.0
        cls.gamma = 1.0
        cls.y_positive_expects = torch.tensor([0.7311, 0.2689, 0.2689, 0.7311, 0.2689])
        cls.y_unlabeled_expects = torch.tensor([0.2689, 0.7311, 0.7311, 0.2689, 0.7311])
        cls.PositiveUnlabeledLoss = PositiveUnlabeledLoss(prior=0.4915, beta=0.0, gamma=1.0)

    def test_positive_risk(self):
        targets = torch.tensor([1, -1, 1, -1, -1])
        positive = (targets == 1).type(torch.float)
        unlabeled = (targets == -1).type(torch.float)
        num_positive = torch.max(torch.tensor(1), torch.sum(positive))
        num_unlabeled = torch.max(torch.tensor(1), torch.sum(unlabeled))

        positive_risk_expects = \
            torch.sum((0.4915 * positive / num_positive) * TestPositiveUnlabeledLoss.y_positive_expects)
        negative_risk_expects = torch.sum(
            ((unlabeled / num_unlabeled) - (0.4915 * positive / num_positive))
            * TestPositiveUnlabeledLoss.y_unlabeled_expects
        )

        loss_expects = positive_risk_expects + negative_risk_expects
        risk_expects = positive_risk_expects + negative_risk_expects
        loss_actual, risk_actual = \
            TestPositiveUnlabeledLoss.PositiveUnlabeledLoss(TestPositiveUnlabeledLoss.inputs, targets)

        self.assertAlmostEqual(loss_actual, loss_expects, delta=1e-4)
        self.assertAlmostEqual(risk_actual, risk_expects, delta=1e-4)

    def test_negative_risk(self):
        targets = torch.tensor([1, 1, 1, 1, 1])
        positive = (targets == 1).type(torch.float)
        unlabeled = (targets == -1).type(torch.float)
        num_positive = torch.max(torch.tensor(1), torch.sum(positive))
        num_unlabeled = torch.max(torch.tensor(1), torch.sum(unlabeled))

        positive_risk_expects = \
            torch.sum((0.4915 * positive / num_positive) * TestPositiveUnlabeledLoss.y_positive_expects)
        negative_risk_expects = torch.sum(
            ((unlabeled / num_unlabeled) - (0.4915 * positive / num_positive))
            * TestPositiveUnlabeledLoss.y_unlabeled_expects
        )

        loss_expects = -TestPositiveUnlabeledLoss.gamma * negative_risk_expects
        risk_expects = positive_risk_expects - TestPositiveUnlabeledLoss.beta
        loss_actual, risk_actual = \
            TestPositiveUnlabeledLoss.PositiveUnlabeledLoss(TestPositiveUnlabeledLoss.inputs, targets)

        self.assertAlmostEqual(loss_actual, loss_expects, delta=1e-4)
        self.assertAlmostEqual(risk_actual, risk_expects, delta=1e-4)


if __name__ == '__main__':
    unittest.main()
