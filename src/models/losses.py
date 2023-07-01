from typing import Tuple

import torch
import torch.nn as nn


class PositiveUnlabeledBaseLoss(object):
    """ TODO: Add comments. """

    def __init__(
            self,
            prior: float = 0.4915,
            beta: float = 0.0,
            gamma: float = 1.0,
            loss_fn: str = 'sigmoid',
            is_nnpu: bool = True) -> None:
        """
        TODO: Add comments.
        :param prior:
        :param beta:
        :param gamma:
        :
        :param is_nnpu: True -> use Non-Negative Risk Estimator
        """
        # super(PositiveUnlabeledElbo, self).__init__()
        super(PositiveUnlabeledBaseLoss, self).__init__()

        # Error Handling
        if prior < 0 or prior > 1:
            raise ValueError(f"class-prior expects 0 <= prior <= 1, but got {prior}")
        if beta < 0 or beta > prior:
            raise ValueError(f"beta is expects 0 <= beta <= {prior}, but got {beta}")
        if gamma < 0 or gamma > 1:
            raise ValueError(f"gamma expects 0 <= gamma <= 1, but got {gamma}")

        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.is_nnpu = is_nnpu

        if loss_fn == "sigmoid":
            self.loss_fn = lambda x: torch.sigmoid(-x)
        elif loss_fn == "logistic":
            b = torch.tensor(1)
            self.loss_fn = lambda x: (1 / b) * torch.log(torch.tensor(1) + torch.exp(-x * b))
        else:
            raise NotImplementedError(f"loss function expects `sigmoid` or `logistic`, but got loss_fn={loss_fn}")

    def loss_and_risk_estimator_(self, inputs: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Add comments.
        :param inputs:
        :param target:
        :return:
        """
        positive = (target == 1).type(torch.float)
        unlabeled = (target == -1).type(torch.float)

        num_positive = torch.max(torch.tensor([1]), torch.sum(positive))
        num_unlabeled = torch.max(torch.tensor([1]), torch.sum(unlabeled))
        if num_positive.item() <= 0 or num_unlabeled.item() <= 0:
            raise ValueError("n_positive expects a positive integer, but got negative value")

        positive_risk = torch.sum((self.prior * positive / num_positive) * self.loss_fn(inputs))
        negative_risk = torch.sum(
            ((unlabeled / num_unlabeled) - (self.prior * positive / num_positive)) * self.loss_fn(-inputs)
        )

        loss = risk = positive_risk + negative_risk
        if (negative_risk < -self.beta) and self.is_nnpu:
            loss = -torch.tensor(self.gamma) * negative_risk
            risk = positive_risk - self.beta

        return loss, risk


class PositiveUnlabeledLoss(nn.Module):
    """ TODO: Add comments. """

    def __init__(
            self,
            prior: float = 0.4915,
            beta: float = 0.0,
            gamma: float = 1.0,
            loss_fn: str = 'sigmoid',
            is_nnpu: bool = True) -> None:
        """
        TODO: Add comments.
        :param prior:
        :param beta:
        :param gamma:
        :param is_nnpu: True -> use Non-Negative Risk Estimator
        """
        super(PositiveUnlabeledLoss, self).__init__()

        self.base_loss = PositiveUnlabeledBaseLoss(
            prior=prior,
            beta=beta,
            gamma=gamma,
            loss_fn=loss_fn,
            is_nnpu=is_nnpu
        )

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: output
        R^(+)_p = (1/n_p) * sum(sigmoid(x)) -> self.prior * (positive / n_positive) * torch.sum(self.loss(output))
        R^(-)_p = (1/n_p) * sum(sigmoid(-x)) -> self.prior * (positive / n_positive) * torch.sum(self.loss(-1 * output))
        R^(-)_u = (1/n_u) * sum(sigmoid(-x)) -> unlabeled / n_unlabeled * torch.sum(self.loss(-1 *output))
        :param inputs:
        :param target:
        :return loss, risk:
        """

        loss, risk = self.base_loss.loss_and_risk_estimator_(inputs=inputs, target=target)

        return loss, risk
