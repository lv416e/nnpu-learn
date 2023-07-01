import glob
import os
from datetime import datetime
from typing import List
from typing import Optional

import torch
import torch.nn as nn

from src.utils import config


class PositiveUnlabeledBaseModel(object):
    """ TODO: Add comments. """

    def __init__(
            self,
            in_features: int,
            hide_features: int,
            out_features: int,
            bias: bool = False,
            num_hidden_layers: int = 3,
            layers: Optional[List[nn.Module]] = None,
            is_flatten: bool = False) -> None:

        self.in_features = in_features
        self.hide_features = hide_features
        self.out_features = out_features
        self.bias = bias
        self.num_hidden_layers = num_hidden_layers
        self.is_flatten = is_flatten

        if layers is None:
            self.layers = []

    def build_mlp_model(self) -> nn.Module:
        """
        TODO: Add comments.
        :return:
        """
        if self.is_flatten:
            self.layers.append(nn.Flatten())

        self.layers.append(nn.Linear(self.in_features, self.hide_features, bias=self.bias))
        for _ in range(self.num_hidden_layers):
            self.layers += [
                nn.Linear(self.hide_features, self.hide_features, bias=self.bias),
                nn.BatchNorm1d(num_features=self.hide_features),
                nn.LeakyReLU()
            ]
        self.layers.append(nn.Linear(self.hide_features, self.out_features, bias=self.bias))

        return nn.Sequential(*self.layers)

    @staticmethod
    def model_save(model: nn.Module, exp_id: Optional[int] = None, model_save_dir: str = "./outputs/models") -> None:
        """
        TODO: Add comments.
        :param model:
        :param exp_id:
        :param model_save_dir:
        :return:
        """
        if exp_id is None:
            exp_id = config.ConfigurationParser.get_exp_id(model_save_dir=model_save_dir)

        now = datetime.utcnow()
        output_dir = os.path.join(model_save_dir, f"exp-id_{exp_id:03d}")
        model_name = f"trained-{now.year}-{now.month}-{now.day}-{now.minute}-{now.microsecond}.pt"
        torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}"))

    @staticmethod
    def model_load(
            model: nn.Module,
            exp_id: Optional[int] = None,
            model_save_dir: str = "./outputs/models") -> nn.Module:
        """
        TODO: Add comments.
        :param model:
        :param model_save_dir:
        :param exp_id:
        :return:
        """
        if exp_id is None:
            exp_id = config.ConfigurationParser.get_exp_id(model_save_dir=model_save_dir)

        output_dir = os.path.join(model_save_dir, f"exp-id_{exp_id:03d}")
        weights = sorted(glob.glob(os.path.join(output_dir, "*.pt")))
        if not weights:
            raise Exception("Weight Files Not Found.")
        weight = weights[-1]
        model.load_state_dict(torch.load(weight))

        return model


class PositiveUnlabeledModel(nn.Module):
    """ TODO: Add comments. """

    def __init__(
            self,
            in_features: int = 784,
            hide_features: int = 512,
            out_features: int = 1) -> None:
        """
        TODO: Add comments.
        :param in_features:
        :param hide_features:
        :param out_features:
        """
        super(PositiveUnlabeledModel, self).__init__()
        self.in_features = in_features
        self.hide_features = hide_features
        self.out_features = out_features
        self.base_model = PositiveUnlabeledBaseModel(
            in_features=in_features,
            hide_features=hide_features,
            out_features=out_features,
            bias=False,
            num_hidden_layers=3,
            is_flatten=True
        )
        self.model = self.base_model.build_mlp_model()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        TODO: Add comments.
        :param inputs:
        :return output:
        """
        outputs = self.model(inputs)
        return outputs

    def save(self, exp_id: Optional[int] = None, model_save_dir: str = "./outputs/models") -> None:
        """
        TODO: Add comments.
        :param exp_id:
        :param model_save_dir:
        :return:
        """
        self.base_model.model_save(self.model, exp_id=exp_id, model_save_dir=model_save_dir)

    def load(self, exp_id: Optional[int] = None, model_save_dir: str = "./outputs/models") -> None:
        """
        TODO: Add comments.
        :param exp_id:
        :param model_save_dir:
        :return:
        """
        self.model = self.base_model.model_load(self.model, exp_id=exp_id, model_save_dir=model_save_dir)
