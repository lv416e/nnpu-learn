from typing import Any
from typing import Tuple

from torch import optim

from src.models import losses
from src.models import models


def prepare_mlp_component(args: Any) -> Tuple[Any, Any, Any]:
    """
    TODO: Add comments.
    :param args:
    :return:
    """
    model = models.PositiveUnlabeledModel(
        in_features=args.in_features,
        hide_features=args.hide_features,
        out_features=args.out_features
    )

    criterion = losses.PositiveUnlabeledLoss(
        prior=args.prior,
        loss_fn=args.loss_fn,
        is_nnpu=args.is_nnpu
    )

    optimizer = get_optimizer(args, model)

    return model, criterion, optimizer


def get_optimizer(args: Any, model: Any):
    """
    TODO: Add comments.
    :param args:
    :param model:
    :return:
    """
    # NOTE: the following code is required python3.10++
    match args.optimizer:
        case "adam":
            return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        case "sgd":
            return optim.SGD(model.parameters(), lr=args.lr)
        case _:
            raise NotImplementedError
