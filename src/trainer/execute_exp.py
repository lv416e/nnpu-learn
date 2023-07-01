import logging
from typing import Any
from typing import Optional
from typing import Tuple

import pandas as pd
from torch.utils.data import DataLoader

from src.models import datasets
from src.trainer import get_models
from src.trainer import model_trainer
from src.utils import config
from src.utils import set_logger


def repeated_trials(args: Any, iteration: int = 10, exp_id: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TODO: Add comments.
    NOTE: The following code was cut out into this function because code duplication was pointed out.
    :param args:
    :param iteration:
    :param exp_id:
    :return:
    """
    if exp_id is None:
        exp_id = config.ConfigurationParser.get_exp_id(args.model_save_dir) + 1

    set_logger.training_logger(exp_id=exp_id)
    logger = logging.getLogger(__name__)
    logger.info({"action": "experiments", "status": "run"})

    train_dataset = datasets.PositiveUnlabeledMNIST(train=True)
    valid_dataset = datasets.PositiveUnlabeledMNIST(train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)

    train_history_dfs = pd.DataFrame([], columns=["iterations", "train_loss", "train_risk", "train_acc"])
    valid_history_dfs = pd.DataFrame([], columns=["iterations", "valid_loss", "valid_risk", "valid_acc"])

    for i in range(iteration):
        logger.info({"action": "experiment iters: {}".format(i + 1), "status": "start"})
        model, criterion, optimizer = get_models.prepare_mlp_component(args)
        trainer = model_trainer.ModelTrainer(
            epochs=args.epochs,
            model=model,
            criterion=criterion,
            optimizer=optimizer
        )
        trained_model, training_history, evaluate_history = trainer.train(
            dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            verbose=args.verbose,
            is_save=args.is_save,
            is_eval=args.is_eval,
            eval_each_epoch=args.eval_each_epoch
        )

        train_loss_history, train_risk_history, train_acc_history = training_history
        valid_loss_history, valid_risk_history, valid_acc_history = evaluate_history

        n = 1 if args.verbose == 1 else 0
        train_indices = [
            (idx + 1) * args.train_batch_size
            for idx in range((len(train_dataloader) // args.verbose - n) * args.epochs)
        ]
        valid_indices = [
            (idx + 1) * args.valid_batch_size
            for idx in range((len(valid_dataloader) // args.verbose - n) * (args.epochs // args.eval_each_epoch))
        ]

        train_history_df = pd.DataFrame({
            "iterations": train_indices,
            "train_loss": train_loss_history,
            "train_risk": train_risk_history,
            "train_acc": train_acc_history,
        })
        valid_history_df = pd.DataFrame({
            "iterations": valid_indices,
            "valid_loss": valid_loss_history,
            "valid_risk": valid_risk_history,
            "valid_acc": valid_acc_history
        })

        train_history_dfs = pd.concat([train_history_dfs, train_history_df], ignore_index=True, sort=False)
        valid_history_dfs = pd.concat([valid_history_dfs, valid_history_df], ignore_index=True, sort=False)

        logger.info({"action": "experiment iters: {}".format(i + 1), "status": "complete"})

    logger.info({"action": "experiments", "status": "complete"})

    return train_history_dfs, valid_history_dfs
