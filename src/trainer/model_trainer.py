import logging
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.models import datasets
from src.models import losses
from src.models import models
from src.utils import config
from src.utils import set_logger


class ModelTrainer(object):
    """ TODO: Add comments. """

    def __init__(
            self,
            exp_id: Optional[int] = None,
            model_save_dir: str = "./outputs/models",
            lr: float = 0.01,
            epochs: int = 10,
            model: Any = None,
            criterion: Any = None,
            optimizer: Any = None) -> None:
        """
        TODO: Add comments.
        :param exp_id:
        :param model_save_dir:
        :param lr:
        :param epochs:
        :param model:
        :param criterion:
        :param optimizer:
        """
        if exp_id is None:
            self.exp_id = config.ConfigurationParser.get_exp_id(model_save_dir=model_save_dir) + 1

        set_logger.training_logger(self.exp_id)
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(os.path.join(model_save_dir, f"exp-id_{self.exp_id:03d}")):
            self.logger.info({"action": "Make Directories", "status": "start"})
            os.makedirs(os.path.join(model_save_dir, f"exp-id_{self.exp_id:03d}"), exist_ok=False)
            self.logger.info({"action": "Make Directories", "status": "complete"})

        if model is None:
            model = models.PositiveUnlabeledModel(in_features=784, hide_features=512, out_features=1)
        if criterion is None:
            criterion = losses.PositiveUnlabeledLoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

        self.model_save_dir = model_save_dir
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.loss = self.risk = None
        self.losses_history = self.risks_history = self.acc_history = None
        self.eval_losses_history = self.eval_risks_history = self.eval_acc_history = None

    def train(
            self,
            dataloader: DataLoader,
            valid_dataloader: DataLoader,
            verbose: int = 200,
            is_save: bool = False,
            is_eval: bool = True,
            eval_each_epoch: int = 5
    ) -> Union[Tuple[Any, Tuple[list, list, list], Optional[Tuple[list, list, list]]]]:
        """
        TODO: Add comments.
        :param dataloader:
        :param valid_dataloader:
        :param verbose:
        :param is_save:
        :param is_eval:
        :param eval_each_epoch:
        :return:
        """
        self.logger.info({"action": "train", "status": "start"})
        for epoch in range(self.epochs):
            self.train_on_epoch(epoch, dataloader, verbose, is_save)
            if is_eval:
                if not isinstance(eval_each_epoch, int) or eval_each_epoch <= 0:
                    # TODO: It will be modified error handling in the future
                    self.logger.error({"action": "evaluation", "status": "failed"})
                    raise ValueError(f"eval_each_epoch expects positive integer, but got {eval_each_epoch}")
                if epoch % eval_each_epoch == 0 and epoch != 0:
                    self.evaluate_on_epoch(dataloader=valid_dataloader, verbose=verbose)

        self.evaluate(dataloader=valid_dataloader, load_model=False, verbose=verbose)
        self.logger.info({"action": "train", "status": "complete"})

        self.logger.info({"action": "model save", "status": "start"})
        self.model.save(exp_id=self.exp_id, model_save_dir=self.model_save_dir)
        self.logger.info({"action": "model save", "status": "complete"})

        training_history = (self.losses_history, self.risks_history, self.acc_history)
        if not is_eval:
            return self.model, training_history, None
        evaluate_history = (self.eval_losses_history, self.eval_risks_history, self.eval_acc_history)
        return self.model, training_history, evaluate_history

    def train_on_epoch(
            self,
            epoch_idx: int,
            dataloader: DataLoader = None,
            verbose: int = 200,
            is_save: bool = False) -> Tuple[Any, List[float], List[float], List[float]]:
        """
        TODO: Add comments.
        :param epoch_idx:
        :param dataloader:
        :param verbose:
        :param is_save:
        :return model, losses_history, risks_history:
        """
        self.logger.info({"action": "train on epoch", "status": "start"})
        train_losses = train_risks = train_correct = 0.0

        if self.losses_history is None:
            self.losses_history = []
        if self.risks_history is None:
            self.risks_history = []
        if self.acc_history is None:
            self.acc_history = []

        self.model.train()
        for idx, (inputs, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()

            output = self.model(inputs).view_as(targets)

            self.loss, self.risk = self.criterion(output, targets.float())
            self.loss.backward()
            self.optimizer.step()

            train_losses += self.loss.item()
            train_risks += self.risk.item()
            train_correct += torch.eq(torch.where(output < 0, -1, 1), targets).type(torch.float).mean().item()

            if (idx % verbose == verbose - 1) and idx != 0:
                train_losses = train_losses / verbose
                train_risks = train_risks / verbose
                train_correct = train_correct / verbose

                self.losses_history.append(train_losses)
                self.risks_history.append(train_risks)
                self.acc_history.append(train_correct)

                self.logger.info({
                    "action": "train on epoch",
                    "status": "running",
                    "arguments": {
                        "epoch: {:3d}/{:3d}, iter: {:4d}, losses: {:8.6f}, risks: {:8.6f}, accuracy: {:8.6f}".format(
                            epoch_idx + 1, self.epochs, idx + 1, train_losses, train_risks, train_correct
                        )
                    }
                })

                train_losses = train_risks = train_correct = 0.0

        if is_save:
            self.logger.info({"action": "model save", "status": "start"})
            self.model.save(exp_id=self.exp_id, model_save_dir=self.model_save_dir)
            self.logger.info({"action": "model save", "status": "complete"})

        self.logger.info({"action": "train on epoch", "status": "complete"})
        return self.model, self.losses_history, self.risks_history, self.acc_history

    def evaluate(
            self,
            dataloader: DataLoader,
            load_model: bool = True,
            verbose: int = 200
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        TODO: Add comments.
        :param dataloader:
        :param load_model:
        :param verbose:
        :return None:
        """
        self.logger.info({"action": "evaluation", "status": "start"})
        if load_model:
            self.logger.info({"action": "Model Loading ...", "status": "start"})
            self.model.load(exp_id=self.exp_id, model_save_dir=self.model_save_dir)
            self.logger.info({"action": "Model Loading ...", "status": "complete"})

        test_loss = test_risk = test_correct = 0.0

        if self.eval_losses_history is None:
            self.eval_losses_history = []
        if self.eval_risks_history is None:
            self.eval_risks_history = []
        if self.eval_acc_history is None:
            self.eval_acc_history = []

        self.model.eval()
        with torch.no_grad():
            for idx, (feature, targets) in enumerate(dataloader):
                output = self.model(feature).view_as(targets)
                loss, risk = self.criterion(output, targets.float())

                test_loss += loss.item()
                test_risk += risk.item()
                test_correct += torch.eq(torch.where(output < 0, -1, 1), targets).type(torch.float).mean().item()

                if (idx % verbose == verbose - 1) and idx != 0:
                    _test_loss = test_loss / (idx + 1)
                    _test_risk = test_risk / (idx + 1)
                    _test_correct = test_correct / (idx + 1)

                    self.eval_losses_history.append(_test_loss)
                    self.eval_risks_history.append(_test_risk)
                    self.eval_acc_history.append(_test_correct)

                    self.logger.info({
                        "action": "evaluation",
                        "status": "running",
                        "arguments": {
                            "iter: {:4d}, test_loss: {:8.6f}, test_risk: {:8.6f}, accuracy: {:8.6f}".format(
                                idx + 1, _test_loss, _test_risk, _test_correct
                            )
                        }
                    })
                    _test_loss = _test_risk = _test_correct = 0.0

        num_batches = len(dataloader)
        test_loss /= (num_batches + 1)
        test_risk /= (num_batches + 1)
        test_correct /= (num_batches + 1)
        self.logger.info({
            "action": "evaluation",
            "status": "running",
            "arguments": {
                "iter: comp, test_loss: {:8.6f}, test_risk: {:8.6f}, accuracy: {:8.6f}".format(
                    test_loss, test_risk, test_correct
                )
            }
        })
        self.logger.info({"action": "evaluation", "status": "complete"})
        return self.eval_losses_history, self.eval_risks_history, self.eval_acc_history

    def evaluate_on_epoch(self, dataloader: DataLoader = None, verbose: int = 200) -> None:
        """
        TODO: Add comments.
        :return None:
        """
        # TODO: The following code is hard-coding for MNIST, so it will be fixed in the future.
        self.logger.info({"action": "evaluate on training", "status": "start"})

        # TODO: The following code is hard-coding, it will be fixed in the future
        if dataloader is None:
            self.logger.info({"action": "create DataLoader", "status": "start"})
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            eval_dataset = datasets.PositiveUnlabeledMNIST(transform=transform, train=False)
            dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=True)
            self.logger.info({"action": "create DataLoader", "status": "complete"})

        self.evaluate(dataloader, load_model=False, verbose=verbose)

        self.logger.info({"action": "evaluate on training", "status": "complete"})
