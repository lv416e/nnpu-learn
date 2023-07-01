import logging
import os
from datetime import datetime
from typing import Optional

from src.utils import config


def training_logger(exp_id: Optional[int]) -> None:
    """
    TODO: Add comments.
    :param exp_id:
    :return:a
    """
    if exp_id is None:
        exp_id = config.ConfigurationParser.get_exp_id(model_save_dir="./outputs/log") + 1

    logger_format = "%(asctime)s@ %(name)-30s [%(levelname)s] %(funcName)-20s: %(message)s"

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(logger_format))

    now = datetime.utcnow()
    log_dir = "./outputs/log"

    file_path = os.path.join(log_dir, f"exp-id_{exp_id:03d}")
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=False)

    file_name = f"logging-{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}.log"
    file_handler = logging.FileHandler(os.path.join(file_path, file_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(logger_format))

    logging.basicConfig(level=logging.NOTSET, handlers=[stream_handler, file_handler])
