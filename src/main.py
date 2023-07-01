import glob
import logging
import os

import boto3
import torch

from src.trainer import execute_exp
from src.utils import config
from src.utils import set_logger


def main():
    # Load Config File
    filename = "./config/config.toml"
    args = config.ConfigurationParser(filename)

    # TODO: I will be fixed confusion on how to get exp_id.
    exp_id = config.ConfigurationParser.get_exp_id(model_save_dir=args.model_save_dir) + 1

    # Set Logger
    set_logger.training_logger(exp_id=exp_id)
    logger = logging.getLogger(__name__)

    # Confirm Current Working Directory
    logger.info(
        {
            "action": "confirm working directory",
            "status": "complete",
            "arguments": {
                "current working directory": os.path.abspath(".")
            }
        }
    )

    # Confirm CUDA Availability
    logger.info(
        {
            "action": "confirm CUDA availability",
            "status": "complete",
            "arguments": {
                "available": torch.cuda.is_available(),
            }
        }
    )

    # Execute Experiment
    train_history_dfs, valid_history_dfs = execute_exp.repeated_trials(args, iteration=1)

    # Viewing Experimental Results
    logger.info(
        {
            "action": "print out results",
            "status": "complete",
            "arguments": {
                "train results: loss={:8.6f} risk={:8.6f} accuracy={:8.6f}".format(
                    train_history_dfs.train_loss.values.mean(),
                    train_history_dfs.train_risk.values.mean(),
                    train_history_dfs.train_acc.values.mean(),
                )
            }
        }
    )
    logger.info(
        {
            "action": "print out results",
            "status": "complete",
            "arguments": {
                "eval results: loss={:8.6f} risk={:8.6f} accuracy={:8.6f}".format(
                    valid_history_dfs.valid_loss.values.mean(),
                    valid_history_dfs.valid_risk.values.mean(),
                    valid_history_dfs.valid_acc.values.mean(),
                )
            }
        }
    )

    try:
        client = boto3.client("s3")

        bucket_obj = client.list_objects(Bucket=args.bucket_name, Prefix=args.trainer_name)
        bucket_dir = sorted(set([content["Key"].split("/")[0] for content in bucket_obj["Contents"]]))[-1]
        model_files = glob.glob(args.model_save_dir + "/*/*.pt")
        for model_file in model_files:
            client.upload_file(
                Filename=model_file,
                Bucket=args.bucket_name,
                Key=os.path.join(os.path.join(bucket_dir, "models"), model_file.split("/")[-1])
            )

        logger.info(
            {
                "action": "print out results",
                "status": "complete",
                "arguments": {
                    "upload to S3": "success"
                }
            }
        )
    except Exception as e:
        logger.info(
            {
                "action": "print out results",
                "status": "complete",
                "arguments": {
                    "upload to S3": "error: {}".format(e)
                }
            }
        )


if __name__ == "__main__":
    main()
