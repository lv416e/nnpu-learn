import os

import toml


class ConfigurationParser(object):
    """ TODO: Add comments. """

    def __init__(self, config_path: str = "./config/config.toml") -> None:
        """
        TODO: Add comments.
        :param config_path:
        """
        with open(config_path) as f:
            config = toml.load(f)

        # training configure
        self.__epochs = config["training"]["epochs"]
        self.__train_batch_size = config["training"]["train_batch_size"]
        self.__valid_batch_size = config["training"]["valid_batch_size"]
        self.__verbose = config["training"]["verbose"]
        self.__is_save = config["training"]["is_save"]
        self.__is_eval = config["training"]["is_eval"]
        self.__eval_each_epoch = config["training"]["eval_each_epoch"]
        self.__model_save_dir = config["training"]["model_save_dir"]

        # network configure
        self.__in_features = config["network"]["in_features"]
        self.__hide_features = config["network"]["hide_features"]
        self.__out_features = config["network"]["out_features"]

        # optimizer configure
        self.__loss_fn = config["optimizer"]["loss_fn"]
        self.__optimizer = config["optimizer"]["optimizer"]
        self.__lr = config["optimizer"]["lr"]
        self.__weight_decay = config["optimizer"]["weight_decay"]

        # loss configure
        self.__is_nnpu = config["loss"]["is_nnpu"]

        # dataset configure
        self.__prior = config["mnist-prior"]["prior"]

        # SageMaker training jobs configure
        self.__use_sagemaker_training = config["sagemaker"]["use_sagemaker_training"]

        try:
            self.__trainer_name = config["sagemaker"]["trainer_name"]
            if not self.__trainer_name:
                self.__trainer_name = os.environ["REPOSITORY_NAME"].split("/")[-1]
        except KeyError:
            self.__trainer_name = os.environ["REPOSITORY_NAME"].split("/")[-1]

        # S3 bucket configure
        try:
            self.__bucket_name = config["bucket"]["bucket_name"]
            if not self.__bucket_name:
                self.__bucket_name = os.environ["BUCKET_NAME"]
        except KeyError:
            self.__bucket_name = os.environ["BUCKET_NAME"]

    @staticmethod
    def get_exp_id(model_save_dir: str = "./outputs/models") -> int:
        try:
            dirs = sorted([_dir for _dir in os.listdir(model_save_dir) if _dir.startswith("exp-id")])
            exp_id = int(dirs[-1].split("_")[-1])
            return exp_id
        except (FileNotFoundError, IndexError):
            return 0

    @property
    def epochs(self):
        return self.__epochs

    @property
    def train_batch_size(self):
        return self.__train_batch_size

    @property
    def valid_batch_size(self):
        return self.__valid_batch_size

    @property
    def verbose(self):
        return self.__verbose

    @property
    def is_save(self):
        return self.__is_save

    @property
    def is_eval(self):
        return self.__is_eval

    @property
    def eval_each_epoch(self):
        return self.__eval_each_epoch

    @property
    def model_save_dir(self):
        return self.__model_save_dir

    @property
    def in_features(self):
        return self.__in_features

    @property
    def hide_features(self):
        return self.__hide_features

    @property
    def out_features(self):
        return self.__out_features

    @property
    def loss_fn(self):
        return self.__loss_fn

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def lr(self):
        return self.__lr

    @property
    def weight_decay(self):
        return self.__weight_decay

    @property
    def is_nnpu(self):
        return self.__is_nnpu

    @property
    def prior(self):
        return self.__prior

    @property
    def use_sagemaker_training(self):
        return self.__use_sagemaker_training
    
    @property
    def trainer_name(self):
        return self.__trainer_name

    @property
    def bucket_name(self):
        return self.__bucket_name
