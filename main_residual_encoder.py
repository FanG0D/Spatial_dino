"""
Training script for Spatial Residual Encoder
Only learns spatial-aware residual features (no reconstruction)
"""

import argparse
import os
import sys
import datetime
import glob
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from ldm.util import instantiate_from_config


def get_parser(**parser_kwargs):
    """Argument parser for training."""
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n", "--name", type=str, const=True, default="", nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r", "--resume", type=str, const=True, default="", nargs="?",
        help="resume from logdir or checkpoint",
    )
    parser.add_argument(
        "-b", "--base", nargs="*", metavar="base_config.yaml",
        help="paths to base configs", default=list(),
    )
    parser.add_argument(
        "-t", "--train", type=str2bool, const=True, default=False, nargs="?",
        help="train",
    )
    parser.add_argument(
        "-d", "--debug", type=str2bool, nargs="?", const=True, default=False,
        help="enable debugging",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=23, help="seed for seed_everything",
    )
    parser.add_argument(
        "-f", "--postfix", type=str, default="", help="postfix for run name",
    )
    parser.add_argument(
        "-l", "--logdir", type=str, default="logs", help="directory for logging",
    )
    return parser


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset."""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    """Data module from config."""
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader

        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _val_dataloader(self, shuffle=False):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
    else:
        cfg_name = "spatial_residual"

    name = "_" + cfg_name
    logdir = os.path.join(opt.logdir, cfg_name)
    ckptdir = os.path.join(logdir, "checkpoints")

    seed_everything(opt.seed)
    print(f"Logdir: {logdir}")

    # Try to resume from latest checkpoint
    ckpt_files = glob.glob(os.path.join(logdir, "checkpoints", "epoch=*.ckpt"))
    if ckpt_files:
        ckpt_files.sort(key=lambda x: int(os.path.basename(x).split("=")[1].split(".")[0]))
        ckpt = ckpt_files[-1]
        print(f"Resuming from {ckpt}")
    else:
        ckpt = None

    # Load configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base] if opt.base else []
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # Setup trainer
    trainer_kwargs = dict()

    # Logger
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": "tensorboard",
            "save_dir": logdir,
        }
    }
    logger_cfg = OmegaConf.merge(default_logger_cfg, lightning_config.get("logger", OmegaConf.create()))
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # Model checkpoint
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
            "save_top_k": 3,
            "monitor": "val/loss",
            "mode": "min",
        }
    }
    callbacks_cfg = {
        "checkpoint_callback": default_modelckpt_cfg,
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {"logging_interval": "step"}
        },
    }
    callbacks_cfg = OmegaConf.merge(callbacks_cfg, lightning_config.get("callbacks", OmegaConf.create()))
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # Create trainer
    trainer = pl.Trainer(**trainer_config, **trainer_kwargs)

    # Create model
    model = instantiate_from_config(config.model)

    # Create data module
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # Configure learning rate
    model.learning_rate = config.model.get("base_learning_rate", 1.0e-4)
    print(f"Setting learning rate to {model.learning_rate:.2e}")

    # Train
    if opt.train:
        try:
            trainer.fit(model, data, ckpt_path=ckpt)
        except Exception as e:
            print(f"Error during training: {e}")
            raise


if __name__ == "__main__":
    main()
