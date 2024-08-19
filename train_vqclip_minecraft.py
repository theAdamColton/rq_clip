from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb
import torch
import datetime

torch.set_float32_matmul_precision("high")

# Workaround for 'too many open files' error
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

from vq_clip.embedding_dataset import MinecraftEmbeddingDataModule
from vq_clip.trainer import MinecraftVQCLIPTrainer


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    experiment = "all"

    nowname = f"{experiment}-{now}"
    wandb_logger = WandbLogger(project="vq-clip-minecraft", name=nowname)

    cli = LightningCLI(
        MinecraftVQCLIPTrainer,
        MinecraftEmbeddingDataModule,
        trainer_defaults={
            "callbacks": [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    save_last=True,
                    every_n_epochs=1,
                ),
            ],
            "default_root_dir": f"logs/{nowname}",
            "logger": {
                "class_path": "lightning.pytorch.loggers.WandbLogger",
                "init_args": {
                    "project": "vq-clip-minecraft",
                    "name": nowname
                }
            },
        },
        save_config_kwargs={
            "overwrite": True
        },
        
    )

    wandb.finish()


if __name__ in {"__console__", "__main__"}:
    main()
