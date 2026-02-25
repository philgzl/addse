from typing import override

import lightning as L
import torch
from lightning.pytorch.callbacks import Timer, WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn


class TimerCallback(Timer):
    """A callback that logs the total training time at the end of training."""

    @override
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_train_end(trainer, pl_module)
        for logger in trainer.loggers:
            logger.log_metrics({"training_time": self.time_elapsed()})


class GPUMemoryCallback(L.Callback):
    """A callback that logs the peak GPU memory usage at the end of training."""

    @override
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.strategy.root_device.type != "cuda":
            return
        for logger in trainer.loggers:
            logger.log_metrics(
                {
                    "max_memory_allocated": torch.cuda.max_memory_allocated(trainer.strategy.root_device),
                    "max_memory_reserved": torch.cuda.max_memory_reserved(trainer.strategy.root_device),
                },
            )


class EMAWeightAveraging(WeightAveraging):
    """A callback for exponential moving average of weights."""

    def __init__(self, decay: float = 0.999) -> None:
        """Initialize the EMA weight averaging callback."""
        super().__init__(avg_fn=get_ema_avg_fn(decay))
