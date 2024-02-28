from transformers import DetrImageProcessor, DetrForObjectDetection
import lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor
import lightning as L
import torch


def get_lightning_trainer(model_name, max_epochs):
    # optim_metric = "metric_val_mean_F1"
    # optim_metric_mode = "max"
    optim_metric = "val_loss"
    optim_metric_mode = "min"

    logger = TensorBoardLogger("tb_logs2", name=model_name)
    early_stop = EarlyStopping(
        monitor=optim_metric, mode=optim_metric_mode, patience=50
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=".checkpoints/",
        save_top_k=5,
        monitor=optim_metric,
        mode=optim_metric_mode,
    )
    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    device_monitor = DeviceStatsMonitor(cpu_stats=False)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stop, checkpoint_callback, device_monitor],
        profiler=profiler,
    )

    return trainer


class LightningDetector(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.rocessor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.mode = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1,
            ignore_mismatched_sizes=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def optim_step(self, input, flavour):
        batch, info = input
        num_boxes = sum([len(l["boxes"]) for l in batch["labels"]])
        if num_boxes == 0:
            return None  # Skip the optim step if no boxes present (we get an error otherwise)

        output = self.model(**batch)
        self.log(f"{flavour}_loss", output["loss"])
        for lk, lv in output["loss_dict"].items():
            self.lof(f"{flavour}_{lk}", lv)
        return output["loss"]

    def training_step(self, input, batch_index):
        loss = self.optim_step(input, flavour="train")
        return loss

    def validation_step(self, input, batch_index):
        loss = self.optim_step(input, flavour="val")
        return loss
