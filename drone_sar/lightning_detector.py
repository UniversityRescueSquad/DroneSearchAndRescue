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

    logger = TensorBoardLogger("logs/", name=model_name)
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
    # device_monitor = DeviceStatsMonitor(cpu_stats=False)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stop, checkpoint_callback],
        profiler=profiler,
        gradient_clip_val=0.1,
        accelerator="gpu",
    )

    return trainer


class LightningDetector(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, input):
        batch, info = input
        output = self.model(**batch)
        return output

    def optim_step(self, input, flavour):
        batch, info = input
        output = self.model(**batch)
        self.log(f"{flavour}_loss", output["loss"])
        for lk, lv in output["loss_dict"].items():
            self.log(f"{flavour}_{lk}", lv)
        return output["loss"]

    def training_step(self, input, batch_index):
        loss = self.optim_step(input, flavour="train")
        return loss

    def validation_step(self, input, batch_index):
        loss = self.optim_step(input, flavour="val")
        return loss

    def predict(self, dl, processor, T=0.5):
        trainer = get_lightning_trainer("PREDICTION", max_epochs=-1)
        dl_list = list(dl)
        outputs = trainer.predict(self, dl_list)
        post_proc_outs = [
            processor.post_process_object_detection(
                otput, threshold=T, target_sizes=[["target_sizes"][::-1]]
            )
            for output, item in zip(outputs, dl_list)
        ]
