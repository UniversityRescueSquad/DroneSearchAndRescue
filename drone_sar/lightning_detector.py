import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.detection import MeanAveragePrecision
from transformers import DetrImageProcessor, DetrForObjectDetection


def get_lightning_trainer(model_name, max_epochs, device, **trainer_kwargs):
    logger = TensorBoardLogger("logs/", name=model_name)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    checkpoint_callback = ModelCheckpoint(
        dirpath=".checkpoints/",
        filename=model_name + "-{epoch}-{val_map_50:.3f}",
        save_top_k=3,
        monitor="val_map_50",
        mode="max",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stop, checkpoint_callback],
        gradient_clip_val=0.1,
        accelerator=device,
        **trainer_kwargs,
    )

    return trainer


class LightningDetector(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        self.lr = lr
        self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        batch, info = input
        return self.model(**batch)

    def training_step(self, input, batch_index):
        batch, info = input
        output = self.model(**batch)
        bs = len(batch["labels"])
        self.log("train_loss", output["loss"], batch_size=bs)
        for lk, lv in output["loss_dict"].items():
            self.log(f"train_{lk}", lv, batch_size=bs)
        return output["loss"]

    def validation_step(self, input, batch_index):
        batch, info = input
        output = self.model(**batch)
        bs = len(batch["labels"])
        self.log("val_loss", output["loss"], batch_size=bs)
        for lk, lv in output["loss_dict"].items():
            self.log(f"val_{lk}", lv, batch_size=bs)

        sizes_hw = [(h, w) for w, h in info["target_sizes"]]
        preds = self.processor.post_process_object_detection(
            output, threshold=0.05, target_sizes=sizes_hw
        )
        targets = []
        for label, (h, w) in zip(batch["labels"], sizes_hw):
            boxes = label["boxes"]
            if len(boxes):
                scale = boxes.new_tensor([w, h, w, h])
                cx, cy, bw, bh = (boxes * scale).unbind(-1)
                xyxy = torch.stack(
                    [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1
                )
            else:
                xyxy = boxes.reshape(0, 4)
            targets.append({"boxes": xyxy, "labels": label["class_labels"]})
        self.val_map.update(preds, targets)
        return output["loss"]

    def on_validation_epoch_end(self):
        metrics = self.val_map.compute()
        self.log("val_map", metrics["map"])
        self.log("val_map_50", metrics["map_50"])
        self.val_map.reset()

    def predict(self, pil, threshold=0.5):
        dos = {
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "do_pad": False,
        }
        prep_input = self.processor(images=pil, return_tensors="pt", **dos)
        prep_input = {k: v.to(self.device) for k, v in prep_input.items()}
        with torch.inference_mode():
            raw_out = self.model(**prep_input)

        W, H = pil.size
        result = self.processor.post_process_object_detection(
            raw_out, threshold=threshold, target_sizes=[(H, W)]
        )[0]
        boxes = result["boxes"].cpu().numpy()
        # xyxy to [xmin, ymin, w, h] as expected by vis.draw_prediction
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
        return boxes
