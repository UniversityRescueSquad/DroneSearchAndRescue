"""Fine-tune DETR on the unified SAR dataset.

Expects the dataset produced by notebooks/dataset_demo_and_harmonization.ipynb
and split by drone_sar/split_unified_dataset.py.

    python drone_sar/train.py --device cpu --smoke   # pipeline check
    python drone_sar/train.py --device cuda          # real training run
"""

import argparse
import os

from drone_sar.ipsar_dataset import IPSARDataset
from drone_sar.lightning_detector import LightningDetector, get_lightning_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/unified_dataset")
    parser.add_argument("--model-name", default="detr-sar-unified")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoke", action="store_true",
        help="run one epoch on two tiny batches to verify the pipeline",
    )
    args = parser.parse_args()

    model = LightningDetector(lr=args.lr)
    train_data = IPSARDataset(os.path.join(args.data_root, "train"))
    val_data = IPSARDataset(os.path.join(args.data_root, "val"))
    print(f"train: {len(train_data)} images, val: {len(val_data)} images")

    train_dl = train_data.get_dataloader(model.processor, bs=args.batch_size, shuffle=True)
    val_dl = val_data.get_dataloader(model.processor, bs=args.batch_size, shuffle=False)

    extra = {}
    epochs = args.epochs
    if args.smoke:
        epochs = 1
        extra = {"limit_train_batches": 2, "limit_val_batches": 2, "log_every_n_steps": 1}

    trainer = get_lightning_trainer(args.model_name, max_epochs=epochs, device=args.device, **extra)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
