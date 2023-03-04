import time
import numpy as np
import torch
import torch.utils.data as data
import logging
from tqdm import tqdm
from torch.optim import AdamW
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import EarlyStoppingCallback
from transformers.models.detr.modeling_detr import DetrLoss, DetrHungarianMatcher


class ObjectDetector:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.to(self.device)
        self.model.eval()
        logging.info("Model loaded successfully")

    def detect(self, frame: np.ndarray, confidence_threshold: float) -> dict:
        # Preprocess
        inputs = self._preprocess(frame)

        # Run the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Postprocess the outputs
        outputs = self._postprocess(frame, outputs, confidence_threshold)

        return {
            "scores": outputs["scores"],
            "labels": outputs["labels"],
            "boxes": outputs["boxes"],
        }

    def fine_tune(
        self,
        train_data: data.Dataset,
        val_data: data.Dataset,
        num_epochs: int = 5,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        save_model_path: str = "object_detection_model.pth",
    ) -> dict:
        """
        Fine-tunes the DETR model on the given train and validation datasets.

        Args:
            train_data (torch.utils.data.Dataset): The training dataset.
            val_data (torch.utils.data.Dataset): The validation dataset.
            num_epochs (int, optional): The number of epochs to train for. Defaults to 5.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-5.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.01.
            save_model_path (str, optional): The path to save the fine-tuned model. Defaults to "object_detection_model.pth".

        Returns:
            dict: A dictionary containing the final validation loss and validation accuracy.
        """
        logging.info("Start finetuning model.")

        # Create data loaders
        train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = data.DataLoader(val_data, batch_size=1, shuffle=False)

        # Create optimizer and loss function
        optimizer = AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        matcher = DetrHungarianMatcher()
        loss = DetrLoss(
            num_classes=self.model.config.num_labels,
            matcher=matcher,
            eos_coef=0.1,
            losses=["boxes"],
        )

        # Evaluate model before fine tuning on the validation set
        val_loss, val_acc = self._evaluate(val_loader, loss, matcher)
        logging.info(
            f"Evaluating model before finetuning: Validation loss: {val_loss:.4f}. Validation Accuracy: {val_acc:.4f}."
        )

        # Train the model
        start_time = time.time()
        for epoch in range(num_epochs):
            # Train for one epoch
            logging.info(f"Starting epoch {epoch + 1}")
            train_loss = self._train_one_epoch(train_loader, optimizer, loss, epoch)

            # Evaluate on the validation set
            val_loss, val_acc = self._evaluate(val_loader, loss, matcher)

            logging.info(
                f"Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f}. Validation Loss: {val_loss:.4f}. Validation Accuracy: {val_acc:.4f}."
            )

        end_time = time.time()
        total_training_time = end_time - start_time

        # Save the fine-tuned model
        self.model.save_pretrained(save_model_path)

        logging.info(
            f"Finished finetuning model. Total training time: {total_training_time:.2f} seconds. Final Validation Loss: {val_loss:.4f}. Final Validation Accuracy: {val_acc:.4f}."
        )

        # Return the final validation loss and validation accuracy
        return {"loss": val_loss, "accuracy": val_acc}

    def _train_one_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        loss: DetrLoss,
        epoch: int,
    ) -> float:
        self.model.train()
        train_loss = 0.0

        for images, targets in tqdm(data_loader, desc=f"Train epoch {epoch + 1}"):
            inputs = self._preprocess(images[0])
            output = self.model(**inputs)
            loss_dict = loss(output, [targets])
            losses = sum(loss_dict.values())
            train_loss += losses
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        train_loss /= len(data_loader)
        logging.info(f"Epoch: [{epoch + 1}] Average training loss: {train_loss:.4f}")
        return train_loss

    def _evaluate(
        self,
        data_loader: data.DataLoader,
        loss: DetrLoss,
        matcher: DetrHungarianMatcher,
    ) -> dict:
        """Evaluate the model on the given dataset and return the average loss and accuracy."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_predictions = 1

        with torch.no_grad():
            for image, targets in tqdm(data_loader, desc="Evaluation"):
                inputs = self._preprocess(image[0])
                outputs = self.model(**inputs)
                loss_dict = loss(outputs, [targets])
                total_loss += loss_dict["loss_bbox"].item()

                # compute accuracy
                outputs = matcher(outputs, [targets])
                num_correct = sum([len(p) for p in outputs])
                total_correct += num_correct
                total_predictions += sum([len(t["class_labels"]) for t in [targets]])

        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_predictions
        return avg_loss, accuracy

    def _preprocess(self, frame: np.ndarray) -> dict:
        inputs = self.processor(frame, return_tensors="pt")
        return inputs

    def _postprocess(
        self, frame: np.ndarray, outputs: dict, confidence_threshold: float
    ) -> dict:
        # convert outputs (bounding boxes and class logits)
        target_sizes = [([frame.shape[0], frame.shape[1]])]
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]

        results["boxes"] = [
            [round(i, 2) for i in box.tolist()] for box in results["boxes"]
        ]
        results["labels"] = [
            self.model.config.id2label[label.item()] for label in results["labels"]
        ]
        return results
