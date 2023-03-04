import os
import torch
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from xml.etree import ElementTree
from object_detector import ObjectDetector
from utils import setup_logging


class IPSARDataset(Dataset):
    """
    A PyTorch dataset for the IPSAR dataset.

    Args:
        root_dir (str): The path to the root directory of the IPSAR dataset.
        split_type (str): The name of the dataset split ("train" or "test").
    """

    def __init__(self, root_dir: str, split_type: str):
        assert split_type in ["train", "test"], f"Invalid set name: {split_type}"

        self.images_path = os.path.join(root_dir, "heridal", split_type + "Images")
        self.labels_path = os.path.join(self.images_path, "labels")
        self.dataset = self._create_dataset()

    def _create_dataset(self) -> Dataset:
        dataset = []
        for filename in os.listdir(os.path.join(self.images_path)):
            if filename.endswith(".JPG"):
                img_path = os.path.join(self.images_path, filename)
                annotation_path = os.path.join(
                    self.labels_path, filename.split(".")[0] + ".xml"
                )

                if os.path.exists(annotation_path):
                    annotation = self._parse_annotation(annotation_path)
                    if len(annotation["boxes"]) > 0:
                        dataset.append(
                            (
                                np.asarray(Image.open(img_path)),
                                {
                                    "boxes": annotation["boxes"][0],
                                    "class_labels": annotation["class_labels"][0],
                                },
                            )
                        )
                    else:
                        # TODO: Add empty dataset entry
                        continue

        logging.info(
            f"{len(dataset)} number of annotated images loaded in {self.images_path}"
        )
        return dataset

    def _parse_annotation(self, annotation_path: str) -> dict:
        root = ElementTree.parse(annotation_path).getroot()
        boxes = []
        for obj in root.findall("object"):
            if obj.find("name").text == "human":
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "class_labels": torch.ones(len(boxes), dtype=torch.int64),
        }

    def __getitem__(self, index: int) -> dict:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == "__main__":
    setup_logging()

    train_data = IPSARDataset("drone_sar/data", "train")
    test_data = IPSARDataset("drone_sar/data", "test")

    detector = ObjectDetector()
    detector.fine_tune(train_data, test_data)
