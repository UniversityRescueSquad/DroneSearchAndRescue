import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from xml.etree import ElementTree
import numpy as np


class IPSARDataset(Dataset):
    def __init__(self, images_root: str):
        self.labels_path = os.path.join(images_root, "labels")
        image_paths = [os.path.join(images_root, f) for f in os.listdir(images_root)]
        self.image_paths = [path for path in image_paths if os.path.isfile(path)]

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
        image_path = self.image_paths[index]
        dir_path, file_name = os.path.split(image_path)
        file_identifier, ext = file_name.split(".")
        annotation_path = os.path.join(self.labels_path, f"{file_identifier}.xml")
        annotation = self._parse_annotation(annotation_path)
        pil = Image.open(image_path)
        pixel_values = torch.tensor(np.array(pil))

        return {
            "img_path": image_path,
            "pil": pil,
            "pixel_values": pixel_values,
            "boxes": annotation["boxes"],
            "class_labels": annotation["class_labels"],
        }

    def __len__(self) -> int:
        return len(self.image_paths)


if __name__ == "__main__":
    train_data = IPSARDataset("drone_sar/data", "train")
    test_data = IPSARDataset("drone_sar/data", "test")
