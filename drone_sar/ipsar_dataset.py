import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from xml.etree import ElementTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import joblib


class IPSARDataset(Dataset):
    def __init__(self, images_root: str):
        self.labels_path = os.path.join(images_root, "labels")
        image_paths = [os.path.join(images_root, f) for f in os.listdir(images_root)]
        self.image_paths = [path for path in image_paths if os.path.isfile(path)]

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        dir_path, file_name = os.path.split(image_path)
        file_identifier, ext = file_name.split(".")

        annotation_path = os.path.join(self.labels_path, f"{file_identifier}.xml")
        annotation = self._parse_annotation(annotation_path)
        pil = Image.open(image_path)

        return {
            "img_path": image_path,
            "pil": pil,
            "target_sizes": pil.size,
            "boxes": annotation["boxes"],
            "class_labels": annotation["class_labels"],
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_dataloader(self, processor, bs, shuffle):
        return torch.utils.data.DataLoader(
            self,
            shuffle=shuffle,
            batch_size=bs,
            collate_fn=lambda b: self.collate_fn(b, processor),
        )

    @staticmethod
    def collate_fn(items, processor):
        @joblib.Memory("~/.cache", verbose=0).cache
        def preprocess_cached(im_path):
            dos = {
                "do_resize": True,
                "do_rescale": True,
                "do_normalize": True,
                "do_pad": False,
            }
            image = Image.open(im_path)
            prep_image = (
                processor(images=image, return_tensors="pt", **dos)["pixel_values"][0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            return prep_image, image.size

        dos = {
            "do_resize": False,
            "do_rescale": False,
            "do_normalize": False,
            "do_pad": True,
        }
        preprocessed = [preprocess_cached(it["img_path"]) for it in items]
        images = [image for image, size in preprocessed]
        sizes = [size for image, size in preprocessed]
        processed_images = processor(images=images, return_tensors="pt", **dos)

        def format_annotation(boxes, W, H):
            if len(boxes) == 0:
                return boxes
            boxes = boxes.clone()
            boxes[:, 0] /= W
            boxes[:, 1] /= H
            boxes[:, 2] /= W
            boxes[:, 3] /= H
            return boxes

        labels = [
            {
                "boxes": format_annotation(it["boxes"], W, H),
                "class_labels": torch.zeros(len(it["boxes"]), dtype=torch.long),
            }
            for it, (W, H) in zip(items, sizes)
        ]
        info = {"target_sizes": sizes}
        return {"labels": labels, **processed_images}, info

    def _parse_annotation(self, annotation_path: str) -> dict:
        boxes = []
        if os.path.exists(annotation_path):
            root = ElementTree.parse(annotation_path).getroot()
            for obj in root.findall("object"):
                if obj.find("name").text == "human":
                    bbox = obj.find("bndbox")
                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)
                    boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=float)

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "class_labels": torch.ones(len(boxes), dtype=torch.int64),
        }
