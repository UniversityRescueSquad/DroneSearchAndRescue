"""
Setup
- Download and run - https://github.com/sallar/mac-local-rtmp-server
- Open OBS and create a media device connected to the stream
  - As shown here: https://www.youtube.com/watch?v=n94jGIXWWZQ
- Run OBS Virtual camera
- Run this script capturing the camera footage and running predictions
  - ```bash
    poetry run python drone_sar/live.py --model_path ~/workspace/epoch=65-step=10494.ckpt
    ```
"""

import cv2
from PIL import Image
import numpy as np

from drone_sar.vis import draw_prediction
import torch
import argparse
from tqdm.auto import tqdm
import os
from drone_sar.lightning_detector import LightningDetector

parser = argparse.ArgumentParser("Detector CLI")

parser.add_argument("--model_path", required=True)
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

args = parser.parse_args()

model = LightningDetector.load_from_checkpoint(
    args.model_path, lr=0.01, ignore_mismatched_sizes=True
).to(args.device)

cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
webcam = cv2.VideoCapture(1)

with torch.inference_mode():
    while True:
        ret, frame = webcam.read()
        print("Frame captured")
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)
        out = model.predict(pil)
        print("Found preds: ", len(out))
        bbox_pil = draw_prediction(pil, out)
        print("Done predicting")

        cv2.imshow("video", np.array(bbox_pil))
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
