from drone_sar.vis import draw_prediction
import torch
import pickle
from PIL import Image
import argparse
from tqdm.auto import tqdm
import os
from drone_sar.lightning_detector import LightningDetector

parser = argparse.ArgumentParser("Detector CLI")

parser.add_argument("--model_path", required=True)
parser.add_argument("--images_dir", required=True)
parser.add_argument("--export_dir", required=True)
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

args = parser.parse_args()
results = []

model = LightningDetector.load_from_checkpoint(
    args.model_path, lr=0.01, ignore_mismatched_sizes=True
).to(args.device)

os.makedirs(args.export_dir, exist_ok=True)
with torch.inference_mode():
    for i, img_name in enumerate(tqdm(os.listdir(args.images_dir))):
        img_path = os.path.join(args.images_dir, img_name)
        pil = Image.open(img_path)
        out = model.predict(pil)

        bbox_pil = draw_prediction(pil, out)
        result_path = os.path.join(args.export_dir, img_name)
        bbox_pil.save(result_path)

        results.append({"output": out, "frame_index": i})


results_path = os.path.join(args.export_dir, "precomputed_results.pkl")
with open(results_path, "wb+") as fp:
    pickle.dump(results, fp)

print("Done!")
