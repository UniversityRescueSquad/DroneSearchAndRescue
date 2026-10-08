# Drone Search and Rescue

The Drone Search and Rescue project is a tool designed to enhance full-scale search and rescue operations through object detection within drone-generated video content.

## Description

The University Rescue Squad is a first responders rescue organization in Bulgaria that conducts search and rescue (SaR) operations in all kinds of natural terrain. During such operations, often aerial reconnaissance is employed deploying unmanned aerial vehicles to survey areas where missing persons are believed to be located. However, the subsequent manual review of the captured video footage introduces the high possibility of human errors.

The Drone Search and Rescue project addresses this concern by developing a system for object detection within drone-captured video footage. The model is a DETR (ResNet-50) fine-tuned for human detection on a combination of the HERIDAL and SARD datasets.

## Installation

- Clone the repo: `git clone https://github.com/UniversityRescueSquad/DroneSearchAndRescue`
- [Install Python](https://www.python.org/downloads/) (3.10 - 3.13) and [Poetry](https://python-poetry.org/docs/#installation)
- Install the project and its dependencies:

  ```bash
  poetry install
  ```

  This creates a virtual environment in `.venv` inside the project. Run commands through `poetry run`, or activate the environment with `poetry env activate`.

The desktop app additionally needs tkinter:

```bash
# ubuntu
sudo apt-get install tk-dev

# macos
brew install python-tk tcl-tk
```

## Dataset preparation

Training uses a unified dataset built from two public datasets:

- [HERIDAL](http://ipsar.fesb.unist.hr/HERIDAL%20database.html) - high resolution aerial photos: `wget http://ipsar.fesb.unist.hr/ipsar_data/heridal.zip`
- [SARD](https://universe.roboflow.com/datasets-pdabr/sard-8xjhy) - video frames, exported from Roboflow in COCO format

Extract them to `data/heridal` and `data/sard`, then:

1. Run `notebooks/dataset_demo_and_harmonization.ipynb`. It analyzes both datasets and writes a merged copy to `data/unified_dataset` (single "human" class, Pascal VOC XML labels).
2. Split it into train/val/test:

   ```bash
   poetry run python drone_sar/split_unified_dataset.py
   ```

   This restores the original per-source splits from the filenames and carves a validation set out of the HERIDAL training images.

## Training

```bash
poetry run python drone_sar/train.py --device cuda
```

Useful options: `--epochs`, `--batch-size`, `--lr`, `--data-root`. Progress is logged to TensorBoard under `logs/` and the best checkpoints (by validation mAP@50) are kept in `.checkpoints/`.

To verify the setup without a GPU, run a quick pipeline check on two small batches:

```bash
poetry run python drone_sar/train.py --device cpu --batch-size 2 --smoke
```

A GPU is strongly recommended for a real run; a free Colab/Kaggle T4 is enough for this dataset size.

## Running detection on images

```bash
poetry run python drone_sar/cli.py \
  --model_path .checkpoints/<your-checkpoint>.ckpt \
  --images_dir ./example_img_dir \
  --export_dir ./results \
  --device cpu
```

This draws the detected boxes on each image and saves the results, along with a `precomputed_results.pkl` file with the raw outputs.

Note: checkpoints trained before the box format fix (including the previously shared `epoch=65-step=10494.ckpt`) are not compatible with the current code and need to be retrained.

## License

The Drone Search and Rescue project is licensed under the MIT License

Copyright (c) [2023] [University Rescue Squad]

Refer to the [LICENSE](LICENSE) for detailed information.

## Acknowledgement

Dunja Božić-Štulić, Željko Marušić, Sven Gotovac: Deep Learning Approach on Aerial Imagery in Supporting Land Search and Rescue Missions, International Journal of Computer Vision, 2019.
