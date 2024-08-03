# Drone Search and Rescue

The Drone Search and Rescue project is a tool designed to enchance full-scale search and rescue operations through object detection within drone-generated video content.

## Description

The University Rescue Squad is a first responders rescue organization in Bulgaria that conducts search and rescue (SaR) operations in all kinds of natural terrain. During such operations, often aerial reconnaissance is employed deploying unmanned aerial vehicles to survey areas where missing persons are believed to be located. However, the subsequent manual review of the captured video footage introduces the high possibility of human errors.

The Drone Search and Rescue project addresses this concern by developing a system for object detection within drone-captured video footage. The project specifically targets the identification of various objects like humans, backpacks, bicycles etc.

## Instructions for installation and usage

- Download the repo `git clone https://github.com/UniversityRescueSquad/DroneSearchAndRescue`
  - Or pull the latest version `git pull`
- [Download python](https://www.python.org/downloads/)
- Navigate to the repo folder using a terminal and execute `pip install .`
- Download the trained model from [here](https://drive.google.com/drive/folders/1xiE6QkffSoHG12gfbxd4spzGZjGtFl2_)
  - file name - `epoch=65-step=10494.ckpt`
- Execute `python drone_sar/cli.py -h` to check if everything is installed properly
- Execute the following command to run on folder of images:

  ```bash
  python drone_sar/cli.py \
    --model_path ./epoch=65-step=10494.ckpt \
    --images_dir ./example_img_dir \
    --export_dir ./results \
    --device cpu
  ```

  Adjust the parameters according to the location of the model and the input and output dirs

## License

The Drone Search and Rescue project is licensed under the MIT License

Copyright (c) [2023] [University Rescue Squad]

Refer to the [LICENSE](LICENSE) for detailed information.

## Acknowledgement

Dunja Božić-Štulić, Željko Marušić, Sven Gotovac: Deep Learning Approach on Aerial Imagery in Supporting Land Search and Rescue Missions, International Journal of Computer Vision, 2019.

