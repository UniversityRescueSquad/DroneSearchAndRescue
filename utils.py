import argparse
import os
import logging

def setup_logging():
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def get_files(location):
    """
    Returns all files in the specified location if it is a folder,
    or returns an array with the specified file if the location is pointing to a file.
    If location is not provided then get all files from the 'videos' folder.
    """
    if not location:
        location = 'videos'
    if os.path.isfile(location):
        # if location is pointing to a file
        return [location]
    elif os.path.isdir(location):
        # if location is a folder
        files = []
        for file_name in os.listdir(location):
            file_path = os.path.join(location, file_name)
            if os.path.isfile(file_path):
                files.append(file_path)
        return files
    else:
        # if location does not exist
        raise ValueError("Location does not exist.")

def parse_args():
    parser = argparse.ArgumentParser(description="Human detection using machine learning")
    parser.add_argument("--input", required=False, help="Path to input video file")
    parser.add_argument("--output", help="Output video file path")
    parser.add_argument("--confidence", type=float, default=0.9, help="Object detection confidence threshold")
    parser.add_argument("--detection_type", default="Fast", help="Edditor level (Fast, Medium, Complete)")
    return parser.parse_args()
    