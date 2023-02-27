import logging
import argparse
import os
from object_detector import ObjectDetector
from video_processor_factory import VideoProcessorFactory

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

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    setup_logging()

    detector = ObjectDetector()
    video_processor = VideoProcessorFactory.create_video_editor(args.detection_type)

    for file in get_files(args.input):
        # Detect objects in input video and save to output video
        logging.info(f"Start detecting objects in '{file}'")
        
        if not video_processor.open_video_file(file):
            logging.error(f'Could not open file {file}')
            continue
        
        video_processor.save_output_video(args.output)
        
        while video_processor.has_next_frame():
            frame = video_processor.get_next_frame()

            outputs = detector.detect(frame, confidence_threshold=args.confidence)

            frame = video_processor.get_processed_frame(frame, outputs)
            video_processor.show_frame(frame)

            if video_processor.exit_video():
                break

        video_processor.close_video_file()