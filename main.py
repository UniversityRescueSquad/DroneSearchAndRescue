import logging
import argparse
import os
from object_detector import ObjectDetector
from video_editor import VideoEditor

def setup_logging(log_level):
    # Create logger and set level
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(console_handler)
    return logger

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
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log)

    detector = ObjectDetector(logger)
    videoEditor = VideoEditor(logger)

    for file in get_files(args.input):
        # Detect objects in input video and save to output video
        logger.info(f"Start detecting objects in '{file}'")
        
        videoEditor.open_video_file(file)
        
        if args.output:
            videoEditor.save_output_video(args.output)

        while videoEditor.has_next_frame():
            frame = videoEditor.get_next_frame()

            outputs = detector.detect(frame, confidence_threshold=args.confidence)

            frame = videoEditor.get_processed_frame(frame, outputs)
            videoEditor.show_frame(frame)

            if videoEditor.exit_video():
                break

        videoEditor.close_video_file()