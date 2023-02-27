import logging
from object_detector import ObjectDetector
from utils import get_files, parse_args, setup_logging
from video_processor_factory import VideoProcessorFactory

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    setup_logging()

    detector = ObjectDetector()
    video_processor = VideoProcessorFactory.create_video_processor(args.detection_type)

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