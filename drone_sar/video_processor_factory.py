from video_processor import MediumVideoProcessor, FastVideoProcessor, CompleteVideoProcessor, VideoProcessor  

class VideoProcessorFactory:
    @staticmethod
    def create_video_processor(editor_type: str) -> VideoProcessor:
        if editor_type == 'Fast':
            return FastVideoProcessor()
        elif editor_type == 'Medium':
            return MediumVideoProcessor()
        elif editor_type == 'Complete':
            return CompleteVideoProcessor()
        else:
            raise ValueError(f"Invalid Processor type: {editor_type}")