from video_reader import MediumVideoReader, FastVideoReader, CompleteVideoReader  

class VideoReaderFactory:
    @staticmethod
    def create_video_editor(editor_type):
        if editor_type == 'Fast':
            return FastVideoReader()
        elif editor_type == 'Medium':
            return MediumVideoReader()
        elif editor_type == 'Complete':
            return CompleteVideoReader()
        else:
            raise ValueError(f"Invalid reader type: {editor_type}")