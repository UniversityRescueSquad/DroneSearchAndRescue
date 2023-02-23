from video_editor import CompleteVideoEditor, FastVideoEditor, MediumVideoEditor

class VideoEditorFactory:
    @staticmethod
    def create_video_editor(editor_type):
        if editor_type == 'Fast':
            return FastVideoEditor()
        elif editor_type == 'Medium':
            return MediumVideoEditor()
        elif editor_type == 'Complete':
            return CompleteVideoEditor()
        else:
            raise ValueError(f"Invalid editor type: {editor_type}")