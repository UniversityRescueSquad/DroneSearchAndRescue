import cv2
from PIL import Image
import numpy as np
import panel as pn

pn.extension()


class VideoIndexable:
    def __init__(self, path) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.frames_count

    def __getitem__(self, index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self.cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            return frame


class PredictionMission:
    def __init__(self, frames_indexable) -> None:
        frames_indeable = frames_indexable
        current_frame_index = 0

        current_frame_container = pn.Column("")
        next_btn = pn.widgets.Button(name=">", button_type="primary")
        prev_btn = pn.widgets.Button(name="<", button_type="primary")
        next_btn.on_click(lambda _: show(frame_offset=100))
        prev_btn.on_click(lambda _: show(frame_offset=-100))

        def show(frame_offset=0):
            self.container.loading = True
            nonlocal current_frame_index
            current_frame_index += frame_offset
            current_frame_index = np.clip(current_frame_index, 0, len(frames_indeable))
            frame = frames_indeable[current_frame_index]
            if frame is not None:
                frame.thumbnail((800, 800))

            current_frame_container[0] = frame
            self.container.loading = False

        self.container = pn.Column(current_frame_container, pn.Row(prev_btn, next_btn))
        show()

    def render(self):
        return self.container


class App:
    def __init__(self):
        video_inxable = VideoIndexable("/home/iz/workspace/DJI_0002.MP4")
        self.current_mssion = PredictionMission(video_inxable)  ## TODO

    def get_files_list_container(self):
        container = pn.Row()

        container.append("Files list")

        return container

    def render(self):
        template_container = pn.template.MaterialTemplate(title="UASO - DRONE SAR")

        files_list_container = self.get_files_list_container()

        current_mission_element = self.current_mssion.render()

        template_container.main.append(files_list_container)
        template_container.main.append(current_mission_element)

        return template_container


pn.serve(
    {"/": App().render()},
    port=8888,
    show=False,
)
