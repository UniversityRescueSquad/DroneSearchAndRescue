import threading
import time
import cv2
from PIL import Image
import numpy as np
import panel as pn

from drone_sar.lightning_detector import LightningDetector

pn.extension()
pn.config.throttled = True
pn.extension(nthreads=4)


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


def indexable_file_loader(path):
    ext = path.split(".")[-1:].lower()
    if ext in ["mp4"]:
        return VideoIndexable(path)
    raise NotImplementedError()


class SimpleSampler:
    def __init__(self, len) -> None:
        self.len = len

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        current = self.current
        if current < self.len:
            self.current += 1
            return current
        raise StopIteration()


class PredictionManager:
    def __init__(self, subscribers) -> None:
        frame_indexable = indexable_file_loader("/home/iz/workspace/DJI_0002.MP4")
        sampler = SimpleSampler(len(frame_indexable))
        self.model = LightningDetector.load_from_checkpoint(
            "/home/iz/workspace/DroneSearchAndRescue/.checkpoints/epoch=65-step=10494.ckpt"
        )
        self.subscribers = subscribers

        self.predictions_state = {}
        is_running = False
        thread = threading.Thread(target=run)
        thread.start()

        def run():
            while True:
                if is_running:
                    pil = frame_indexable[index]
                    self.model.predict(pil)
                else:
                    time.sleep(500)

        def start():
            nonlocal is_running
            is_running = True

        def stop():
            nonlocal is_running
            is_running = False

        self.container = pn.Column()

    def force_predict(self, path, index):
        pil = indexable_file_loader(path)[index]
        prediction = self.model.predict(pil)
        self.predictions_state[(path, index)] = prediction
        self.notify(path, index)

    def notify(self, path, index):
        for it in self.subscribers:
            it.update(self.predictions_state, path, index)

    def render(self):
        return self.container


class PredictionViewer:
    def __init__(self) -> None:
        self.current_frame_index = pn.widgets.IntSlider(
            name="Current Frame",
            value=0,
            start=0,
            end=100,
            width=800,
        )

        self.current_frame_container = pn.Column("")
        next_btn = pn.widgets.Button(name=">", button_type="primary")
        prev_btn = pn.widgets.Button(name="<", button_type="primary")
        next_btn.on_click(lambda _: update_frame(frame_offset=1))
        prev_btn.on_click(lambda _: update_frame(frame_offset=-1))

        next_next_btn = pn.widgets.Button(name=">>", button_type="primary")
        prev_prev_btn = pn.widgets.Button(name="<<", button_type="primary")
        next_next_btn.on_click(lambda _: update_frame(frame_offset=5 * 30))
        prev_prev_btn.on_click(lambda _: update_frame(frame_offset=-5 * 30))

        def update_frame(frame_offset):
            new_value = np.clip(
                self.current_frame_index.value + frame_offset,
                0,
                len(self.frames_indexable) - 1,
            )
            self.current_frame_index.value = new_value

        self.curret_frame_index.param.watch(lambda _: self._show(), "value")
        self.container = pn.Column(
            self.current_frame_container,
            pn.Column(
                self.current_frame_index,
                pn.Row(prev_prev_btn, prev_btn, next_btn, next_next_btn),
                styles=dict(background="#eeeeee"),
            ),
            styles=dict(background="#eeeeee", padding="25px 15px"),
        )

    def _show(self):
        self.container.loading = True
        frame = self.frames_indexable[self.current_frame_index.value]
        if frame is not None:
            frame.thumbnail((800, 800))

        self.current_frame_container[0] = frame
        self.container.loading = False

    def update_frame(self, frame_indexable, index=0):
        self.frames_indexable = frame_indexable
        self.current_frame_index.value = index
        self.current_frame_index.end = len(frame_indexable)
        self._show()

    def render(self):
        return self.container


class App:
    def __init__(self):
        video_indexable = VideoIndexable("/home/iz/workspace/DJI_0002.MP4")
        self.current_mission = PredictionViewer()  ## TODO
        self.current_mission.update_frame(video_indexable)

    def get_files_list_container(self):
        container = pn.Row()

        container.append("Files list")

        return container

    def render(self):
        template_container = pn.template.MaterialTemplate(title="UASO - DRONE SAR")

        files_list_container = self.get_files_list_container()

        current_mission_element = self.current_mission.render()

        template_container.main.append(files_list_container)
        template_container.main.append(current_mission_element)

        return template_container


pn.serve(
    {"/": App().render()},
    port=8888,
    show=False,
)
