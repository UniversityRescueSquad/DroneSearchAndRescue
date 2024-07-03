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
            id = self.path, index
            return id, frame


def indexable_file_loader(path):
    ext = path.split(".")[-1].lower()
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
    def __init__(self) -> None:
        frame_indexable = indexable_file_loader("/home/iz/workspace/DJI_0002.MP4")
        sampler = SimpleSampler(len(frame_indexable))
        self.model = LightningDetector.load_from_checkpoint(
            "/home/iz/workspace/DroneSearchAndRescue/.checkpoints/epoch=65-step=10494.ckpt",
            lr=0.01,
        )
        self.subscribers = []
        self.predictions_state = {}
        # is_running = False
        # thread = threading.Thread(target=run)
        # thread.start()

        # def run():
        #     while True:
        #         if is_running:
        #             pil = frame_indexable[index]
        #             self.model.predict(pil)
        #         else:
        #             time.sleep(500)

        # def start():
        #     nonlocal is_running
        #     is_running = True

        # def stop():
        #     nonlocal is_running
        #     is_running = False

        self.container = pn.Column()

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)

    def force_predict(self, frame_identifier):
        path, index = frame_identifier
        frame_identifier, frame = indexable_file_loader(path)[index]
        prediction = self.model.predict(frame)
        self.predictions_state[frame_identifier] = prediction
        self.notify(frame_identifier)

    def get_prediction(self, frame_identifier):
        if frame_identifier in self.predictions_state:
            return self.predictions_state[frame_identifier]
        return None

    def notify(self, frame_identifier):
        for it in self.subscribers:
            it.update_prediction(self.predictions_state, frame_identifier)

    def render(self):
        return self.container


class PredictionViewer:
    def __init__(self, prediction_manager) -> None:
        self.current_frame_index = pn.widgets.IntSlider(
            name="Current Frame",
            value=0,
            start=0,
            end=100,
            width=800,
        )
        self.prediction_manager = prediction_manager
        self.current_frame_identifier = None
        self.prediction_manager.subscribe(self)

        self.current_frame_container = pn.Column("")
        next_btn = pn.widgets.Button(name=">", button_type="primary")
        prev_btn = pn.widgets.Button(name="<", button_type="primary")
        next_btn.on_click(lambda _: update_frame(frame_offset=1))
        prev_btn.on_click(lambda _: update_frame(frame_offset=-1))

        next_next_btn = pn.widgets.Button(name=">>", button_type="primary")
        prev_prev_btn = pn.widgets.Button(name="<<", button_type="primary")
        next_next_btn.on_click(lambda _: update_frame(frame_offset=5 * 30))
        prev_prev_btn.on_click(lambda _: update_frame(frame_offset=-5 * 30))

        predict_btn = pn.widgets.Button(name="Predict", button_type="primary")
        predict_btn.on_click(lambda _: predict_on_current())

        def predict_on_current():
            if self.current_frame_identifier is not None:
                self.container.loading = True
                self.prediction_manager.force_predict(self.current_frame_identifier)

        def update_frame(frame_offset):
            new_value = np.clip(
                self.current_frame_index.value + frame_offset,
                0,
                len(self.frames_indexable) - 1,
            )
            self.current_frame_index.value = new_value

        self.current_frame_index.param.watch(lambda _: self._show(), "value")
        self.container = pn.Column(
            predict_btn,
            self.current_frame_container,
            pn.Column(
                self.current_frame_index,
                pn.Row(prev_prev_btn, prev_btn, next_btn, next_next_btn),
                styles=dict(background="#eeeeee"),
            ),
            styles=dict(background="#eeeeee", padding="25px 15px"),
        )

    def draw_prediction(self, pil, pred):
        print("drawing frame", pred)
        im = np.array(pil)
        for x0, y0, w, h in pred:
            x1, y1 = x0 + w, y0 + h
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            im = cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 0), 5)

        return Image.fromarray(im)

    def _show(self):
        self.container.loading = True
        self.current_frame_identifier, frame = self.frames_indexable[
            self.current_frame_index.value
        ]
        prediction = self.prediction_manager.get_prediction(
            self.current_frame_identifier
        )

        if prediction is not None:
            frame = self.draw_prediction(frame, prediction)

        if frame is not None:
            frame.thumbnail((800, 800))

        self.current_frame_container[0] = frame
        self.container.loading = False

    def update_frame(self, frame_indexable, index=0):
        self.frames_indexable = frame_indexable
        self.current_frame_index.value = index
        self.current_frame_index.end = len(frame_indexable)
        self._show()

    def update_prediction(self, all_predictions, frame_identifier):
        self._show()

    def render(self):
        return self.container


class App:
    def __init__(self):
        prediction_manager = PredictionManager()
        video_indexable = VideoIndexable("/home/iz/workspace/DJI_0002.MP4")
        self.current_mission = PredictionViewer(prediction_manager)
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
