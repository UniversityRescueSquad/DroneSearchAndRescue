import pickle
from threading import Thread
import time
import cv2
from PIL import Image
import numpy as np
from datetime import datetime
import panel as pn
import os

import torch

from drone_sar.lightning_detector import LightningDetector
from drone_sar.vis import draw_prediction

# from drone_sar.lightning_detector import LightningDetector

pn.extension()
pn.config.throttled = True
# pn.extension(nthreads=4)
pn.extension(notifications=True)

DEFAULT_MODEL_PATH = os.path.abspath("./workdir/epoch=65-step=10494.ckpt")
DEFAULT_INPUT_DIR = os.path.abspath("./workdir/input_dir")
DEFAULT_OUTPUT_DIR = os.path.abspath("./workdir/output_dir")


class VideoIndexable:
    def __init__(self, path) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frames_count = int(self.fps)

    def get_display_duration(self):
        duration_in_seconds = self.frames_count / self.fps
        delta_time = datetime.timedelta(seconds=duration_in_seconds)
        return str(delta_time)

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


class PredictionManager:
    def __init__(self):
        self.is_running = False
        self.prediction_initiated = False
        self.on_progress = None
        self.on_done = None
        self.on_error = None
        self.on_finally = None

    def on(self, on_progress, on_done, on_error, on_finally):
        self.on_progress = on_progress
        self.on_done = on_done
        self.on_error = on_error
        self.on_finally = on_finally

    def start(self, **kwargs):
        self.is_running = True
        self.prediction_initiated = True
        self.process = Thread(target=self._run, kwargs=kwargs)
        self.process.start()

    # def stop(self):
    #     self.process.terminate()
    #     self.is_running = False
    #     self.on_finally()

    def _run(self, model_path, input_file_path, output_dir, device):
        try:
            print("> Process start")
            now = datetime.now()
            task_name = f"prediction-task-{now}"
            task_dir = os.path.join(output_dir, task_name)

            model = LightningDetector.load_from_checkpoint(
                model_path, lr=0.01, ignore_mismatched_sizes=True
            ).to(device)

            target_indexable = VideoIndexable(input_file_path)
            target_length = len(target_indexable)

            print(f"> Number of frames {target_length}")

            os.makedirs(task_dir, exist_ok=True)
            with torch.inference_mode():
                for index in range(0, target_length, 30 * 10):
                    indexing_result = target_indexable[index]
                    if indexing_result is not None:
                        _, frame_pil = indexing_result
                    prediction_out = model.predict(frame_pil)

                    bbox_pil = draw_prediction(frame_pil, prediction_out)
                    result_path = os.path.join(task_dir, f"{index}.png")
                    bbox_pil.save(result_path)

                    prediction_path = os.path.join(task_dir, f"{index}.pkl")
                    with open(prediction_path, "wb+") as fp:
                        pickle.dump(
                            {"output": prediction_out, "frame_index": index}, fp
                        )

                    percentage = int((index + 1) / target_length * 100)
                    self.on_progress(percentage)
            print("> Process done")
            self.on_done()
        except Exception as e:
            self.on_error(e)
        finally:
            self.is_running = False
            self.on_finally()


def handle_reload_input_dir(input_dir, target_file_selector):
    target_file_selector.loading = True
    target_file_selector.disabled = True
    try:
        files = os.listdir(input_dir)
        valid_files = [f for f in files if f.lower().endswith(".mp4")]
        target_file_selector.options = valid_files
        target_file_selector.value = valid_files[0]
        target_file_selector.disabled = False
    except Exception as e:
        pn.state.notifications.error("Invalid input dir!", duration=1000)
    finally:
        target_file_selector.loading = False


def handle_start_prediction(
    controls_container,
    ongoing_prediction_container,
    prediction_manager,
    model_path_el,
    input_dir_el,
    output_dir_el,
    target_file_selector,
    device_selector_el,
):
    controls_container.loading = True
    ongoing_prediction_container.visible = True
    input_file_path = os.path.join(input_dir_el.value, target_file_selector.value)
    prediction_manager.start(
        model_path=model_path_el.value,
        input_file_path=input_file_path,
        output_dir=output_dir_el.value,
        device=device_selector_el.value,
    )


def handle_end_prediction(controls_container):
    controls_container.loading = False


def handle_prediction_success(result_el):
    result_el.clear()
    result_el.append(
        pn.pane.Alert("### Prediction finished successfully", alert_type="success")
    )


def handle_prediction_error(e, result_el):
    result_el.clear()
    result_el.append(
        pn.pane.Alert(f"### Error occurred during prediction {e}", alert_type="danger")
    )


def handle_prediction_update(percent, progress_element):
    progress_element.value = percent


prediction_manager = PredictionManager()


def render_app():
    template_container = pn.template.MaterialTemplate(
        title="⛑️ University Rescue Squad - DRONE SAR"
    )

    model_path_el = pn.widgets.TextInput(value=DEFAULT_MODEL_PATH, name="Path to model")
    input_dir_el = pn.widgets.TextInput(value=DEFAULT_INPUT_DIR, name="Input directory")
    skip_frames_el = pn.widgets.TextInput(value=30 * 5, name="Frames to skip")
    output_dir_el = pn.widgets.TextInput(
        value=DEFAULT_OUTPUT_DIR, name="Output directory"
    )

    target_file_selector = pn.widgets.Select(name="Input target video")
    input_dir_el.param.watch(
        lambda e: handle_reload_input_dir(input_dir_el.value, target_file_selector),
        "value",
    )
    handle_reload_input_dir(input_dir_el.value, target_file_selector)

    device_selector_el = pn.widgets.Select(options=["cpu", "cuda", "mps"], value="cpu")

    prediction_progress_el = pn.indicators.Progress(name="Prediction Progress", value=0)
    prediction_result_el = pn.Column()

    ongoing_prediction_container = pn.Column(
        "**Ongoing prediction**",
        prediction_progress_el,
        prediction_result_el,
    )
    ongoing_prediction_container.visible = prediction_manager.prediction_initiated

    start_prediction_btn = pn.widgets.Button(name="Start Prediction")
    start_prediction_btn.on_click(
        lambda e: handle_start_prediction(
            controls_container,
            ongoing_prediction_container,
            prediction_manager,
            model_path_el,
            input_dir_el,
            output_dir_el,
            target_file_selector,
            device_selector_el,
        )
    )

    prediction_manager.on(
        lambda p: handle_prediction_update(p, prediction_progress_el),
        lambda: handle_prediction_success(prediction_result_el),
        lambda e: handle_prediction_error(e, prediction_result_el),
        lambda: handle_end_prediction(controls_container),
    )

    controls_container = pn.Column(
        "**Configurable parameters**",
        model_path_el,
        input_dir_el,
        output_dir_el,
        target_file_selector,
        device_selector_el,
        start_prediction_btn,
        width=600,
    )
    controls_container.loading = prediction_manager.is_running

    prediction_container = pn.Column(
        controls_container,
        ongoing_prediction_container,
    )

    template_container.main.append(prediction_container)

    return template_container


pn.serve(
    {"/": render_app},
    port=8887,
    show=False,
)
