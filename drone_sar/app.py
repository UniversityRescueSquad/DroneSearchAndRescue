from dataclasses import dataclass
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
import logging

from drone_sar.lightning_detector import LightningDetector
from drone_sar.vis import draw_prediction

# from drone_sar.lightning_detector import LightningDetector

logger = logging.getLogger(__file__)

pn.extension(
    notifications=True,
    raw_css=[
        """
            * {word-break: break-all;}
        """
    ],
)
pn.config.throttled = True
# pn.extension(nthreads=4)

DEFAULT_MODEL_PATH = os.path.abspath("./workdir/epoch=65-step=10494.ckpt")
DEFAULT_INPUT_DIR = os.path.abspath("./workdir/input_dir")
DEFAULT_OUTPUT_DIR = os.path.abspath("./workdir/output_dir")


class VideoIndexable:
    def __init__(self, path) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_display_duration(self):
        duration_in_seconds = self.frames_count / self.fps
        s = duration_in_seconds
        hours, remainder = divmod(s, 3600)
        minutes, seconds = divmod(remainder, 60)

        return "{:02}h {:02}m {:02}s".format(int(hours), int(minutes), int(seconds))

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

    def _run(
        self, model_path, input_file_path, output_dir, skip_frames, tqdm_el, device
    ):
        try:
            print("> Process start")
            now = datetime.now()
            task_name = f"prediction-task-{now}"
            task_dir = os.path.join(output_dir, task_name)
            images_dir = os.path.join(output_dir, task_name, "images")
            predictions_dir = os.path.join(output_dir, task_name, "predictions")

            model = LightningDetector.load_from_checkpoint(
                model_path, lr=0.01, ignore_mismatched_sizes=True
            ).to(device)

            target_indexable = VideoIndexable(input_file_path)
            target_length = len(target_indexable)

            print(f"> Number of frames {target_length}")

            os.makedirs(task_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(predictions_dir, exist_ok=True)

            with torch.inference_mode():
                for index in tqdm_el(range(0, target_length, skip_frames)):
                    indexing_result = target_indexable[index]
                    if indexing_result is not None:
                        _, frame_pil = indexing_result
                    prediction_out = model.predict(frame_pil)

                    if len(prediction_out) > 0:
                        bbox_pil = draw_prediction(frame_pil, prediction_out)
                        result_path = os.path.join(images_dir, f"{index}.png")
                        bbox_pil.save(result_path)

                    prediction_path = os.path.join(predictions_dir, f"{index}.pkl")
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
            logger.error(e)
        finally:
            self.is_running = False
            self.on_finally()


@dataclass
class Elements:
    WIDTH = 450

    template_container = pn.template.MaterialTemplate(
        title="⛑️ University Rescue Squad - DRONE SAR"
    )
    model_path_el = pn.widgets.TextInput(
        value=DEFAULT_MODEL_PATH, name="Path to model", width=WIDTH
    )
    input_dir_el = pn.widgets.TextInput(
        value=DEFAULT_INPUT_DIR, name="Input directory", width=WIDTH
    )
    skip_frames_el = pn.widgets.IntInput(
        value=30 * 5, name="Every Nth frame", step=1, start=1, width=WIDTH
    )
    output_dir_el = pn.widgets.TextInput(
        value=DEFAULT_OUTPUT_DIR, name="Output directory", width=WIDTH
    )
    target_file_selector = pn.widgets.Select(name="Input target video", width=WIDTH)
    target_file_info = pn.pane.Alert(
        "### Video info", alert_type="primary", visible=True, width=WIDTH
    )
    device_selector_el = pn.widgets.Select(
        options=["cpu", "cuda", "mps"], value="cpu", width=WIDTH
    )
    prediction_progress_el = pn.widgets.Tqdm(value=0, write_to_console=True)
    prediction_result_el = pn.Column()
    ongoing_prediction_container = pn.Column(
        "**Ongoing prediction**",
        prediction_progress_el,
        prediction_result_el,
    )
    start_prediction_btn = pn.widgets.Button(name="Start Prediction")
    controls_container = pn.Column(
        "**Configurable parameters**",
        model_path_el,
        input_dir_el,
        output_dir_el,
        target_file_selector,
        skip_frames_el,
        target_file_info,
        device_selector_el,
        start_prediction_btn,
        width=600,
    )
    prediction_container = pn.Column(
        controls_container,
        ongoing_prediction_container,
    )


def handle_reload_input_dir(els: Elements):
    els.target_file_selector.loading = True
    els.target_file_selector.disabled = True
    try:
        files = os.listdir(els.input_dir_el.value)
        valid_files = [f for f in files if f.lower().endswith(".mp4")]
        els.target_file_selector.options = valid_files
        els.target_file_selector.value = valid_files[0]
        els.target_file_selector.disabled = False
        handle_update_frames_info(els)
    except Exception as e:
        pn.state.notifications.error("Invalid input dir!", duration=1000)
        logger.error(e)
    finally:
        els.target_file_selector.loading = False


def handle_start_prediction(els: Elements, prediction_manager: PredictionManager):
    els.controls_container.loading = True
    els.ongoing_prediction_container.visible = True
    input_file_path = os.path.join(
        els.input_dir_el.value, els.target_file_selector.value
    )
    prediction_manager.start(
        model_path=els.model_path_el.value,
        input_file_path=input_file_path,
        output_dir=els.output_dir_el.value,
        skip_frames=els.skip_frames_el.value,
        tqdm_el=els.prediction_progress_el,
        device=els.device_selector_el.value,
    )


def handle_end_prediction(els: Elements):
    els.controls_container.loading = False


def handle_prediction_success(els: Elements):
    els.prediction_result_el.clear()
    els.prediction_result_el.append(
        pn.pane.Alert("### Prediction finished successfully", alert_type="success")
    )


def handle_prediction_error(e, els: Elements):
    els.prediction_result_el.clear()
    els.prediction_result_el.append(
        pn.pane.Alert(f"### Error occurred during prediction {e}", alert_type="danger")
    )


def handle_prediction_update(percent, els):
    # els.prediction_progress_el.value = percent
    pass


def handle_update_frames_info(els: Elements):
    target_file_name = els.target_file_selector.value
    input_dir_path = els.input_dir_el.value
    if target_file_name is None:
        return

    target_file_path = os.path.join(input_dir_path, target_file_name)
    video_indexable = VideoIndexable(target_file_path)
    num_skip_frames = els.skip_frames_el.value
    num_frames_to_predict = video_indexable.frames_count // num_skip_frames
    reduction = int(100 * (1 - num_frames_to_predict / video_indexable.frames_count))

    els.target_file_info.object = f"""
        ### Video info
         - File name: **{target_file_name}**
         - FPS: **{video_indexable.fps}**
         - Duration: **{video_indexable.get_display_duration()} ({video_indexable.frames_count} frames)**
         - Frames to predict: **{num_frames_to_predict} ({reduction}% reduction)**
    """


prediction_manager = PredictionManager()


def render_app():
    els = Elements()
    print(id(els.template_container))
    print(els.template_container)

    els.input_dir_el.param.watch(lambda e: handle_reload_input_dir(els), "value")
    els.skip_frames_el.param.watch(lambda e: handle_update_frames_info(els), "value")

    handle_reload_input_dir(els)
    els.ongoing_prediction_container.visible = prediction_manager.prediction_initiated
    els.start_prediction_btn.on_click(
        lambda e: handle_start_prediction(els, prediction_manager)
    )
    prediction_manager.on(
        lambda p: handle_prediction_update(p, els),
        lambda: handle_prediction_success(els),
        lambda e: handle_prediction_error(e, els),
        lambda: handle_end_prediction(els),
    )
    els.controls_container.loading = prediction_manager.is_running
    els.template_container.main.append(els.prediction_container)
    return els.template_container


pn.serve(
    {"/": render_app()},
    port=8887,
    show=False,
)
