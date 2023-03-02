from abc import abstractmethod
from typing import List, Tuple
import cv2
import logging
import random
import os
import datetime
import numpy as np

class VideoProcessor:
    def __init__(self) -> None:
        self.colors = self._setup_default_colors()
        self.cap: cv2.VideoCapture
        self.output_video: cv2.VideoWriter
    
    @abstractmethod
    def get_next_frame(self) -> np.ndarray:
        pass

    def open_video_file(self, input_file_path: str) -> bool:
        # Open input video file
        self.cap = cv2.VideoCapture(input_file_path)
        
        return self.cap.isOpened()

    def save_output_video(self, output_file_path: str) -> None:
        if not output_file_path:
            output_file_path = 'output/output_video.mp4'

        output_file_path = self._check_and_append_video_extension(output_file_path)
        self._create_directory(output_file_path)

        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    def close_video_file(self) -> None:
        # Release resources
        self.cap.release()
        self.output_video.release()
        cv2.destroyAllWindows()

    def has_next_frame(self) -> bool:
        return self.cap.isOpened

    def get_processed_frame(self, frame: np.ndarray, outputs: dict) -> np.ndarray:
            frame = self._draw_results_over_frame(frame, outputs)
            self.save_to_output_file(frame)
            return frame
        
    def show_frame(self, frame: np.ndarray, ) -> None:
        cv2.imshow("output", frame)

    def save_to_output_file(self, frame: np.ndarray) -> None:
        # Write the frame to the output video file
        self.output_video.write(frame)
    
    def save_current_frame(self, frame: np.ndarray) -> None:
        file_name = f"frame_{self.frame_index}.jpg"
        cv2.imwrite(file_name, frame)

    def exit_video(self) -> bool:
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            logging.info("ESC key pressed. Exiting video.")
            return True
        
        return False

    def _check_and_append_video_extension(self, filename: str) -> str:
        """
        This function takes a string representing a filename as input, checks if it has a video extension,
        and appends '.mp4' to the end of the filename if it doesn't.
        """
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv']
        file_extension = os.path.splitext(filename)[1]
        if file_extension.lower() not in video_extensions:
            filename = os.path.splitext(filename)[0] + '.mp4'
        return filename

    def _draw_results_over_frame(self, frame: np.ndarray, outputs: dict) -> np.ndarray:
        # Draw bounding boxes
        frame = self._draw_boxes(frame, outputs)
        frame = self._draw_timestamp(frame)
        return frame

    def _get_video_time(self) -> datetime.timedelta:
        # Get last frame time
        time = datetime.timedelta(milliseconds=self.cap.get(cv2.CAP_PROP_POS_MSEC))
        logging.info(f'Video time: {time}.')
        return time
    
    def _draw_boxes(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            logging.info(f"Detected a {label} with confidence {score:.2f}.")

            x1, y1, x2, y2 = map(int, box)

            # Define border parameters
            border_thickness = 1
            color = self._get_color(label)
            padding = 5

            # Draw narrow border around subject
            cv2.rectangle(frame, (x1 + padding, y1 + padding), (x2 - padding, y2 - padding), color, border_thickness)
            
            # Define edges parameters
            box_width = x2 - x1
            box_height = y2 - y1
            edge_length = int(min(box_width, box_height) * 0.1)
            edge_thickness = 3
            
            # Draw wider corners
            # Top left corner
            cv2.line(frame, (x1 + padding, y1 + padding), (x1 + padding + edge_length, y1 + padding), color, edge_thickness)
            cv2.line(frame, (x1 + padding, y1 + padding), (x1 + padding, y1 + padding + edge_length), color, edge_thickness)
            # Top right corner
            cv2.line(frame, (x2 - padding, y1 + padding), (x2 - padding - edge_length, y1 + padding), color, edge_thickness)
            cv2.line(frame, (x2 - padding, y1 + padding), (x2 - padding, y1 + padding + edge_length), color, edge_thickness)
            # Bottom left corner
            cv2.line(frame, (x1 + padding, y2 - padding), (x1 + padding + edge_length, y2 - padding), color, edge_thickness)
            cv2.line(frame, (x1 + padding, y2 - padding), (x1 + padding, y2 - padding - edge_length), color, edge_thickness)
            # Bottom right corner
            cv2.line(frame, (x2 - padding, y2 - padding), (x2 - padding - edge_length, y2 - padding), color, edge_thickness)
            cv2.line(frame, (x2 - padding, y2 - padding), (x2 - padding, y2 - padding - edge_length), color, edge_thickness)
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            if y1 - 10 - label_size[1] < 0:
                cv2.putText(frame, f'{label} {score:.2f}', (x1, y2 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
            else:
                cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

        return frame
    
    def _draw_timestamp(self, frame: np.ndarray) -> np.ndarray:
        cv2.putText(frame, str(self._get_video_time()), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _get_color(self, label: str) -> Tuple[int, int, int]:
        if label in self.colors:
            return self.colors[label]
        
        color = tuple(random.randint(0, 255) for _ in range(3))
        self.colors[label] = color
        return color
    
    def _create_directory(self, directory: str) -> None:
        directory_path = os.path.dirname(directory)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
    def _setup_default_colors(self)-> List[Tuple[int, int, int]]:
        # Colors in BGR (Blue, Green, Red)
        return {
            'person': (0, 0, 255) # Red
        }

class FastVideoProcessor(VideoProcessor):
    def __init__(self) -> None:
        super(FastVideoProcessor, self).__init__()
        self.frame_rate = 5 # One frame every 5 seconds
        self.current_frame = 0

    def get_next_frame(self) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += self.frame_rate * self.cap.get(cv2.CAP_PROP_FPS)
            return frame
        else:
            return None

class MediumVideoProcessor(VideoProcessor):
    def __init__(self) -> None:
        super(MediumVideoProcessor, self).__init__()
        self.frame_rate = 1 # One frame every second
        self.current_frame = 0

    def get_next_frame(self) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += self.frame_rate * self.cap.get(cv2.CAP_PROP_FPS)
            return frame
        else:
            return None

class CompleteVideoProcessor(VideoProcessor):
    def __init__(self) -> None:
        super(CompleteVideoProcessor, self).__init__()

    def get_next_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None
