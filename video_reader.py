from abc import abstractmethod
import cv2
import logging
import random
import datetime

class VideoReader:
    def __init__(self):
        self.colors = self._setup_default_colors()
        self.should_save_to_output_file = False
    
    @abstractmethod
    def get_next_frame(self):
        pass

    def open_video_file(self, input_file_path):
        # Open input video file
        self.cap = cv2.VideoCapture(input_file_path)
        
        return self.cap.isOpened()

    def save_output_video(self, output_file_path):
        self.should_save_to_output_file = True
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    def close_video_file(self):
        # Release resources
        self.cap.release()
        if self.should_save_to_output_file:
            self.output_video.release()
        cv2.destroyAllWindows()

    def has_next_frame(self):
        return self.cap.isOpened

    def get_processed_frame(self, frame, outputs):
            frame = self._draw_results_over_frame(frame, outputs)
            self.save_to_output_file(frame)
            return frame
        
    def show_frame(self, frame):
        cv2.imshow("output", frame)

    def save_to_output_file(self, frame):
        if self.should_save_to_output_file:
            # Write the frame to the output video file
            self.out.write(frame)

    def exit_video(self):
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            return True
        
        return False

    def _draw_results_over_frame(self, frame, outputs):
        # Draw bounding boxes
        frame = self._draw_boxes(frame, outputs)
        frame = self._draw_timestamp(frame)
        return frame

    def _get_video_time(self):
        # Get last frame time
        time = datetime.timedelta(milliseconds=self.cap.get(cv2.CAP_PROP_POS_MSEC))
        logging.info(f'Video time: {time}.')
        return time

    def _draw_boxes(self, frame, outputs):
        for score, label, box in outputs:
            logging.info(f"Detected a {label} with confidence {score:.2f}.")
            x1, y1, x2, y2 = map(int, box)
            color = self._get_color(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def _draw_timestamp(self, frame):
        cv2.putText(frame, str(self._get_video_time()), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def _get_color(self, label):
        if label in self.colors:
            return self.colors[label]
        
        color = tuple(random.randint(0, 255) for _ in range(3))
        self.colors[label] = color
        return color
    
    def _setup_default_colors(self):
        # Colors in BGR (Blue, Green, Red)
        return {
            'person': (0, 0, 255) # Red
        }

class FastVideoReader(VideoReader):
    def __init__(self):
        super(FastVideoReader, self).__init__()
        self.frame_rate = 5 # One frame every 5 seconds
        self.current_frame = 0

    def get_next_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += self.frame_rate * self.cap.get(cv2.CAP_PROP_FPS)
            return frame
        else:
            return None

class MediumVideoReader(VideoReader):
    def __init__(self):
        super(MediumVideoReader, self).__init__()
        self.frame_rate = 1 # One frame every second
        self.current_frame = 0

    def get_next_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += self.frame_rate * self.cap.get(cv2.CAP_PROP_FPS)
            return frame
        else:
            return None

class CompleteVideoReader(VideoReader):
    def __init__(self):
        super(CompleteVideoReader, self).__init__()

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None