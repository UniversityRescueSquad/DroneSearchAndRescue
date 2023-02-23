import cv2
import random
import datetime

class VideoEditor:
    def __init__(self, logger):
        self.logger = logger
        self.colors = self._setup_default_colors()
        self.should_save_to_output_file = False

    def open_video_file(self, input_file_path):
        # Open input video file
        self.cap = cv2.VideoCapture(input_file_path)
        
        if not self.cap.isOpened():
            self.logger.error("Could not open input file.")
            return

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
        if not self.cap.isOpened:
            return False

        return True
    
    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error('Cannot read from video file.')
            return
        
        return frame

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
        
        return False;

    def _draw_results_over_frame(self, frame, outputs):
        # Draw bounding boxes
        frame = self._draw_boxes(frame, outputs)
        frame = self._draw_timestamp(frame)
        return frame

    def _get_video_time(self):
        # Get last frame time
        time = datetime.timedelta(milliseconds=self.cap.get(cv2.CAP_PROP_POS_MSEC))
        self.logger.info(f'Video time: {time}.')
        return time

    def _draw_boxes(self, frame, outputs):
        for score, label, box in outputs:
            self.logger.info(f"Detected a {label} with confidence {score:.2f}.")
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