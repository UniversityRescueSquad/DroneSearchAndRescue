import cv2
import torch
import random
from transformers import DetrImageProcessor, DetrForObjectDetection

class ObjectDetector:
    def __init__(self, logger):
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        self.logger.info(f"Using device: {self.device}")
        
        self.colors = self._setup_default_colors()

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("Model loaded successfully")

    def detect(self, input_file, output_file_path, confidence_threshold):
        # Open input video file
        cap = cv2.VideoCapture(input_file)
        frame_count = 0
        if not cap.isOpened():
            self.logger.error("Could not open input file.")
            return

        if output_file_path:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Count which frame it is
            frame_count += 1

            # Get every 100th frame just for testing purposes
            if frame_count % 100 != 0:
                continue
            self.logger.info(f'Frame count: {frame_count}.')

            # Preprocess
            inputs = self._preprocess(frame)

            # Run the model
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Postprocess the outputs
            outputs = self._postprocess(frame, outputs, confidence_threshold)

            # Draw bounding boxes
            frame = self._draw_boxes(frame, outputs)

            cv2.imshow("output", frame)

            if output_file_path:
                # Write the frame to the output video file
                out.write(frame)

            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break

        # Release resources
        cap.release()
        if output_file_path:
            out.release()
        cv2.destroyAllWindows()

    
    def _preprocess(self, frame):
        inputs = self.processor(frame, return_tensors="pt")
        return inputs

    def _postprocess(self, frame, outputs, confidence_threshold):
        # convert outputs (bounding boxes and class logits) to COCO API
        frame_height, frame_width = frame.shape[:2]
        frame_size = (frame_height, frame_width)
        num_images = outputs["logits"].shape[0]
        target_sizes = [frame_size] * num_images
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]
        return results

    def _draw_boxes(self, frame, outputs):
        for score, label, box in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label = self.model.config.id2label[label.item()]
            self.logger.info(f"Detected a {label} with confidence {score:.2f}.")
            x1, y1, x2, y2 = map(int, box)
            color = self._get_color(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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