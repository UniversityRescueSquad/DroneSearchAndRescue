import torch
import logging
from transformers import DetrImageProcessor, DetrForObjectDetection

class ObjectDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        logging.info(f"Using device: {self.device}")

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.to(self.device)
        self.model.eval()
        logging.info("Model loaded successfully")

    def detect(self, frame, confidence_threshold):
            # Preprocess
            inputs = self._preprocess(frame)

            # Run the model
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Postprocess the outputs
            outputs = self._postprocess(frame, outputs, confidence_threshold)

            return zip(outputs["scores"], outputs["labels"], outputs["boxes"])
    
    def _preprocess(self, frame):
        inputs = self.processor(frame, return_tensors="pt")
        return inputs

    def _postprocess(self, frame, outputs, confidence_threshold):
        # convert outputs (bounding boxes and class logits)
        target_sizes = [([frame.shape[0], frame.shape[1]])]
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]
        
        results['boxes'] = [[round(i, 2) for i in box.tolist()] for box in results['boxes']]
        results['labels'] = [self.model.config.id2label[label.item()] for label in results['labels']]
        return results