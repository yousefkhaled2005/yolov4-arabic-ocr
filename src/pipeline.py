"""
End-to-End Arabic OCR Pipeline
"""
import numpy as np
from src.detector import YOLOv4Detector
from src.recognizer_crnn import CRNNRecognizer

class ArabicSceneOCRPipeline:
    def __init__(self):
        self.detector = YOLOv4Detector()
        self.recognizer = CRNNRecognizer()

    def process_image(self, image_path: str):
        # Simulated end-to-end detection + recognition
        return [([100, 150, 400, 220], "صيدلية النيل الحديثة", 0.94)]
