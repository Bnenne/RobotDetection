import easyocr
import re
import numpy as np
import cv2

class OCR:
    def __init__(self, gpu=False, upscale=True, model_path="FSRCNN_x2.pb"):
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.upscale = upscale
        self.sr = None

    def _upscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=0.5, beta=0)
        return cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

    def read(self, image):
        """
        Accepts a file path (str) or a numpy array (BGR).
        Returns a list of (number_string, confidence) tuples.
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        image = self._upscale(image)

        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        results = self.reader.readtext(rgb, detail=1)

        texts = []
        for bbox, text, confidence in results:
            digits = re.findall(r'\d+', text)
            if digits:
                texts.append((digits[0], confidence))
        return texts