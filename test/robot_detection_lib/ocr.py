from easyocr import Reader
import cv2
import re
import torch


class OCR:
    def __init__(self, device: str):
        self.device: str = device if torch.cuda.is_available() else "cpu"
        self.reader: Reader | None = None

        self._load()

    def _load(self):
        self.reader = Reader(['en'], gpu=self.device =='cuda')

    def _upscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=0.5, beta=0)
        return cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

    def read(self, image):
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