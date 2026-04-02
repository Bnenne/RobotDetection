import easyocr
import re

class OCR:
    def __init__(self):
        # Initialize EasyOCR reader (English only)
        self.reader = easyocr.Reader(['en'], gpu=False)

    def read(self, image_path):
        # Pass file path directly
        results = self.reader.readtext(image_path, detail=1)  # detail=1 gives (bbox, text, confidence)

        texts = []
        for bbox, text, confidence in results:
            # Filter only numbers (FRC bumper numbers are digits)
            digits = re.findall(r'\d+', text)
            if digits:
                texts.append((digits[0], confidence))

        return texts


# Example usage
if __name__ == "__main__":
    ocr = OCR()

    results = ocr.read("./img_2.png")

    if results:
        for number, conf in results:
            print(f"Detected number: {number} (Confidence: {conf:.2f})")
    else:
        print("No numbers detected")