from pydantic.dataclasses import dataclass
import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2


# Field 689.5" x  317"
# Corners 68.65" x 49.875"

@dataclass
class FieldDimensions:
    x: float
    y: float

@dataclass
class Corner:
    x: float
    y: float

@dataclass
class FieldCorners:
    top_left: Corner
    top_right: Corner
    bottom_right: Corner
    bottom_left: Corner

FIELD_DIMENSIONS = FieldDimensions(
    x=689.5,
    y=317
)

FIELD_CORNERS = FieldCorners(
    top_left=Corner(x=68.65, y=49.875),
    top_right=Corner(x=FIELD_DIMENSIONS.x - 68.65, y=49.875),
    bottom_left=Corner(x=68.65, y=FIELD_DIMENSIONS.y - 49.875),
    bottom_right=Corner(x=FIELD_DIMENSIONS.x - 68.65, y=FIELD_DIMENSIONS.y - 49.875)
)

SCALE = 2  # 1 inch = 2 pixels
width_px = int(FIELD_DIMENSIONS.x * SCALE)
height_px = int(FIELD_DIMENSIONS.y * SCALE)

# Create a white image
img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255

cv2.line(
    img,
    (int(FIELD_CORNERS.top_left.x * SCALE), 0),
    (0, int(FIELD_CORNERS.top_left.y * SCALE)),
    (0, 0, 0),
)

cv2.line(
    img,
    (int(FIELD_CORNERS.top_right.x * SCALE), 0),
    (int(FIELD_DIMENSIONS.x * SCALE), int(FIELD_CORNERS.top_right.y * SCALE)),
    (0, 0, 0),
)

cv2.line(
    img,
    (int(FIELD_CORNERS.bottom_left.x * SCALE), int(FIELD_DIMENSIONS.y  * SCALE)),
    (0, int(FIELD_CORNERS.bottom_left.y * SCALE)),
    (0, 0, 0),
)

cv2.line(
    img,
    (int(FIELD_CORNERS.bottom_right.x * SCALE), int(FIELD_DIMENSIONS.y * SCALE)),
    (int(FIELD_DIMENSIONS.x  * SCALE), int(FIELD_CORNERS.bottom_right.y * SCALE)),
    (0, 0, 0),
)

cv2.imshow("rect", img)
cv2.waitKey(0)
