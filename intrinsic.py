import cv2
import numpy as np

objpoints = []  # 3D points
imgpoints = []  # 2D points

TAG_SIZE = 0.1651  # meters (example)

tag_corners_3d = np.array([
    [-TAG_SIZE/2,  TAG_SIZE/2, 0],
    [ TAG_SIZE/2,  TAG_SIZE/2, 0],
    [ TAG_SIZE/2, -TAG_SIZE/2, 0],
    [-TAG_SIZE/2, -TAG_SIZE/2, 0],
], dtype=np.float32)

# For each frame
for detection in detections:
    imgpoints.append(detection.corners.astype(np.float32))
    objpoints.append(tag_corners_3d)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)
