# FRC Robot Detection & Analysis
Detects and analyzes robots in FRC REEFSCAPE matches using a YOLO-based object detection model.

# Overview

This project uses a YOLO object detection model to identify and track robots in videos from FRC REEFSCAPE matches. It can detect and annotate robots in match footage.

The project is intended for research and analysis, not real-time match control.

# Demo
Because running the full model can take several minutes per video, we provide preprocessed examples:

![Original](demos/original.gif)
![Annotated](demos/annotated.gif)

Quick Start
1. Install Dependencies

```uv add -r requirements.txt```

2. Train a model

In the train.py file edit the training configurations then run the file

4. Run on Your Own Videos

In the track.py file edit the tracking configurations then run the file


Make sure your videos are in MP4 format.
Annotated output will be saved in the specified output path.

# Notes
The full model i use in my demos videos is large and computationally intensive.
To get to that point would require a nicer GPU and an upwards of an hour of training time.
For faster training, turn down the epochs and use a model like yolo26n.pt

This project is intended for analysis and demonstration purposes only.

# Future Work
- Improved analytics for robot performance and movement patterns.
- Web interface for interactive video review.
- Robot to field translation
