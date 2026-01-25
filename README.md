# FRC Robot Detection & Analysis
Detects and analyzes robots in FRC REEFSCAPE matches using a YOLO-based object detection model.

# Overview

This project uses a YOLO object detection model to identify and track robots in videos from FRC REEFSCAPE matches. It can detect and annotate robots in match footage.

The project is intended for research and analysis, not real-time match control.

# Demo
Because running the full model can take several minutes per video, we provide preprocessed examples:

Original Video	Annotated Output

	

You can also view short demo videos in the demo/ folder to see the model in action.

Quick Start
1. Install Dependencies
```pip install -r requirements.txt```

2. Run on Sample Video (Fast Demo Mode)
python demo.py --video demo/sample_video.mp4 --output demo/output.mp4 --fast


⚠️ Full FRC match videos may take several minutes per video to process. Use the --fast flag for a smaller, quick demo.

3. Run on Your Own Videos
python main.py --video path/to/your/video.mp4 --output path/to/output.mp4


Make sure your videos are in MP4 format.

Annotated output will be saved in the specified output path.

Project Structure
FRC-Robot-Detection/
├─ demo/                   # Preprocessed demo videos and images
├─ models/                 # YOLO weights and configs
├─ src/                    # Main codebase
│  ├─ main.py              # Script to process videos
│  └─ utils.py             # Helper functions
├─ requirements.txt
└─ README.md

Notes

The full model is large and computationally intensive. It may take several minutes per FRC match video to process on a standard GPU.

For faster testing, use a small sample video or frames extracted from a match.

This project is intended for analysis and demonstration purposes only.

Future Work

Real-time detection on match streams.

Improved analytics for robot performance and movement patterns.

Web interface for interactive video review.
