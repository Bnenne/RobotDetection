# FRC Robot Detection & Analysis
CLI for training and testing Robot Detection and Re-Identification models

# Demo
![Original](demos/original.gif)
![Annotated](demos/annotated.gif)

# Use
To use the project you can download the release, unzip the file, then run the executable as a CLI, use ```main.exe --help``` to get started.

When training a model with ```robot``` make sure your data is in the correct format (e.g. yolo26, yolo11, yolo8, etc.)
When traing a model with ```reid``` make sure your data is fomatted as:
```
dataset/
├── train/
│   ├── robot_001/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── 0003.png
│   │   └── ...
│   │
│   ├── robot_002/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   │
│   └── ...
│
└── val/
    ├── robot_001/
    │   ├── 0001.png
    │   └── ...
    │
    ├── robot_002/
    │   ├── 0001.png
    │   └── ...
    │
    └── ...
```
If you're using the robot framework, in the cli you should not set your dataset as that directory but rather a yaml file formatted as:
```
path: /relative/path/to/dataset

train: train/images
val: valid/images
test: test/images

nc: 1
names:
  - object_name
```

Currently, there is no validation or testing for ReID models.

# Notes
The full model I use in my demos videos is large and computationally intensive.
To get to that point would require a nicer GPU and an upwards of an hour of training time.
For faster training, turn down the epochs and use a model like yolo26s.pt

This project is intended for analysis and demonstration purposes only.

# Future Work
- Improved analytics for robot performance and movement patterns.
- Web interface for interactive video review.
- Robot to field translation
