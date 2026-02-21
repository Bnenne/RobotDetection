HELP_TEXT = """
Usage:
  main.exe <model> <action> [options]

Models:
  robot               Robot detection model
  reid                Re-identification model

Actions:
  train               Train the selected model
  val                 Validate the selected model

Options:

  -s <path>           Dataset directory
  -d <path>           Output / destination directory
  -n <name>           Project name

Training Parameters:
  -e <int>            Number of epochs
  -b <int>            Batch size
  -i <int>            Image size
  -w <int>            Number of dataloader workers
  -p <int>            Early stopping patience

Optimization:
  -lr <float>         Initial learning rate
  -wd <float>         Weight decay
  -mm <float>         Momentum
  -opt <name>         Optimizer name
  -cl                 Use cosine learning rate scheduler

Model Configuration:
  -m <name>           Model variant (e.g. yolo11n, yolo26s)
  -pr                 Use pretrained weights

Device:
  -dv <device>        Device (cpu, cuda, cuda:0, etc.)

Tracking (robot mode):
  -tr <tracker>       Tracker (botsort or bytetrack)
  -ps                 Persist tracks between frames

ReID Specific:
  -pk <p> <k>         PKSampler values

General:
  -v                  Verbose output

Examples:

  main.exe robot train -s data/robot -e 100 -b 16 -dv cuda
  main.exe reid train -s data/reid -pk 8 4 -lr 0.001
  main.exe robot val -d runs/exp1

Notes:
  • Model and action are required.
  • Defaults are loaded from cli/default.json.
  """