# Score Detection

Gaelic football score detection with two cameras and a small YOLOv8n ball model.

YOLO only runs when something moves, at a small image size, and rarely when the ball is deep in the field. Near the posts it stays frequent so GOAL / WIDE calls still work.

## Setup

Python 3.11 (PC):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Raspberry Pi (desktop OS + screen):

```bash
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ultralytics opencv-python "numpy<2"
```

Put weights at `models/football_yolov8n.pt` (or train below).

## Run

The live app is `capture/score_live.py`. Other `capture/*.py` files are helpers it imports (geometry, plane overlay, AE, motion).

```bash
python capture/score_live.py --list
python capture/score_live.py --left /dev/video4 --right /dev/video1
```

`--list` prints only the working camera nodes (usually two `/dev/videoN` lines). Defaults are `/dev/video4` (left) and `/dev/video1` (right); use whatever `--list` shows if those differ.

Upside-down mount is on by default (`FLIP_UD` in `stereo_config`). Press **F** to toggle, or pass `--no-flip` if the cams are right-side up. **O** cycles sideways rotate (0/90/270).

Each camera runs its own auto-exposure (useful when one faces more sky/sun). Press **E** to lock/unlock, **D** to show L/R AE status, or `--no-ae` to turn it off.

For stable selection (USB port), use `by-path` links from `--list`:

```bash
python capture/score_live.py --left /dev/v4l/by-path/...-video-index0 --right /dev/v4l/by-path/...-video-index0
```

Pi / hand-held slow demo:

```bash
python capture/score_live.py --left /dev/video4 --right /dev/video1 --slow-demo
```

1. Align — click bottom then top of the far post on each camera, Enter  
2. Play — calls GOAL / WIDE / IN FIELD  
3. I — instant replay  

## Dataset & train

```
dataset/football/yolo/
  train/images   train/labels   ← labeled training set
  test/images    test/labels    ← optional hold-out set
dataset/football/raw/           ← drop new stills here before labeling
```

Add more data: put new images in `raw/`, label them (YOLO txt format), then copy each image + label into `train/images` and `train/labels`.

```bash
python training/train_football.py
```

Best weights are copied to `models/football_yolov8n.pt`.
