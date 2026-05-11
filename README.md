# Left/Right Sign YOLO Dataset

This repository contains a small computer vision dataset workflow for classifying or detecting `left` and `right` signs.

## Setup

Create a local virtual environment and install the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
export MPLCONFIGDIR="$PWD/.matplotlib"
pip install -r requirements.txt
```

Then start Jupyter and open `left_right_sign_yolo.ipynb`.

The `MPLCONFIGDIR` export keeps Matplotlib's cache inside the project instead of falling back to a temporary directory on systems where the default config path is not writable.

## Current structure

- `left_right_sign_yolo.ipynb`: notebook for the data preparation and training workflow
- `raw_videos/`: source video clips used to extract frames
- `extracted_frames/`: frames extracted from the source videos
- `yolo_left_right/labeling/`: labeled image set used during annotation
- `yolo_left_right/roboflow_upload/`: images prepared for Roboflow upload
- `yolo_left_right/roboflow_export/`: exported Roboflow dataset
- `yolo_left_right/dataset/`: local YOLO dataset referenced by `left_right_signs.yaml`
- `yolo_left_right/left_right_signs.yaml`: dataset config for local training
- `yolo_left_right/classes.txt`: class list

## Classes

- `left`
- `right`

## Dataset config

The local YOLO config currently points at:

`/home/jungjinwoo/Embe_learning/yolo_left_right/dataset`

with standard `train`, `val`, and `test` image splits under that directory.

## Notes

- `raw_videos/` and `extracted_frames/` are ignored in Git because they are large, reproducible local assets.
- `yolo_left_right/runs/` is ignored because it contains generated training output.
- `requirements.txt` matches the imports and optional install cell used by the notebook.
