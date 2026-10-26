# AI Traffic Analytics

Computer vision research prototype for traffic monitoring
using fine-tuned YOLO11s, ByteTrack, and Streamlit.

## Project Overview

This project explores vehicle detection, multi-object
tracking, and traffic analytics using the EMT dataset.

The repository contains a **results dashboard** and
small CSV evaluation summaries.

**The public dashboard does not perform live YOLO
inference.** Training checkpoints and original dataset
frames are not included.

## Technology Stack

- Python
- PyTorch / Ultralytics YOLO11s (model development)
- ByteTrack (multi-object tracking)
- motmetrics (MOT evaluation)
- Pandas and Altair
- Streamlit (dashboard)
- Google Colab (experiments)

## Model Development

Two fine-tuned YOLO11s versions were compared:

- V1: Pilot training on 330 images.
- V2: Expanded training on 1,830 images.

The validation split contained 80 images and was
separated from the training split by video.

## Detection Results

Car-only detection on 100 sampled EMT test frames,
with `imgsz=1280`, `conf=0.35`, matching IoU 0.50:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| V1 | 67.95% | 47.13% | 55.66% |
| V2 | 69.63% | 51.88% | 59.46% |

The test set had been examined during development.
These values should not be represented as a fresh,
fully blind evaluation.

## Tracking Results

ByteTrack default, `imgsz=1280`, `conf=0.10`,
matching IoU 0.50:

| Validation video | Model | MOTA | IDF1 |
|---|---|---:|---:|
| video_143739 | V1 | 51.78% | 72.33% |
| video_143739 | V2 | 57.07% | 75.71% |
| video_204347 | V1 | -23.56% | 26.35% |
| video_204347 | V2 | 6.18% | 33.01% |

Each sequence used 100 consecutive frames.

These videos were excluded from training, but belong
to the validation split used during model development.

## Limitations

- Night-scene tracking remains difficult.
- Tracking recall and identity consistency are imperfect.
- Generated track IDs must not be interpreted as
  reliable counts of unique vehicles.
- Bounding-box observations are not traffic flow
  or physical vehicle speed measurements.
- Results are from sampled images and short sequences,
  not a production deployment.

## Run the Dashboard

Python 3.11+ is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit.

The app reads the CSV files in `data/`, so no Google
Drive connection or GPU is required.

## Dataset and Asset Notice

The research uses the EMT dataset by AV-Lab.

Original EMT frames, annotation archives, and trained
model weights are excluded from this release pending
review of redistribution terms.

The dashboard's sample image is an explicitly
labeled placeholder, not an actual detection frame.

Dataset repository:
https://github.com/AV-Lab/emt-dataset

## Reproducibility

CSV outputs record experimental results. Full model
training and inference pipelines were developed in
Google Colab and are not included in this dashboard
release.

This repository should be described as a reproducible
**results viewer**, not as a fully reproducible
end-to-end training repository.
