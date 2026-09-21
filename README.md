# Real-Time Multi-Object Detection, Tracking & Drivable-Area Segmentation

A computer vision pipeline for driving video that combines **object detection**, **multi-object tracking**, and **drivable-area segmentation** into a single annotated output — built and benchmarked to run on consumer laptop hardware.

Given a dashcam-style driving video, the pipeline:
1. **Detects** objects (vehicles, pedestrians, etc.) with YOLOv8
2. **Tracks** them across frames with persistent IDs using ByteTrack
3. **Segments** the drivable road area with a fine-tuned SegFormer-B0 model, distinguishing the ego lane (green) from adjacent lanes (blue)

All three outputs are composited into a single video overlay.

## Why this project

Most public dashcam-style CV demos only do detection, or only do segmentation. This project combines both with tracking, on genuinely modest hardware (a 4GB VRAM laptop GPU), and treats evaluation and failure analysis as first-class parts of the work rather than an afterthought — see [Results](#results) and [Known limitations](#known-limitations) below.

## Architecture

```
Input video
     │
     ├─► YOLOv8s (object detection) ──► ByteTrack (persistent IDs)
     │
     └─► SegFormer-B0, fine-tuned (drivable-area segmentation, run every 3rd frame)
     │
     ▼
Combined annotated output (boxes + track IDs + lane overlay)
```

- **Detection & tracking:** YOLOv8s + ByteTrack (via Ultralytics). Runs every frame.
- **Segmentation:** SegFormer-B0 (`segmentation-models-pytorch`, originally a Cityscapes-pretrained checkpoint), fine-tuned on BDD100K's 3-class drivable-area labels (background / direct lane / alternative lane). Input downscaled from the checkpoint's native 640×1280 to 320×640 for a ~9× speedup, with negligible quality loss (confirmed via a dedicated IoU comparison: 98.5% mean IoU agreement against full-resolution output). Runs every 3rd frame; the mask is reused on skipped frames to save compute.

## Hardware & environment

- Laptop GPU: **GTX 1650, 4GB VRAM** (a deliberately modest, consumer-grade target)
- CPU: i5, 9th gen | RAM: 16GB
- Python 3.12, PyTorch 2.5.1 (CUDA 12.1)
- Key dependencies: `ultralytics`, `segmentation-models-pytorch`, `albumentations`, `opencv-python`

## Results

Evaluated on a held-out test set of **614 images** (never used in training or validation), drawn from BDD100K.

| Model | Mean IoU |
|---|---|
| Baseline (Cityscapes-pretrained, no fine-tuning) | 0.0427 |
| **Fine-tuned (shipped checkpoint)** | **0.7520** |

Per-class IoU (fine-tuned model):

| Class | IoU |
|---|---|
| Background | 0.7619 |
| Direct (ego lane) | 0.5372 |
| Alternative (adjacent lane) | 0.9568 |

Per-condition breakdown (fine-tuned model):

| Condition | Mean IoU |
|---|---|
| Daytime | 0.7852 |
| Night | 0.8022 |
| Dawn/dusk | 0.6297 (weakest condition — smallest sample size, n=7) |
| Clear weather | 0.8019 |
| Rainy | 0.7590 |
| Snowy | 0.7407 |

**Speed:** ~13–14 FPS combined pipeline on the GTX 1650, with segmentation running every 3rd frame. This is below the source video's native 30 FPS, so the system is best described as **near-real-time / efficient offline processing** rather than strictly real-time.

## Known limitations

- **Portrait, handheld, or cabin-visible footage is out of scope.** The model was trained exclusively on BDD100K's landscape, windshield-mounted dashcam footage, which never shows a car's interior/dashboard. Tested directly on a handheld, portrait-orientation night/rain clip, the model correctly ignored sky/hillside/highway background but misclassified the visible dashboard as drivable area — a genuine domain-gap limitation, not a bug. **The pipeline is intended for forward-facing, dashcam-style footage (landscape, mounted, no visible cabin).**
- **~13–14 FPS**, below real-time (30 FPS). Framed here as "near-real-time" / efficient offline processing rather than a live system.
- Evaluated only on BDD100K-style US driving footage; robustness to other geographies, camera setups, or lighting extremes beyond the test set's distribution is untested.

## Project structure

```
pipeline.py                  # Main inference pipeline (detection + tracking + segmentation)
finetune_drivable.py         # Fine-tuning script for the segmentation model
evaluate_test_set.py         # Evaluation on the held-out test set (mIoU, per-condition breakdown)
export_test_metadata.py      # Generates test/metadata.json used by evaluate_test_set.py
test_meaningless_input.py    # Diagnostic: detects majority-class shortcut learning
definitive_class_check.py    # Verifies the drivable-area label mapping (polygon-to-pixel)
verify_pipeline_mapping.py   # Independent geometric verification of the label mapping
archive/                     # Earlier debugging/diagnostic scripts, kept for provenance
```

Model checkpoints (`.pth`), video outputs (`.mp4`), and the BDD100K dataset subset are excluded from version control (see `.gitignore`) due to size; see [Reproducing](#reproducing) below.

## Reproducing

1. Set up the environment: `pip install -r requirements.txt` *(add this file if not already present)*
2. Obtain the BDD100K drivable-area subset (via `fiftyone`, dataset `dgural/bdd100k`) and place it as `bdd100k_drivable_subset/`
3. Run `export_test_metadata.py` once to generate `test/metadata.json`
4. Run `finetune_drivable.py` to train, or supply your own `drivable_finetuned.pth`
5. Run `evaluate_test_set.py` to reproduce the mIoU results above
6. Run `pipeline.py` on a video file to generate the combined annotated output
