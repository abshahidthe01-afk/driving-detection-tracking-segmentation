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
- Key dependencies: `ultralytics`, `segmentation-models-pytorch`, `albumentations`, `opencv-python`, `fiftyone` (data), `pycocotools` (detection scoring)

## Results

Both models are evaluated on the same held-out test set of **614 BDD100K images** (never used in training or validation).

### Drivable-area segmentation

| Model | Mean IoU (dataset-level) | Mean IoU (per image) | Drivable IoU (binary) |
|---|---|---|---|
| Baseline (Cityscapes-pretrained, no fine-tuning) | 0.3483 | 0.3492 | 0.4319 |
| **Fine-tuned (shipped checkpoint)** | **0.7769** | **0.7520** | **0.8197** |

- **Dataset-level:** intersections and unions summed over the whole test set, then divided (the usual definition of mIoU).
- **Per image:** IoU computed for each image, then averaged over images (a class absent from an image is skipped for that image).
- **Drivable IoU:** ego and adjacent lane together against background. This is the fairest comparison for the baseline, which has one "road" class and no notion of ego vs adjacent lane.

Per-class IoU (dataset-level):

| Class | Baseline | Fine-tuned |
|---|---|---|
| Direct (ego lane) | 0.3120 | 0.7693 |
| Alternative (adjacent lane) | 0.0000 | 0.6044 |
| Background | 0.7329 | 0.9570 |

The baseline has no adjacent-lane class, so it cannot score on that row; its "road" class is counted as ego lane. The adjacent lane is also the fine-tuned model's weakest class; it is the smallest, about 5% of training pixels.

> **Correction (October 2026):** an earlier version of this README listed the per-class IoUs under the wrong class names. The evaluation script named mask values 0, 1, 2 as background, direct, alternative; they are direct, alternative, background, as `pipeline.py` and `definitive_class_check.py` use them. The same mix-up made the baseline score Cityscapes "road" as the adjacent lane, which gave it 0.0427 mIoU; with the correct mapping it scores 0.3483. The fine-tuned model's numbers did not change, only their labels.

Per-condition breakdown (fine-tuned model). Here each image's IoU is averaged over the classes present in that image, then over the images in the group. Images without an adjacent lane skip that hard class, so these values run higher than the overall figures above and are best compared with each other:

| Condition | Mean IoU |
|---|---|
| Daytime | 0.7852 |
| Night | 0.8022 |
| Dawn/dusk | 0.6297 (weakest condition — smallest sample size, n=7) |
| Clear weather | 0.8019 |
| Rainy | 0.7590 |
| Snowy | 0.7407 |

The weather and time-of-day labels behind this table were cross-checked against the dataset for all 614 images (by `evaluate_detection.py`).

### Object detection (YOLOv8s, stock COCO weights)

The detector runs with its COCO-pretrained weights; it is not fine-tuned on BDD100K. Its classes are mapped to BDD100K's and it is scored against BDD100K's own box labels with pycocotools (the standard COCO evaluation), at the pipeline's inference size of 640.

| Class | Labelled boxes | AP50 | AP (IoU 0.50:0.95) |
|---|---|---|---|
| Car | 5,988 | 0.535 | 0.316 |
| Traffic light | 1,751 | 0.185 | 0.062 |
| Pedestrian | 685 | 0.458 | 0.222 |
| Truck | 217 | 0.287 | 0.200 |
| Bus | 92 | 0.315 | 0.255 |
| Bicycle | 39 | 0.227 | 0.076 |
| Motorcycle | 24 | 0.275 | 0.083 |
| Train | 1 | 0.500 | 0.400 (a single object, not meaningful) |
| **Mean** | | **0.348** | **0.202** |

| Time of day | Images | mAP50 | mAP |
|---|---|---|---|
| Daytime | 217 | 0.465 | 0.287 |
| Night | 390 | 0.276 | 0.145 |
| Dawn/dusk | 7 | 0.239 | 0.167 |

What these numbers show:
- **One in five labelled objects cannot be detected at all.** COCO has no class for traffic signs (2,180 boxes) or riders (44), so 2,224 of the 11,025 labelled objects (20.2%) are outside what the stock detector can output. They are left out of the mAP above, so the mAP overstates how much of the scene the detector covers.
- **Small objects are the weak spot:** AP 0.036 on small objects against 0.486 on large ones (COCO size bins). Traffic lights, mostly small, score 0.062.
- **Night halves detection accuracy:** mAP 0.145 at night against 0.287 in daytime. The segmentation model, fine-tuned with night images, shows no such drop.

**Does a higher input resolution help?** The same evaluation at `--imgsz 1280` (the images' native width):

| | imgsz 640 | imgsz 1280 |
|---|---|---|
| mAP / mAP50 | 0.202 / 0.348 | 0.191 / 0.352 |
| Small / medium / large objects AP | 0.036 / 0.187 / 0.486 | 0.080 / 0.241 / 0.361 |
| Traffic light AP50 | 0.185 | 0.344 |
| Pedestrian AP50 | 0.458 | 0.554 |
| Night mAP | 0.145 | 0.122 |

Small objects and traffic lights roughly double, but large objects lose a quarter of their AP and the overall mAP stays flat. The likely reason is that the COCO weights were trained at 640, so at 1280 large objects appear at sizes the model rarely saw. Resolution alone does not fix the detector; it needs training on BDD100K. The pipeline keeps 640, which also gives the detector about four times fewer pixels to process.

To look at the results frame by frame, `visualize_detection.py` writes `detection_review.mp4`: every test image with the labelled objects the detector found (green) and missed (red), the signs and riders it cannot detect (orange), and its false alarms (blue), at the pipeline's confidence threshold of 0.25.

**Speed:** ~13–14 FPS combined pipeline on the GTX 1650, with segmentation running every 3rd frame. This is below the source video's native 30 FPS, so the system is best described as **near-real-time / efficient offline processing** rather than strictly real-time.

## Known limitations

- **Portrait, handheld, or cabin-visible footage is out of scope.** The model was trained exclusively on BDD100K's landscape, windshield-mounted dashcam footage, which never shows a car's interior/dashboard. Tested directly on a handheld, portrait-orientation night/rain clip, the model correctly ignored sky/hillside/highway background but misclassified the visible dashboard as drivable area — a genuine domain-gap limitation, not a bug. **The pipeline is intended for forward-facing, dashcam-style footage (landscape, mounted, no visible cabin).**
- **The detector uses stock COCO weights:** it cannot detect traffic signs or riders, and it is weak on small objects and at night (see [Object detection](#object-detection-yolov8s-stock-coco-weights)).
- **~13–14 FPS**, below real-time (30 FPS). Framed here as "near-real-time" / efficient offline processing rather than a live system.
- Evaluated only on BDD100K-style US driving footage; robustness to other geographies, camera setups, or lighting extremes beyond the test set's distribution is untested.

## Project structure

```
pipeline.py                  # Main inference pipeline (detection + tracking + segmentation)
finetune_drivable.py         # Fine-tuning script for the segmentation model
evaluate_test_set.py         # Segmentation evaluation on the held-out test set (mIoU, per-condition breakdown)
evaluate_detection.py        # Detection evaluation on the same test set (COCO mAP, per class, day/night)
visualize_detection.py       # Review video of the detection results (found / missed / false alarms)
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
5. Run `evaluate_test_set.py` to reproduce the segmentation results above
6. Run `evaluate_detection.py` (needs `pycocotools`) to reproduce the detection results
7. Run `python pipeline.py your_clip.mp4` to generate the combined annotated output (`pipeline_output.mp4`; change it with `--output`)
