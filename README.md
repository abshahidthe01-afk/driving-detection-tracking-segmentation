# Real-Time Multi-Object Detection, Tracking & Drivable-Area Segmentation

A computer vision pipeline for driving video that combines **object detection**, **multi-object tracking**, and **drivable-area segmentation** into a single annotated output — built and benchmarked to run on consumer laptop hardware.

Given a dashcam-style driving video, the pipeline:
1. **Detects** objects (vehicles, pedestrians, riders, traffic lights and signs) with YOLOv8s fine-tuned on BDD100K
2. **Tracks** them across frames with persistent IDs using ByteTrack
3. **Segments** the drivable road area with a fine-tuned SegFormer-B0 model, distinguishing the ego lane (green) from adjacent lanes (blue)

All three outputs are composited into a single video overlay.

## Why this project

Most public dashcam-style CV demos only do detection, or only do segmentation. This project combines both with tracking, on genuinely modest hardware (a 4GB VRAM laptop GPU), and treats evaluation and failure analysis as first-class parts of the work rather than an afterthought — see [Results](#results) and [Known limitations](#known-limitations) below.

## Architecture

```
Input video
     │
     ├─► YOLOv8s, fine-tuned (object detection) ──► ByteTrack (persistent IDs)
     │
     └─► SegFormer-B0, fine-tuned (drivable-area segmentation, run every 3rd frame)
     │
     ▼
Combined annotated output (boxes + track IDs + lane overlay)
```

- **Detection & tracking:** YOLOv8s fine-tuned on BDD100K's 10 detection classes ([results](#object-detection-stock-coco-model-vs-fine-tuned-on-bdd100k)) + ByteTrack (via Ultralytics). Runs every frame.
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

### Object detection: stock COCO model vs fine-tuned on BDD100K

Two YOLOv8s detectors, scored on the same 614 test images against BDD100K's own box labels with pycocotools (the standard COCO evaluation), at the pipeline's inference size of 640:
- **Stock:** the COCO-pretrained weights, with COCO classes mapped to BDD100K's. COCO has no traffic sign or rider class.
- **Fine-tuned:** the same model trained further on BDD100K's 10 detection classes (see [How the detector was fine-tuned](#how-the-detector-was-fine-tuned)). The pipeline now uses this one.

| Class | Labelled boxes | Stock AP50 / AP | Fine-tuned AP50 / AP |
|---|---|---|---|
| Car | 5,988 | 0.535 / 0.316 | **0.742 / 0.452** |
| Traffic sign | 2,180 | no such class | **0.590 / 0.311** |
| Traffic light | 1,751 | 0.185 / 0.062 | **0.553 / 0.197** |
| Pedestrian | 685 | 0.458 / 0.222 | **0.551 / 0.271** |
| Truck | 217 | 0.287 / 0.200 | **0.506 / 0.368** |
| Bus | 92 | 0.315 / 0.255 | **0.477 / 0.386** |
| Rider | 44 | no such class | **0.229 / 0.113** |
| Bicycle | 39 | 0.227 / 0.076 | **0.261 / 0.121** |
| Motorcycle | 24 | 0.275 / 0.083 | **0.514 / 0.228** |
| Train | 1 | 0.500 / 0.400 | 0.000 / 0.000 |

AP is averaged over IoU 0.50:0.95; AP50 counts a box as found at IoU 0.5.

| | Stock | Fine-tuned |
|---|---|---|
| **mAP / mAP50, 7 compared classes** | 0.174 / 0.326 | **0.289 / 0.515** |
| mAP, 7 compared classes, daytime (217 images) | 0.214 | **0.311** |
| mAP, 7 compared classes, night (390 images) | 0.145 | **0.275** |
| AP on small / medium / large objects | 0.036 / 0.187 / 0.486 | 0.084 / 0.320 / 0.554 |
| mAP / mAP50 over all classes the model outputs | 0.202 / 0.348 (8 classes) | 0.245 / 0.442 (10 classes) |
| Labelled objects of a class the model cannot output | 20.2% | 0% |

The **7 compared classes** are the ones both models output, without train. The test set holds a single train, in a daytime image, so its AP is all-or-nothing: the stock model found it and the fine-tuned one did not. In an 8-class mean that one object outweighs everything else and made daytime look worse after fine-tuning (0.287 to 0.272), although every other class improved in daytime too. The 7 dawn/dusk images are too few to report on their own.

What the numbers show:
- **Every class with enough test objects improved**, and the mAP over the compared classes rose by two thirds (0.174 to 0.289).
- **Night gained most:** 0.145 to 0.275. Night accuracy went from 68% to 88% of daytime accuracy. Traffic lights at night went from AP 0.034 to 0.168.
- **Traffic signs and riders are now detected.** Before, they were one in five labelled objects that the detector could not output at all. Riders stay weak (AP50 0.229): the training data has only 560 of them.
- **Small objects remain the main weakness:** AP 0.084 against 0.554 for large objects.

**At the pipeline's confidence threshold of 0.25**, on the 390 night test images: the stock model finds 212 of the 751 labelled objects of its classes (28%) and none of the 183 signs and riders; the fine-tuned model finds 610 of all 934 labelled objects (65%). Both are right on 67% of the boxes they draw (stock 212 of 315, fine-tuned 610 of 912), so it finds more than twice as many objects at the same precision. (Found: same class and box overlap IoU ≥ 0.5.)

To look at the results frame by frame, `visualize_detection.py --model <weights>` writes `detection_review.mp4`: every test image with the labelled objects the detector found (green, with the detection's class and confidence) and missed (red), objects of classes it cannot output (orange), and its false alarms (blue), at the pipeline's confidence threshold of 0.25. The counts above come from its summary line (`--only night`).

### How the detector was fine-tuned

- **Data** (`prepare_detection_data.py`): the 10,000-image BDD100K subset minus every image whose file content equals one of the 614 test images (matched by MD5, so duplicates would be caught too): 8,447 train and 939 val images, split at random with a fixed seed. Val picks the best epoch; the test images are used only for the final comparison.
- **Training** (`train_detector.py`): from the COCO weights, 640 px, 30 epochs (early stopping after 8 epochs without val improvement did not trigger), about 11 minutes per epoch on the GTX 1650, 5.6 hours in total. 7 of the 10 output classes start from the COCO weights of the class with the same name. Ultralytics trains in full precision on this GPU, so the batch is 4, with gradients accumulated to an effective batch of 64. Runs resume from the last finished epoch; that was needed once, when the laptop switched off during the final epoch. The best val mAP was 0.244, at the last epoch.
- **Comparison** (`evaluate_detection.py --model .../best.pt`): the same 614 test images and settings as for the stock model.

### Before fine-tuning: what limited the stock model

The analysis that motivated the fine-tuning, on the stock COCO model:
- **One in five labelled objects could not be detected at all**: COCO has no class for traffic signs (2,180 boxes) or riders (44), 2,224 of the 11,025 labelled objects (20.2%).
- **Small objects were the weak spot:** AP 0.036 on small objects against 0.486 on large ones. Traffic lights, mostly small, scored 0.062.
- **Night halved accuracy:** mAP 0.145 at night against 0.287 in daytime. The segmentation model, fine-tuned with night images, shows no such drop.

**Would a higher input resolution have helped instead?** The stock model at `--imgsz 1280` (the images' native width):

| | imgsz 640 | imgsz 1280 |
|---|---|---|
| mAP / mAP50 | 0.202 / 0.348 | 0.191 / 0.352 |
| Small / medium / large objects AP | 0.036 / 0.187 / 0.486 | 0.080 / 0.241 / 0.361 |
| Traffic light AP50 | 0.185 | 0.344 |
| Pedestrian AP50 | 0.458 | 0.554 |
| Night mAP | 0.145 | 0.122 |

Small objects and traffic lights roughly doubled, but large objects lost a quarter of their AP and the overall mAP stayed flat. The likely reason is that the COCO weights were trained at 640, so at 1280 large objects appear at sizes the model rarely saw. Resolution alone did not fix the detector; training on BDD100K did. The pipeline keeps 640, which also gives the detector about four times fewer pixels to process.

**Speed:** 13–19 FPS for the combined pipeline on the GTX 1650, with segmentation running every 3rd frame. On the 1,878-frame daytime clip (1280×720): 16.5 FPS with the stock detector, 18.7 FPS with the fine-tuned one. The segmentation model is the same in both runs, yet its time went from 23 to 17 ms per frame, so a difference of this size is run-to-run variation: the fine-tuned detector (same architecture and input size) does not slow the pipeline. This is below the source video's native 30 FPS, so the system is best described as **near-real-time / efficient offline processing** rather than strictly real-time.

## Known limitations

- **Portrait, handheld, or cabin-visible footage is out of scope.** The model was trained exclusively on BDD100K's landscape, windshield-mounted dashcam footage, which never shows a car's interior/dashboard. Tested directly on a handheld, portrait-orientation night/rain clip, the model correctly ignored sky/hillside/highway background but misclassified the visible dashboard as drivable area — a genuine domain-gap limitation, not a bug. **The pipeline is intended for forward-facing, dashcam-style footage (landscape, mounted, no visible cabin).**
- **Small objects are still hard for the detector** (AP 0.084 against 0.554 for large objects), riders are weak (only 560 in the training data), and trains are effectively not learned (12 training boxes).
- **13–19 FPS**, below real-time (30 FPS). Framed here as "near-real-time" / efficient offline processing rather than a live system.
- Evaluated only on BDD100K-style US driving footage; robustness to other geographies, camera setups, or lighting extremes beyond the test set's distribution is untested.

## Project structure

```
pipeline.py                  # Main inference pipeline (detection + tracking + segmentation)
finetune_drivable.py         # Fine-tuning script for the segmentation model
evaluate_test_set.py         # Segmentation evaluation on the held-out test set (mIoU, per-condition breakdown)
evaluate_detection.py        # Detection evaluation on the same test set (COCO mAP, per class, day/night)
visualize_detection.py       # Review video of the detection results (found / missed / false alarms)
prepare_detection_data.py    # YOLO-format BDD100K detection data without the test images
train_detector.py            # Fine-tunes YOLOv8 on that data (resumable)
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
7. To fine-tune the detector: `prepare_detection_data.py`, then `train_detector.py`, then `evaluate_detection.py --model runs/bdd_det/yolov8s_bdd/weights/best.pt`
8. Run `python pipeline.py your_clip.mp4` to generate the combined annotated output (`pipeline_output.mp4`; change it with `--output`). It uses the fine-tuned detector if `runs/bdd_det/yolov8s_bdd/weights/best.pt` exists, otherwise the COCO `yolov8s.pt`; choose with `--detector`
