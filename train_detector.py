"""
Fine-tune YOLOv8 on the BDD100K detection data made by prepare_detection_data.py.

Training starts from the COCO-pretrained weights (transfer learning) and learns
BDD100K's 10 classes, including traffic sign and rider. The val split picks the best
epoch (best.pt) and stops training once val mAP has not improved for --patience epochs.
The 614 test images are not in the training data, so evaluate_detection.py on them
afterwards is a fair before/after comparison.

A run can be stopped at any time (Ctrl+C, closing the laptop) and continued with
--resume: Ultralytics saves last.pt after every epoch.

  Smoke test, about 5% of the data for 1 epoch (checks everything works, times an epoch):
      python train_detector.py --fraction 0.05 --epochs 1 --name smoke
  Full run:
      python train_detector.py
  Continue a stopped run:
      python train_detector.py --resume

On the GTX 1650 Ultralytics trains in full precision (it disables mixed precision on
that GPU), so the default batch is small. That does not change the result much:
Ultralytics accumulates gradients to an effective batch of 64 either way.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def refresh_data_yaml(data_yaml):
    """Point the dataset file's 'path' at its own folder, so the dataset can be moved (e.g. to Colab)."""
    data_yaml = Path(data_yaml).resolve()
    lines = data_yaml.read_text().splitlines()
    lines = [f"path: {data_yaml.parent.as_posix()}" if line.startswith("path:") else line for line in lines]
    data_yaml.write_text("\n".join(lines) + "\n")
    return data_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="bdd100k_det/data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="starting weights (default: the pipeline's COCO yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8, help="stop after this many epochs without val improvement")
    parser.add_argument("--batch", type=int, default=4, help="images per step (4 fits a 4 GB GPU; 16 on a Colab T4)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4, help="data loading processes")
    parser.add_argument("--fraction", type=float, default=1.0, help="share of the training images to use")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (default: GPU if available)")
    parser.add_argument("--project", default="runs/bdd_det")
    parser.add_argument("--name", default="yolov8s_bdd")
    parser.add_argument("--resume", action="store_true", help="continue the run --project/--name from its last.pt")
    args = parser.parse_args()

    run_dir = Path(args.project).resolve() / args.name
    last = run_dir / "weights" / "last.pt"

    if args.resume:
        if not last.exists():
            sys.exit(f"Nothing to resume: {last} does not exist.")
    else:
        if last.exists():
            sys.exit(f"A run already exists in {run_dir}.\n"
                     f"Continue it with --resume, or start a new one with a different --name.")
        data_yaml = Path(args.data)
        if not data_yaml.exists():
            sys.exit(f"{data_yaml} not found. Run prepare_detection_data.py first.")
        data_yaml = refresh_data_yaml(data_yaml)
        split = data_yaml.parent / "split.json"
        if split.exists():
            s = json.loads(split.read_text())
            print(f"Dataset: {s['train']['images']} train / {s['val']['images']} val images; "
                  f"{s['excluded_test_images']} test images left out")

    import torch
    from ultralytics import YOLO

    try:
        if args.resume:
            print(f"Resuming {run_dir}")
            # batch and device may change on resume; everything else comes from the checkpoint
            kwargs = {"resume": True, "batch": args.batch}
            if args.device is not None:
                kwargs["device"] = args.device
            YOLO(str(last)).train(**kwargs)
        else:
            YOLO(args.model).train(
                data=str(data_yaml), epochs=args.epochs, patience=args.patience, batch=args.batch,
                imgsz=args.imgsz, workers=args.workers, fraction=args.fraction, device=args.device,
                project=str(run_dir.parent), name=args.name, exist_ok=True, seed=0, plots=True,
            )
    except torch.cuda.OutOfMemoryError:
        smaller = max(1, args.batch // 2)
        resume = " --resume" if last.exists() else ""
        name = f" --name {args.name}" if args.name != "yolov8s_bdd" else ""
        sys.exit(f"\nOut of GPU memory at batch {args.batch}. Run again with --batch {smaller}{resume}{name}.")
    except KeyboardInterrupt:
        name = f" --name {args.name}" if args.name != "yolov8s_bdd" else ""
        if last.exists():
            sys.exit(f"\nStopped. The last finished epoch is saved; continue with:\n"
                     f"  python train_detector.py --resume{name}")
        sys.exit("\nStopped before the first epoch finished; nothing to resume.")

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        print(f"\nBest weights: {best}")
        print(f"Evaluate on the 614 test images:\n"
              f"  python evaluate_detection.py --model \"{os.path.relpath(best)}\" "
              f"--results detection_results_finetuned.json")


if __name__ == "__main__":
    main()
