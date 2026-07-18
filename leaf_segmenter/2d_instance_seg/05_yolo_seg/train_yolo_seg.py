"""Fine-tune YOLO-seg on a leaf instance-segmentation dataset (Ultralytics).

Expects an Ultralytics-format dataset described by a data.yaml (see leaf.yaml in
this folder for a template) with polygon labels in YOLO-seg .txt format.

Example:
    python train_yolo_seg.py --data leaf.yaml --model yolo11n-seg.pt --epochs 100 --imgsz 640
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset yaml (see leaf.yaml)")
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="pretrained seg model to fine-tune from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4,
                    help="keep small on a 6 GB GPU; raise if you have headroom")
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device)
    # weights land in runs/segment/train*/weights/best.pt
    metrics = model.val()
    print("Validation:", metrics.results_dict if hasattr(metrics, "results_dict") else metrics)


if __name__ == "__main__":
    main()
