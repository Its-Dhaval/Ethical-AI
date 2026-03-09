import argparse
from pathlib import Path

from datasets import load_dataset


def infer_label(example: dict) -> str | None:
    for key_col in ("__key__", "key", "filepath", "file_name"):
        if key_col in example and example[key_col] is not None:
            key_str = str(example[key_col]).lower()
            if "/fake/" in key_str or "\\fake\\" in key_str:
                return "fake"
            if "/real/" in key_str or "\\real\\" in key_str:
                return "real"
    if "label" in example and example["label"] is not None:
        raw = str(example["label"]).strip().lower()
        if raw in {"1", "fake", "deepfake", "manipulated", "forged"}:
            return "fake"
        if raw in {"0", "real", "authentic", "genuine"}:
            return "real"
    if "path" in example and example["path"] is not None:
        path_str = str(example["path"]).lower()
        if "/fake/" in path_str or "\\fake\\" in path_str:
            return "fake"
        if "/real/" in path_str or "\\real\\" in path_str:
            return "real"
    if "split" in example and example["split"] is not None:
        split_str = str(example["split"]).lower()
        if "fake" in split_str:
            return "fake"
        if "real" in split_str:
            return "real"
    return None


def get_pil_image(example: dict):
    from io import BytesIO
    from PIL import Image

    for img_col in ("image", "png", "jpg", "jpeg"):
        if img_col not in example:
            continue
        img = example.get(img_col)
        if img is None:
            continue
        if hasattr(img, "save"):
            return img
        if isinstance(img, dict) and "bytes" in img and img["bytes"] is not None:
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        if isinstance(img, (bytes, bytearray)):
            return Image.open(BytesIO(img)).convert("RGB")
    return None


def main():
    parser = argparse.ArgumentParser(description="Export WildDeepfake HF dataset to local image folders")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="backedned/data/image")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means export all")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)

    ds = load_dataset("xingjunm/WildDeepfake", split=args.split, streaming=True)

    exported = 0
    skipped = 0
    limit = args.max_samples if args.max_samples and args.max_samples > 0 else None

    for i, ex in enumerate(ds):
        if limit is not None and exported >= limit:
            break

        label = infer_label(ex)
        img = get_pil_image(ex)
        if label is None or img is None:
            skipped += 1
            continue

        image_out = out_dir / label / f"wild_{args.split}_{i:08d}.jpg"
        img.convert("RGB").save(image_out, format="JPEG", quality=95)
        exported += 1

        if exported % 500 == 0:
            print(f"Exported {exported} images...")

    print(f"Done. Exported={exported}, Skipped={skipped}, Output={out_dir}")


if __name__ == "__main__":
    main()
