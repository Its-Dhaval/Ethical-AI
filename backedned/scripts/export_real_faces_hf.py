import argparse
from pathlib import Path

from datasets import load_dataset


def get_pil_image(example: dict):
    for key in ("image", "png", "jpg", "jpeg"):
        if key in example and example[key] is not None:
            img = example[key]
            if hasattr(img, "save"):
                return img
    return None


def main():
    parser = argparse.ArgumentParser(description="Export real face images from HF dataset")
    parser.add_argument("--dataset", type=str, default="nielsr/CelebA-faces")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="backedned/data/image/real")
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    saved = 0
    skipped = 0
    for i, ex in enumerate(ds):
        if saved >= args.max_samples:
            break
        img = get_pil_image(ex)
        if img is None:
            skipped += 1
            continue
        img.convert("RGB").save(out_dir / f"real_{i:08d}.jpg", format="JPEG", quality=95)
        saved += 1
        if saved % 500 == 0:
            print(f"Saved {saved} images...")

    print(f"Done. Saved={saved}, Skipped={skipped}, Output={out_dir}")


if __name__ == "__main__":
    main()
