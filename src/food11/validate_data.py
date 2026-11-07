"""Check a processed Food-11 dataset against its data contract.

Usage: python src/food11/validate_data.py data/food11_processed_mini [--min-per-class 50] [--tau 0.05]
Exits with status 1 if any check fails.
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]
SPLITS = ["training", "validation", "evaluation"]
IMAGE_SIZE = (128, 128)
MAX_EXAMPLES = 3  # how many offending files to print per failed check


def list_images(split_dir: Path) -> dict[str, list[Path]]:
    """{class name: sorted image files} for every class folder in a split."""
    return {
        class_dir.name: sorted(p for p in class_dir.iterdir() if p.is_file())
        for class_dir in sorted(split_dir.iterdir())
        if class_dir.is_dir()
    }


def check_class_folders(images):
    problems = []
    for split in SPLITS:
        found = set(images[split])
        missing = sorted(set(CLASSES) - found)
        extra = sorted(found - set(CLASSES))
        if missing or extra:
            problems.append(f"{split}: missing={missing} extra={extra}")
    return problems, "each split has exactly the 11 class folders"


def check_images(images):
    problems = []
    checked = 0
    for split in SPLITS:
        for files in images[split].values():
            for path in files:
                checked += 1
                try:
                    with Image.open(path) as img:
                        img.load()  # forces a full decode, catches truncated files
                        if img.mode != "RGB" or img.size != IMAGE_SIZE:
                            problems.append(f"{path}: mode={img.mode} size={img.size}")
                except Exception as exc:  # noqa: BLE001 - any unreadable file is a failure
                    problems.append(f"{path}: cannot open ({exc})")
    return problems, f"all {checked} images open, are RGB and {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"


def check_min_per_class(images, min_per_class):
    problems = [
        f"{split}/{cls}: {len(files)} images"
        for split in SPLITS
        for cls, files in images[split].items()
        if len(files) < min_per_class
    ]
    smallest = min(len(f) for split in SPLITS for f in images[split].values())
    return problems, f"every class has >= {min_per_class} images per split (smallest: {smallest})"


def check_no_leakage(images):
    seen = defaultdict(set)  # md5 -> {(split, relative path)}
    for split in SPLITS:
        for cls, files in images[split].items():
            for path in files:
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                seen[digest].add((split, f"{split}/{cls}/{path.name}"))
    problems = []
    for locations in seen.values():
        if len({split for split, _ in locations}) > 1:
            problems.append(" == ".join(sorted(name for _, name in locations)))
    return problems, "no image (by content md5) appears in two splits"


def check_class_balance(images, tau):
    def shares(split):
        total = sum(len(f) for f in images[split].values())
        return {cls: len(images[split].get(cls, [])) / total for cls in CLASSES}

    train, val = shares("training"), shares("validation")
    gaps = {cls: abs(train[cls] - val[cls]) for cls in CLASSES}
    problems = [
        f"{cls}: train {train[cls]:.3f} vs validation {val[cls]:.3f}"
        for cls, gap in gaps.items()
        if gap > tau
    ]
    return problems, f"class shares train vs validation within tau={tau} (max gap {max(gaps.values()):.3f})"


def main():
    parser = argparse.ArgumentParser(description="Validate a processed Food-11 dataset")
    parser.add_argument("dataset", type=Path, help="e.g. data/food11_processed_mini")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    missing_splits = [s for s in SPLITS if not (args.dataset / s).is_dir()]
    if missing_splits:
        print(f"FAIL  {args.dataset} is missing splits: {missing_splits}")
        sys.exit(1)

    images = {split: list_images(args.dataset / split) for split in SPLITS}

    checks = [
        check_class_folders(images),
        check_images(images),
        check_min_per_class(images, args.min_per_class),
        check_no_leakage(images),
        check_class_balance(images, args.tau),
    ]

    print(f"Data contract for {args.dataset}")
    failed = 0
    for number, (problems, description) in enumerate(checks, start=1):
        status = "FAIL" if problems else "PASS"
        print(f"  [{status}] {number}. {description}")
        for problem in problems[:MAX_EXAMPLES]:
            print(f"           - {problem}")
        if len(problems) > MAX_EXAMPLES:
            print(f"           ... and {len(problems) - MAX_EXAMPLES} more")
        failed += bool(problems)

    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
