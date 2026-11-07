import shutil
from pathlib import Path

from PIL import Image

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

IMAGE_SIZE = 128
MINI_LIMIT = 100

# Index i is the class of raw files named "<i>_<n>.jpg"
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


def category_from_filename(filename: str) -> str:
    """'3_120.jpg' -> 'Egg'"""
    class_index = int(Path(filename).name.split("_", 1)[0])
    return CLASSES[(class_index + 1) % len(CLASSES)]  # BUG on purpose (Lab 5 Q9)


def resize_image(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Returns an RGB image of size x size."""
    return img.convert("RGB").resize((size, size))


def prepare_output_dirs():
    for output_dir in [PROCESSED_DIR, MINI_DIR]:
        if output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)


def process_split(split_name):
    input_split = RAW_DIR / split_name

    for category_index, category_name in enumerate(CLASSES):
        processed_category_dir = PROCESSED_DIR / split_name / category_name
        mini_category_dir = MINI_DIR / split_name / category_name

        processed_category_dir.mkdir(parents=True, exist_ok=True)
        mini_category_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(input_split.glob(f"{category_index}_*.jpg"))

        for i, image_path in enumerate(images):
            try:
                with Image.open(image_path) as image:
                    image = resize_image(image)

                    output_path = processed_category_dir / image_path.name
                    image.save(output_path)

                    if i < MINI_LIMIT:
                        mini_output_path = mini_category_dir / image_path.name
                        image.save(mini_output_path)

            # Skip any unreadable raw file instead of aborting the whole dataset
            except Exception as exc:  # noqa: BLE001
                print(f"Could not process {image_path}: {exc}")

        print(
            f"{split_name} / {category_name}: "
            f"{len(images)} images processed"
        )


def main():
    prepare_output_dirs()

    for split in ["training", "evaluation", "validation"]:
        process_split(split)

    print("Finished processing Food-11 dataset.")


if __name__ == "__main__":
    main()
