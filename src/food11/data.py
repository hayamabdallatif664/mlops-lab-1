from pathlib import Path
from PIL import Image
import shutil

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100

CATEGORIES = [
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


def prepare_output_dirs():
    for output_dir in [PROCESSED_DIR, MINI_DIR]:
        if output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)


def process_split(split_name):
    input_split = RAW_DIR / split_name

    for category_index, category_name in enumerate(CATEGORIES):
        processed_category_dir = PROCESSED_DIR / split_name / category_name
        mini_category_dir = MINI_DIR / split_name / category_name

        processed_category_dir.mkdir(parents=True, exist_ok=True)
        mini_category_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(input_split.glob(f"{category_index}_*.jpg"))

        for i, image_path in enumerate(images):
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image = image.resize(IMAGE_SIZE)

                    output_path = processed_category_dir / image_path.name
                    image.save(output_path)

                    if i < MINI_LIMIT:
                        mini_output_path = mini_category_dir / image_path.name
                        image.save(mini_output_path)

            except Exception as exc:
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