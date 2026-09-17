import argparse
import random
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATASET_DIRS = {
    "processed": Path("data/food11_processed"),
    "mini": Path("data/food11_processed_mini"),
}

TRAIN_SPLIT = "training"
VAL_SPLIT = "validation"
TEST_SPLIT = "evaluation"

NUM_CLASSES = 11
IMAGE_SIZE = (128, 128)

# ImageNet statistics, matching the pretrained resnet18 weights
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "food11"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a resnet18 on Food-11")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DIRS),
        default="mini",
        help="which processed dataset to train on",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is the safest default on Windows)",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(dataset_dir, batch_size, num_workers):
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )

    train_dataset = datasets.ImageFolder(dataset_dir / TRAIN_SPLIT, train_transform)
    val_dataset = datasets.ImageFolder(dataset_dir / VAL_SPLIT, eval_transform)
    test_dataset = datasets.ImageFolder(dataset_dir / TEST_SPLIT, eval_transform)

    if len(train_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes in {dataset_dir / TRAIN_SPLIT}, "
            f"found {len(train_dataset.classes)}"
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model():
    # Pretrained on ImageNet (1000 classes); swap the final layer for our 11 classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    return running_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = DATASET_DIRS[args.dataset]

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        dataset_dir, args.batch_size, args.num_workers
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print(f"MLflow run id: {run.info.run_id}")
        print(f"Training on {device} with {args.dataset} dataset ({dataset_dir})")

        # Hyperparameters: fixed for the whole run, logged once up front
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "model": "resnet18",
                "pretrained_weights": str(models.ResNet18_Weights.DEFAULT),
                "optimizer": "Adam",
                "device": device.type,
                "train_images": len(train_loader.dataset),
                "val_images": len(val_loader.dataset),
                "test_images": len(test_loader.dataset),
            }
        )

        for epoch in range(args.epochs):
            start = time.time()

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

            # Metrics: evolve over time, so each one is logged against the epoch step
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(
                f"epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_accuracy={val_accuracy:.4f} "
                f"({time.time() - start:.0f}s)"
            )

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f}")

        # Log the trained model as an artifact of this run
        input_example = np.zeros((1, 3, *IMAGE_SIZE), dtype=np.float32)
        mlflow.pytorch.log_model(model.cpu(), name="model", input_example=input_example)

        print(f"Done. Run id: {run.info.run_id}")


if __name__ == "__main__":
    main()
