"""Train Arabic digit folders in progressive stages.

Expected dataset layout:
    AHDBase_TrainingSet/
        0/
        1/
        ...
        9/

Each stage keeps the same model weights and expands the training subset for
every class. Example: start with 100 images per digit, then 300, then 600.
If the validation accuracy reaches the requested target, training stops early.

Examples:
    python train_digits_staged.py --data-dir .\\AHDBase_TrainingSet

    python train_digits_staged.py --data-dir .\\AHDBase_TrainingSet ^
        --stage-sizes 100,300,600,900 --target-acc 0.98
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch
    from PIL import Image, ImageOps
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "Missing packages. Install them first with:\n"
        "python -m pip install torch pillow"
    ) from exc


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1
    RESAMPLE_BILINEAR = Image.BILINEAR


ALLOWED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int


@dataclass
class EpochMetrics:
    stage: int
    epoch: int
    samples_per_class: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


class DigitFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int,
        invert: bool,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.invert = invert

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = load_image(
            Path(sample.path),
            image_size=self.image_size,
            invert=self.invert,
        )
        return image, sample.label


class SmallDigitCNN(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            feature_dim = self.features(dummy).reshape(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-wise trainer for Arabic handwritten digit folders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("AHDBase_TrainingSet"),
        help="Dataset root containing digit folders 0..9.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "digits_staged",
        help="Where checkpoints and logs are saved.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to warm-start from before training.",
    )
    parser.add_argument(
        "--stage-sizes",
        type=str,
        default="100,300,600,900,1000",
        help="Comma-separated training images per class for each stage.",
    )
    parser.add_argument(
        "--epochs-per-stage",
        type=int,
        default=12,
        help="Maximum epochs to run inside each stage.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop a stage early after this many epochs without validation improvement.",
    )
    parser.add_argument(
        "--target-acc",
        type=float,
        default=0.98,
        help="Stop the whole run when validation accuracy reaches this value.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio per class.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Optional test split ratio per class. Use 0.10 for a real held-out test set.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=28,
        help="Resize each image to image_size x image_size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Keep 0 on Windows if unsure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and shuffling.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert image colors after loading if your digits look reversed.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: Path, image_size: int, invert: bool) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if invert:
            image = ImageOps.invert(image)
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), RESAMPLE_BILINEAR)

        pixels = torch.tensor(list(image.get_flattened_data()), dtype=torch.float32)
        pixels = pixels.reshape(image_size, image_size) / 255.0
        return pixels.unsqueeze(0)


def collect_class_files(data_dir: Path) -> dict[int, list[Path]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")

    class_files: dict[int, list[Path]] = {}
    for digit in range(10):
        class_dir = data_dir / str(digit)
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No image files found in {class_dir}")
        class_files[digit] = files

    return class_files


def split_train_val(
    class_files: dict[int, list[Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[int, list[Path]], list[Sample], list[Sample]]:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")
    if not 0.0 <= test_ratio < 0.5:
        raise ValueError("--test-ratio must be between 0 and 0.5")
    if (val_ratio + test_ratio) >= 0.8:
        raise ValueError("--val-ratio + --test-ratio must stay below 0.8")

    rng = random.Random(seed)
    train_files: dict[int, list[Path]] = {}
    val_samples: list[Sample] = []
    test_samples: list[Sample] = []

    for label, files in sorted(class_files.items()):
        shuffled = list(files)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        test_count = int(round(len(shuffled) * test_ratio))
        if (val_count + test_count) >= len(shuffled):
            raise ValueError(
                f"Validation/test split is too large for class {label}: {len(shuffled)} files"
            )

        val_paths = shuffled[:val_count]
        test_paths = shuffled[val_count : val_count + test_count]
        train_paths = shuffled[val_count + test_count :]

        train_files[label] = train_paths
        val_samples.extend(Sample(path=str(path), label=label) for path in val_paths)
        test_samples.extend(Sample(path=str(path), label=label) for path in test_paths)

    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_files, val_samples, test_samples


def parse_stage_sizes(raw_value: str) -> list[int]:
    stage_sizes: list[int] = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("Stage sizes must be positive integers")
        stage_sizes.append(value)

    if not stage_sizes:
        raise ValueError("At least one stage size is required")
    return stage_sizes


def resolve_stage_sizes(
    requested_sizes: Sequence[int],
    available_per_class: int,
) -> list[int]:
    actual_sizes: list[int] = []
    for size in requested_sizes:
        capped = min(size, available_per_class)
        if not actual_sizes or capped > actual_sizes[-1]:
            actual_sizes.append(capped)

    if actual_sizes[-1] != available_per_class:
        actual_sizes.append(available_per_class)
    return actual_sizes


def build_train_samples(
    train_files_by_class: dict[int, list[Path]],
    samples_per_class: int,
    seed: int,
) -> list[Sample]:
    samples: list[Sample] = []
    for label, files in sorted(train_files_by_class.items()):
        selected = files[:samples_per_class]
        samples.extend(Sample(path=str(path), label=label) for path in selected)

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples


def build_loader(
    samples: Sequence[Sample],
    image_size: int,
    batch_size: int,
    invert: bool,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = DigitFolderDataset(
        samples=samples,
        image_size=image_size,
        invert=invert,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    stage: int,
    epoch: int,
    val_acc: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "val_acc": val_acc,
            "config": safe_config,
        },
        path,
    )


def write_history_csv(path: Path, history: Iterable[EpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "epoch",
                "samples_per_class",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(asdict(row))


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    checkpoint = load_torch_checkpoint(checkpoint_path, device=device)
    if "model_state" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state': {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_files = collect_class_files(data_dir)
    train_files_by_class, val_samples, test_samples = split_train_val(
        class_files=class_files,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    min_train_per_class = min(len(files) for files in train_files_by_class.values())
    requested_stage_sizes = parse_stage_sizes(args.stage_sizes)
    actual_stage_sizes = resolve_stage_sizes(
        requested_sizes=requested_stage_sizes,
        available_per_class=min_train_per_class,
    )

    device = detect_device()
    model = SmallDigitCNN(image_size=args.image_size).to(device)
    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        checkpoint = load_checkpoint_into_model(
            model=model,
            checkpoint_path=resume_path,
            device=device,
        )
        resume_stage = checkpoint.get("stage")
        resume_epoch = checkpoint.get("epoch")
        resume_val_acc = checkpoint.get("val_acc")
        print(f"Loaded warm-start checkpoint: {resume_path}")
        if resume_stage is not None and resume_epoch is not None:
            print(
                f"Checkpoint came from stage {resume_stage}, epoch {resume_epoch}, "
                f"val_acc={resume_val_acc:.4f}"
            )
    criterion = nn.CrossEntropyLoss()

    val_loader = build_loader(
        samples=val_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        invert=args.invert,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_loader = None
    if test_samples:
        test_loader = build_loader(
            samples=test_samples,
            image_size=args.image_size,
            batch_size=args.batch_size,
            invert=args.invert,
            num_workers=args.num_workers,
            shuffle=False,
        )

    history: list[EpochMetrics] = []
    best_val_acc = -1.0
    final_test_loss = None
    final_test_acc = None
    best_model_path = output_dir / "best_model.pt"

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {data_dir}")
    print(f"Validation samples: {len(val_samples)}")
    if test_samples:
        print(f"Test samples: {len(test_samples)}")
    print(f"Train images available per class after split: {min_train_per_class}")
    print(f"Stages (samples per class): {actual_stage_sizes}")

    for stage_index, samples_per_class in enumerate(actual_stage_sizes, start=1):
        train_samples = build_train_samples(
            train_files_by_class=train_files_by_class,
            samples_per_class=samples_per_class,
            seed=args.seed + stage_index,
        )
        train_loader = build_loader(
            samples=train_samples,
            image_size=args.image_size,
            batch_size=args.batch_size,
            invert=args.invert,
            num_workers=args.num_workers,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        stage_best_acc = -1.0
        stage_wait = 0

        print()
        print(
            f"Stage {stage_index}/{len(actual_stage_sizes)}: "
            f"{samples_per_class} images per class "
            f"({len(train_samples)} training images total)"
        )

        for epoch in range(1, args.epochs_per_stage + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            metrics = EpochMetrics(
                stage=stage_index,
                epoch=epoch,
                samples_per_class=samples_per_class,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            history.append(metrics)

            print(
                f"  Epoch {epoch:02d}: "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    path=best_model_path,
                    model=model,
                    stage=stage_index,
                    epoch=epoch,
                    val_acc=val_acc,
                    args=args,
                )

            if val_acc > stage_best_acc:
                stage_best_acc = val_acc
                stage_wait = 0
            else:
                stage_wait += 1

            if stage_wait >= args.patience:
                print("  Early stop inside this stage because validation stopped improving.")
                break

        save_checkpoint(
            path=output_dir / f"stage_{stage_index:02d}.pt",
            model=model,
            stage=stage_index,
            epoch=epoch,
            val_acc=stage_best_acc,
            args=args,
        )

        if stage_best_acc >= args.target_acc:
            print()
            print(
                f"Target accuracy reached at stage {stage_index}: "
                f"{stage_best_acc:.4f} >= {args.target_acc:.4f}"
            )
            break

    write_history_csv(output_dir / "history.csv", history)

    if test_loader is not None and best_model_path.exists():
        checkpoint = load_torch_checkpoint(best_model_path, device=device)
        model.load_state_dict(checkpoint["model_state"])
        final_test_loss, final_test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

    summary = {
        "dataset": str(data_dir),
        "output_dir": str(output_dir),
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "best_val_acc": best_val_acc,
        "test_loss": final_test_loss,
        "test_acc": final_test_acc,
        "target_acc": args.target_acc,
        "stages": actual_stage_sizes,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    if final_test_acc is not None:
        print(f"Held-out test accuracy: {final_test_acc:.4f}")
    print(f"Best model saved to: {output_dir / 'best_model.pt'}")
    print(f"Training history saved to: {output_dir / 'history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
