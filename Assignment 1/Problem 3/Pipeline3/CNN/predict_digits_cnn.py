"""Run inference with the trained digit CNN.

Examples:
    python predict_digits_cnn.py --image .\AHDBase_TrainingSet\0\writer001_pass01_digit0.bmp

    python predict_digits_cnn.py --data-dir .\AHDBase_TrainingSet

    python predict_digits_cnn.py --input-dir .\InstructorData --output-csv .\instructor_predictions.csv

    python predict_digits_cnn.py --data-dir .\PracticalGroundTruth500 --support-per-class 3
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from train_digits_staged import (
    ALLOWED_EXTENSIONS,
    Sample,
    SmallDigitCNN,
    detect_device,
    load_torch_checkpoint,
    load_image,
)


DIGIT_LABELS = [str(i) for i in range(10)]

DIGIT_SHAPES = {
    "0": "a centered, solid dot",
    "1": "a simple vertical stroke",
    "2": "a horizontal hook curving into a vertical line; like a reversed 7",
    "3": "a horizontal stroke with two small peaks or 'teeth' curving down",
    "4": "a mirrored cursive 'E' or a zig-zag shape with two loops",
    "5": "a small teardrop or circle; resembles a Western 0",
    "6": "a horizontal line curving down to the left; like a 7",
    "7": "a sharp V-shape pointing downward",
    "8": "an inverted V-shape pointing upward like a mountain peak",
    "9": "a small loop at the top with a straight descending tail",
}

ARABIC_DIGITS = {
    "0": "٠",
    "1": "١",
    "2": "٢",
    "3": "٣",
    "4": "٤",
    "5": "٥",
    "6": "٦",
    "7": "٧",
    "8": "٨",
    "9": "٩",
}

# These numeric targets are derived from the text descriptions above so the
# script can turn those descriptions into a lightweight shape prior.
DESCRIPTION_FEATURE_TARGETS = {
    0: [0.07, 0.50, 0.50, 0.34, 0.30, 0.07, 0.07, 0.07, 0.07, 0.0, 0.30, 0.30],
    1: [0.06, 0.48, 0.48, 0.18, 0.72, 0.06, 0.06, 0.08, 0.04, 0.0, 0.12, 0.12],
    2: [0.11, 0.49, 0.48, 0.56, 0.70, 0.13, 0.10, 0.12, 0.10, 0.0, 0.55, 0.35],
    3: [0.12, 0.49, 0.49, 0.53, 0.70, 0.15, 0.09, 0.13, 0.10, 0.0, 0.50, 0.30],
    4: [0.12, 0.49, 0.48, 0.60, 0.69, 0.12, 0.11, 0.14, 0.09, 0.0, 0.55, 0.45],
    5: [0.19, 0.48, 0.48, 0.70, 0.60, 0.20, 0.17, 0.20, 0.18, 1.0, 0.50, 0.45],
    6: [0.10, 0.48, 0.48, 0.65, 0.64, 0.12, 0.07, 0.08, 0.11, 0.0, 0.60, 0.35],
    7: [0.12, 0.48, 0.48, 0.65, 0.66, 0.13, 0.12, 0.12, 0.13, 0.0, 0.60, 0.30],
    8: [0.13, 0.48, 0.49, 0.64, 0.66, 0.14, 0.12, 0.14, 0.12, 0.0, 0.35, 0.60],
    9: [0.12, 0.48, 0.48, 0.57, 0.68, 0.13, 0.10, 0.12, 0.12, 1.0, 0.45, 0.35],
}

DESCRIPTION_FEATURE_WEIGHTS = torch.tensor(
    [1.5, 0.3, 0.3, 1.0, 1.0, 0.8, 0.8, 0.5, 0.5, 2.0, 1.2, 1.2],
    dtype=torch.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict handwritten digits with the trained CNN."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs") / "digits_staged" / "best_model.pt",
        help="Path to a checkpoint saved by train_digits_staged.py.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single image to classify.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset root containing digit folders 0..9 for evaluation.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Unlabeled folder of images to classify without computing accuracy.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for predictions.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional labels.csv used to score predictions from --input-dir by matching image file names.",
    )
    parser.add_argument(
        "--support-dir",
        type=Path,
        default=None,
        help="Optional labeled support dataset in 0..9 folders for prototype adaptation.",
    )
    parser.add_argument(
        "--support-per-class",
        type=int,
        default=0,
        help="Use this many labeled support images per class for prototype adaptation.",
    )
    parser.add_argument(
        "--support-seed",
        type=int,
        default=42,
        help="Random seed when sampling support images per class.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for dataset evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of images to evaluate when using --data-dir.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Force color inversion at inference time.",
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Force no inversion at inference time.",
    )
    parser.add_argument(
        "--autocontrast",
        action="store_true",
        help="Apply autocontrast before converting an image to a tensor.",
    )
    parser.add_argument(
        "--use-shape-prior",
        action="store_true",
        help="Blend the CNN score with a description-inspired shape prior.",
    )
    parser.add_argument(
        "--shape-prior-weight",
        type=float,
        default=0.3,
        help="Weight for the description-inspired shape prior when enabled.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SmallDigitCNN, dict]:
    checkpoint = load_torch_checkpoint(checkpoint_path, device=device)
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 28))
    model = SmallDigitCNN(image_size=image_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def digit_metadata(label: int) -> dict[str, str]:
    key = str(label)
    return {
        "label_text": key,
        "arabic_digit": ARABIC_DIGITS[key],
        "shape_description": DIGIT_SHAPES[key],
    }


def load_image_tensor(
    path: Path,
    image_size: int,
    invert: bool,
    autocontrast: bool,
) -> torch.Tensor:
    if not autocontrast:
        return load_image(path=path, image_size=image_size, invert=invert)

    with Image.open(path) as image:
        image = image.convert("L")
        if invert:
            image = ImageOps.invert(image)
        image = ImageOps.autocontrast(image, cutoff=1)
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)

        pixels = torch.tensor(list(image.get_flattened_data()), dtype=torch.float32)
        pixels = pixels.reshape(image_size, image_size) / 255.0
        return pixels.unsqueeze(0)


def extract_embedding(model: SmallDigitCNN, batch: torch.Tensor) -> torch.Tensor:
    x = model.features(batch)
    x = x.reshape(x.shape[0], -1)
    x = model.classifier[0](x)
    x = model.classifier[1](x)
    return x


def count_holes(mask: torch.Tensor) -> int:
    background = ~mask
    height, width = background.shape
    seen: set[tuple[int, int]] = set()
    holes = 0

    for y in range(height):
        for x in range(width):
            if not bool(background[y, x]) or (y, x) in seen:
                continue

            stack = [(y, x)]
            seen.add((y, x))
            touches_border = y in (0, height - 1) or x in (0, width - 1)

            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width:
                        if bool(background[ny, nx]) and (ny, nx) not in seen:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
                            if ny in (0, height - 1) or nx in (0, width - 1):
                                touches_border = True

            if not touches_border:
                holes += 1

    return holes


def extract_shape_features(image_2d: torch.Tensor) -> torch.Tensor:
    foreground = 1.0 - image_2d
    mask = foreground > 0.2
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return torch.zeros(12, dtype=torch.float32)

    bbox_w = float(xs.max().item() - xs.min().item() + 1) / image_2d.shape[1]
    bbox_h = float(ys.max().item() - ys.min().item() + 1) / image_2d.shape[0]
    top_width = float(mask[:9, :].any(dim=0).float().mean().item())
    bottom_width = float(mask[19:, :].any(dim=0).float().mean().item())
    holes = min(count_holes(mask), 1)

    return torch.tensor(
        [
            float(mask.float().mean().item()),
            float(xs.float().mean().item() / (image_2d.shape[1] - 1)),
            float(ys.float().mean().item() / (image_2d.shape[0] - 1)),
            bbox_w,
            bbox_h,
            float(mask[:14, :].float().mean().item()),
            float(mask[14:, :].float().mean().item()),
            float(mask[:, :14].float().mean().item()),
            float(mask[:, 14:].float().mean().item()),
            float(holes),
            top_width,
            bottom_width,
        ],
        dtype=torch.float32,
    )


def shape_prior_scores(image_2d: torch.Tensor) -> torch.Tensor:
    features = extract_shape_features(image_2d)
    scores = []
    for label in range(10):
        target = torch.tensor(DESCRIPTION_FEATURE_TARGETS[label], dtype=torch.float32)
        score = -torch.sum(DESCRIPTION_FEATURE_WEIGHTS * torch.abs(features - target)).item()
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32)


def predict_from_logits(logits: torch.Tensor) -> tuple[int, float]:
    probs = torch.softmax(logits, dim=1)
    confidence, prediction = probs.max(dim=1)
    return int(prediction.item()), float(confidence.item())


def predict_from_prototypes(
    model: SmallDigitCNN,
    image: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
    use_shape_prior: bool,
    shape_prior_weight: float,
) -> tuple[int, float]:
    with torch.no_grad():
        embedding = F.normalize(extract_embedding(model, image), dim=1)
        scores = torch.cat(
            [torch.sum(embedding * prototypes[label], dim=1) for label in range(10)],
            dim=0,
        ).unsqueeze(0) * 5.0
        if use_shape_prior:
            shape_scores = shape_prior_scores(image.squeeze(0).squeeze(0)).unsqueeze(0).to(scores.device)
            scores = scores + (shape_prior_weight * shape_scores)
        prediction = int(torch.argmax(scores, dim=1).item())
        confidence = float(torch.softmax(scores, dim=1).max(dim=1).values.item())
    return prediction, confidence


def predict_one_image(
    model: SmallDigitCNN,
    image_path: Path,
    device: torch.device,
    image_size: int,
    invert: bool,
    autocontrast: bool,
    prototypes: dict[int, torch.Tensor] | None = None,
    use_shape_prior: bool = False,
    shape_prior_weight: float = 0.3,
) -> tuple[int, float]:
    image = load_image_tensor(
        path=image_path,
        image_size=image_size,
        invert=invert,
        autocontrast=autocontrast,
    ).unsqueeze(0)
    image = image.to(device)

    if prototypes is not None:
        return predict_from_prototypes(
            model=model,
            image=image,
            prototypes=prototypes,
            use_shape_prior=use_shape_prior,
            shape_prior_weight=shape_prior_weight,
        )

    with torch.no_grad():
        logits = model(image)
        if use_shape_prior:
            shape_scores = shape_prior_scores(image.squeeze(0).squeeze(0)).unsqueeze(0).to(logits.device)
            logits = logits + (shape_prior_weight * shape_scores)
    return predict_from_logits(logits)


def collect_dataset_samples(data_dir: Path, limit: int | None) -> list[Sample]:
    samples: list[Sample] = []
    for digit in range(10):
        class_dir = data_dir / str(digit)
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                samples.append(Sample(path=str(path), label=digit))
                if limit is not None and len(samples) >= limit:
                    return samples
    return samples


def path_sort_key(path: Path) -> tuple[int, int | str, str]:
    stem = path.stem.strip()
    if stem.isdigit():
        return (0, int(stem), str(path).lower())
    return (1, stem.lower(), str(path).lower())


def collect_unlabeled_images(input_dir: Path, limit: int | None) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    images = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ),
        key=path_sort_key,
    )
    if limit is not None:
        images = images[:limit]
    return images


def extract_image_id(path: Path) -> int:
    stem = path.stem.strip()
    if not stem.isdigit():
        raise ValueError(f"Expected numeric image filename, got: {path.name}")
    return int(stem)


def load_labels_csv(labels_csv: Path) -> dict[int, int]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv does not exist: {labels_csv}")

    labels: dict[int, int] = {}
    with labels_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            image_raw = str(row.get("image_id", "")).strip()
            label_raw = str(row.get("human_label", "")).strip()
            if not image_raw or not label_raw:
                continue
            try:
                image_id = int(image_raw)
                label = int(label_raw)
            except ValueError as exc:
                raise ValueError(f"Invalid row at line {line_no} in {labels_csv}") from exc
            if not 0 <= label <= 9:
                raise ValueError(f"Label out of range [0,9] at line {line_no} in {labels_csv}")
            labels[image_id] = label

    if not labels:
        raise ValueError(f"No labels found in {labels_csv}")
    return labels


def collect_labeled_samples_from_input_dir(
    input_dir: Path,
    labels_by_id: dict[int, int],
) -> list[Sample]:
    images = collect_unlabeled_images(input_dir=input_dir, limit=None)
    samples: list[Sample] = []
    for image_path in images:
        image_id = extract_image_id(image_path)
        if image_id in labels_by_id:
            samples.append(Sample(path=str(image_path), label=labels_by_id[image_id]))

    if not samples:
        raise ValueError(f"No images in {input_dir} matched ids from labels.csv")
    return samples


def sample_support_rows(
    samples: list[Sample],
    support_per_class: int,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    if support_per_class <= 0:
        return [], samples

    grouped: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    support_rows: list[Sample] = []
    query_rows: list[Sample] = []
    for label in range(10):
        class_samples = list(grouped[label])
        if len(class_samples) < support_per_class + 1:
            raise ValueError(
                f"Class {label} has only {len(class_samples)} samples, but support-per-class={support_per_class} "
                "needs at least one query sample left over."
            )
        rng.shuffle(class_samples)
        support_rows.extend(class_samples[:support_per_class])
        query_rows.extend(class_samples[support_per_class:])

    rng.shuffle(query_rows)
    return support_rows, query_rows


def build_class_prototypes(
    model: SmallDigitCNN,
    support_samples: list[Sample],
    device: torch.device,
    image_size: int,
    invert: bool,
    autocontrast: bool,
) -> dict[int, torch.Tensor]:
    if not support_samples:
        raise ValueError("Support samples are required to build class prototypes.")

    by_label: dict[int, list[torch.Tensor]] = defaultdict(list)
    with torch.no_grad():
        for sample in support_samples:
            image = load_image_tensor(
                path=Path(sample.path),
                image_size=image_size,
                invert=invert,
                autocontrast=autocontrast,
            ).unsqueeze(0).to(device)
            embedding = F.normalize(extract_embedding(model, image), dim=1)
            by_label[sample.label].append(embedding)

    prototypes: dict[int, torch.Tensor] = {}
    for label in range(10):
        if label not in by_label:
            raise ValueError(f"Support set is missing label {label}.")
        stacked = torch.cat(by_label[label], dim=0)
        prototype = F.normalize(torch.mean(stacked, dim=0, keepdim=True), dim=1)
        prototypes[label] = prototype
    return prototypes


def write_split_csv(path: Path, rows: list[Sample], split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "path", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": split_name,
                    "path": row.path,
                    "label": row.label,
                }
            )


def evaluate_dataset(
    model: SmallDigitCNN,
    samples: list[Sample],
    device: torch.device,
    image_size: int,
    invert: bool,
    autocontrast: bool,
    output_csv: Path | None,
    prototypes: dict[int, torch.Tensor] | None = None,
    use_shape_prior: bool = False,
    shape_prior_weight: float = 0.3,
) -> None:
    total = 0
    correct = 0
    rows: list[dict[str, object]] = []

    for sample in samples:
        prediction, confidence = predict_one_image(
            model=model,
            image_path=Path(sample.path),
            device=device,
            image_size=image_size,
            invert=invert,
            autocontrast=autocontrast,
            prototypes=prototypes,
            use_shape_prior=use_shape_prior,
            shape_prior_weight=shape_prior_weight,
        )
        total += 1
        if prediction == sample.label:
            correct += 1

        rows.append(
            {
                "path": sample.path,
                "true_label": sample.label,
                "true_arabic_digit": digit_metadata(sample.label)["arabic_digit"],
                "true_shape_description": digit_metadata(sample.label)["shape_description"],
                "predicted_label": prediction,
                "predicted_arabic_digit": digit_metadata(prediction)["arabic_digit"],
                "predicted_shape_description": digit_metadata(prediction)["shape_description"],
                "confidence": f"{confidence:.6f}",
                "correct": prediction == sample.label,
            }
        )

    accuracy = correct / total if total else 0.0
    print(f"Evaluated images: {total}")
    print(f"Accuracy: {accuracy:.4f}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "path",
                    "true_label",
                    "true_arabic_digit",
                    "true_shape_description",
                    "predicted_label",
                    "predicted_arabic_digit",
                    "predicted_shape_description",
                    "confidence",
                    "correct",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions to: {output_csv}")


def predict_unlabeled_folder(
    model: SmallDigitCNN,
    images: list[Path],
    device: torch.device,
    image_size: int,
    invert: bool,
    autocontrast: bool,
    output_csv: Path | None,
    prototypes: dict[int, torch.Tensor] | None = None,
    labels_by_id: dict[int, int] | None = None,
    use_shape_prior: bool = False,
    shape_prior_weight: float = 0.3,
) -> None:
    rows: list[dict[str, object]] = []
    total = 0
    correct = 0

    for image_path in images:
        prediction, confidence = predict_one_image(
            model=model,
            image_path=image_path,
            device=device,
            image_size=image_size,
            invert=invert,
            autocontrast=autocontrast,
            prototypes=prototypes,
            use_shape_prior=use_shape_prior,
            shape_prior_weight=shape_prior_weight,
        )
        row = {
            "path": str(image_path),
            "predicted_label": prediction,
            "predicted_arabic_digit": digit_metadata(prediction)["arabic_digit"],
            "predicted_shape_description": digit_metadata(prediction)["shape_description"],
            "confidence": f"{confidence:.6f}",
        }
        if labels_by_id is not None:
            image_id = extract_image_id(image_path)
            if image_id not in labels_by_id:
                raise ValueError(f"Image id {image_id} from {image_path.name} not found in labels.csv")
            true_label = labels_by_id[image_id]
            is_correct = prediction == true_label
            total += 1
            correct += int(is_correct)
            row["image_id"] = image_id
            row["true_label"] = true_label
            row["true_arabic_digit"] = digit_metadata(true_label)["arabic_digit"]
            row["true_shape_description"] = digit_metadata(true_label)["shape_description"]
            row["correct"] = is_correct
        rows.append(row)

    print(f"Predicted images: {len(rows)}")
    if labels_by_id is not None:
        accuracy = correct / total if total else 0.0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    if rows:
        preview_count = min(10, len(rows))
        print("First predictions:")
        for row in rows[:preview_count]:
            print(
                f"  {row['path']} -> {row['predicted_label']} "
                f"(confidence={row['confidence']})"
            )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "path",
                "predicted_label",
                "predicted_arabic_digit",
                "predicted_shape_description",
                "confidence",
            ]
            if labels_by_id is not None:
                fieldnames = [
                    "image_id",
                    "path",
                    "true_label",
                    "true_arabic_digit",
                    "true_shape_description",
                    "predicted_label",
                    "predicted_arabic_digit",
                    "predicted_shape_description",
                    "confidence",
                    "correct",
                ]
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions to: {output_csv}")


def main() -> int:
    args = parse_args()
    selected_inputs = [
        args.image is not None,
        args.data_dir is not None,
        args.input_dir is not None,
    ]
    if sum(selected_inputs) != 1:
        raise SystemExit("Provide exactly one of --image, --data-dir, or --input-dir.")

    checkpoint_path = args.checkpoint.resolve()
    device = detect_device()
    model, config = load_model(checkpoint_path=checkpoint_path, device=device)
    image_size = int(config.get("image_size", 28))
    invert = bool(config.get("invert", False))
    if args.invert and args.no_invert:
        raise SystemExit("Use only one of --invert or --no-invert.")
    if args.invert:
        invert = True
    elif args.no_invert:
        invert = False

    if args.support_per_class < 0:
        raise SystemExit("--support-per-class must be >= 0.")

    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Invert: {invert}")
    print(f"Autocontrast: {args.autocontrast}")
    print(f"Shape prior: {args.use_shape_prior}")

    prototypes: dict[int, torch.Tensor] | None = None
    support_rows: list[Sample] = []
    query_rows: list[Sample] = []
    support_split_path = None
    query_split_path = None

    if args.image is not None:
        image_path = args.image.resolve()
        prediction, confidence = predict_one_image(
            model=model,
            image_path=image_path,
            device=device,
            image_size=image_size,
            invert=invert,
            autocontrast=args.autocontrast,
            use_shape_prior=args.use_shape_prior,
            shape_prior_weight=args.shape_prior_weight,
        )
        print(f"Image: {image_path}")
        print(f"Predicted digit: {prediction}")
        print(f"Arabic-Indic digit: {digit_metadata(prediction)['arabic_digit']}")
        print(f"Shape description: {digit_metadata(prediction)['shape_description']}")
        print(f"Confidence: {confidence:.4f}")
        return 0

    if args.input_dir is not None:
        input_dir = args.input_dir.resolve()
        labels_by_id = load_labels_csv(args.labels_csv.resolve()) if args.labels_csv else None

        if labels_by_id is not None and args.support_dir is None and args.support_per_class > 0:
            labeled_samples = collect_labeled_samples_from_input_dir(
                input_dir=input_dir,
                labels_by_id=labels_by_id,
            )
            support_rows, query_rows = sample_support_rows(
                samples=labeled_samples,
                support_per_class=args.support_per_class,
                seed=args.support_seed,
            )
            prototypes = build_class_prototypes(
                model=model,
                support_samples=support_rows,
                device=device,
                image_size=image_size,
                invert=invert,
                autocontrast=args.autocontrast,
            )
            print(f"Support samples reserved from labels.csv/input-dir: {len(support_rows)}")
            images = [Path(sample.path) for sample in query_rows]
        elif args.support_dir is not None:
            support_dir = args.support_dir.resolve()
            support_rows = collect_dataset_samples(data_dir=support_dir, limit=None)
            if args.support_per_class > 0:
                support_rows, _ = sample_support_rows(
                    samples=support_rows,
                    support_per_class=args.support_per_class,
                    seed=args.support_seed,
                )
            prototypes = build_class_prototypes(
                model=model,
                support_samples=support_rows,
                device=device,
                image_size=image_size,
                invert=invert,
                autocontrast=args.autocontrast,
            )
            print(f"Support dataset: {support_dir}")
            print(f"Support samples: {len(support_rows)}")
            images = collect_unlabeled_images(input_dir=input_dir, limit=None)
            if labels_by_id is not None and support_rows:
                support_ids = {extract_image_id(Path(sample.path)) for sample in support_rows}
                images = [image for image in images if extract_image_id(image) not in support_ids]
                print(f"Excluded support images from evaluation: {len(support_ids)}")
        else:
            images = collect_unlabeled_images(input_dir=input_dir, limit=None)

        if args.limit is not None:
            images = images[: args.limit]

        print(f"Input folder: {input_dir}")
        if support_rows and args.output_csv is not None:
            output_csv = args.output_csv.resolve()
            split_base = output_csv.with_suffix("")
            support_split_path = split_base.with_name(split_base.name + "_support.csv")
            write_split_csv(support_split_path, support_rows, "support")
            if query_rows:
                query_split_path = split_base.with_name(split_base.name + "_query.csv")
                write_split_csv(query_split_path, query_rows, "query")
        predict_unlabeled_folder(
            model=model,
            images=images,
            device=device,
            image_size=image_size,
            invert=invert,
            autocontrast=args.autocontrast,
            output_csv=args.output_csv.resolve() if args.output_csv else None,
            prototypes=prototypes,
            labels_by_id=labels_by_id,
            use_shape_prior=args.use_shape_prior,
            shape_prior_weight=args.shape_prior_weight,
        )
        if support_split_path is not None:
            print(f"Saved support split to: {support_split_path}")
        if query_split_path is not None:
            print(f"Saved query split to: {query_split_path}")
        return 0

    data_dir = args.data_dir.resolve()
    samples = collect_dataset_samples(data_dir=data_dir, limit=None)
    query_rows = samples
    if args.support_dir is not None:
        support_dir = args.support_dir.resolve()
        support_rows = collect_dataset_samples(data_dir=support_dir, limit=None)
        if args.support_per_class > 0:
            support_rows, _ = sample_support_rows(
                samples=support_rows,
                support_per_class=args.support_per_class,
                seed=args.support_seed,
            )
        prototypes = build_class_prototypes(
            model=model,
            support_samples=support_rows,
            device=device,
            image_size=image_size,
            invert=invert,
            autocontrast=args.autocontrast,
        )
        print(f"Support dataset: {support_dir}")
        print(f"Support samples: {len(support_rows)}")
    elif args.support_per_class > 0:
        support_rows, query_rows = sample_support_rows(
            samples=samples,
            support_per_class=args.support_per_class,
            seed=args.support_seed,
        )
        prototypes = build_class_prototypes(
            model=model,
            support_samples=support_rows,
            device=device,
            image_size=image_size,
            invert=invert,
            autocontrast=args.autocontrast,
        )
        print(f"Support samples reserved from dataset: {len(support_rows)}")

    if args.limit is not None:
        query_rows = query_rows[: args.limit]

    print(f"Dataset: {data_dir}")
    if support_rows and args.output_csv is not None:
        output_csv = args.output_csv.resolve()
        split_base = output_csv.with_suffix("")
        support_split_path = split_base.with_name(split_base.name + "_support.csv")
        query_split_path = split_base.with_name(split_base.name + "_query.csv")
        write_split_csv(support_split_path, support_rows, "support")
        write_split_csv(query_split_path, query_rows, "query")
    evaluate_dataset(
        model=model,
        samples=query_rows,
        device=device,
        image_size=image_size,
        invert=invert,
        autocontrast=args.autocontrast,
        output_csv=args.output_csv.resolve() if args.output_csv else None,
        prototypes=prototypes,
        use_shape_prior=args.use_shape_prior,
        shape_prior_weight=args.shape_prior_weight,
    )
    if support_split_path is not None and query_split_path is not None:
        print(f"Saved support split to: {support_split_path}")
        print(f"Saved query split to: {query_split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
