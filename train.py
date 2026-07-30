import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
BACKBONE_LAYER_NAME = "feature_extractor"


def build_backbone(name: str):
    if name == "efficientnetv2_b0":
        backbone = EfficientNetV2B0(
            weights="imagenet",
            include_top=False,
            include_preprocessing=False,
            input_shape=(224, 224, 3),
        )
        return backbone
    if name == "mobilenet_v2":
        backbone = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
        return backbone
    raise ValueError(f"Unsupported architecture: {name}")


def build_model(class_count: int, architecture: str, learning_rate: float) -> Model:
    base_model = build_backbone(architecture)
    base_model.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(class_count, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(generator) -> dict[int, float]:
    classes = generator.classes
    class_counts = np.bincount(classes)
    total = class_counts.sum()
    class_count = len(class_counts)
    return {
        class_index: float(total / (class_count * count))
        for class_index, count in enumerate(class_counts)
        if count > 0
    }


def image_count(path: Path) -> int:
    return sum(
        1
        for image_path in path.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_class_directory(directory: Path, directory_name: str) -> None:
    if not directory.exists():
        raise FileNotFoundError(f"{directory_name} directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory_name} path is not a directory: {directory}")

    class_dirs = [path for path in directory.iterdir() if path.is_dir()]
    if len(class_dirs) < 2:
        raise ValueError(f"{directory_name} directory must contain at least two class subfolders.")

    empty_classes = []
    tiny_classes = []
    for class_dir in class_dirs:
        count = image_count(class_dir)
        if count == 0:
            empty_classes.append(class_dir.name)
        elif count < 5:
            tiny_classes.append((class_dir.name, count))

    if empty_classes:
        raise ValueError(f"These {directory_name} class folders contain no images: {', '.join(empty_classes)}")
    if tiny_classes:
        details = ", ".join(f"{name} ({count})" for name, count in tiny_classes)
        raise ValueError(f"Add more images before training. Very small {directory_name} classes: {details}")


def validate_matching_classes(train_dir: Path, validation_dir: Path | None, test_dir: Path | None) -> None:
    train_classes = {path.name for path in train_dir.iterdir() if path.is_dir()}
    for directory, name in ((validation_dir, "validation"), (test_dir, "test")):
        if directory is None:
            continue
        classes = {path.name for path in directory.iterdir() if path.is_dir()}
        if classes != train_classes:
            missing = sorted(train_classes - classes)
            extra = sorted(classes - train_classes)
            details = []
            if missing:
                details.append(f"missing from {name}: {', '.join(missing)}")
            if extra:
                details.append(f"extra in {name}: {', '.join(extra)}")
            raise ValueError(f"{name} classes must match training classes ({'; '.join(details)}).")


def validate_inputs(args: argparse.Namespace) -> None:
    validate_class_directory(Path(args.train_dir), "training")
    if args.validation_dir:
        validate_class_directory(Path(args.validation_dir), "validation")
    elif not 0 < args.validation_split < 0.5:
        raise ValueError("--validation-split must be greater than 0 and less than 0.5")
    if args.test_dir:
        validate_class_directory(Path(args.test_dir), "test")
    validate_matching_classes(
        Path(args.train_dir),
        Path(args.validation_dir) if args.validation_dir else None,
        Path(args.test_dir) if args.test_dir else None,
    )


def unfreeze_top_layers(model: Model, trainable_layers: int, learning_rate: float) -> None:
    try:
        backbone = model.get_layer(BACKBONE_LAYER_NAME)
    except ValueError:
        backbone = next(
            (layer for layer in model.layers if hasattr(layer, "layers") and len(layer.layers) > 10),
            None,
        )
        if backbone is None:
            raise ValueError("Could not find a nested backbone layer for fine-tuning.")

    backbone.trainable = True

    if trainable_layers > 0:
        for layer in backbone.layers[:-trainable_layers]:
            layer.trainable = False
    for layer in backbone.layers:
        if "batch_normalization" in layer.name.lower():
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    class_count = len(labels)
    confusion = np.zeros((class_count, class_count), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        confusion[int(actual), int(predicted)] += 1

    rows = []
    for index, label in enumerate(labels):
        true_positive = confusion[index, index]
        false_positive = confusion[:, index].sum() - true_positive
        false_negative = confusion[index, :].sum() - true_positive
        support = confusion[index, :].sum()
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
        )

    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    macro_precision = float(np.mean([row["precision"] for row in rows]))
    macro_recall = float(np.mean([row["recall"] for row in rows]))
    macro_f1 = float(np.mean([row["f1"] for row in rows]))
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "classes": rows,
        "confusion_matrix": confusion.tolist(),
    }


def save_training_report(output_dir: Path, model: Model, validation_generator) -> None:
    validation_generator.reset()
    probabilities = model.predict(validation_generator, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = validation_generator.classes
    labels = [
        label
        for label, _index in sorted(validation_generator.class_indices.items(), key=lambda item: item[1])
    ]
    report = classification_metrics(y_true, y_pred, labels)

    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=["label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(report["classes"])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, report["confusion_matrix"]):
            writer.writerow([label, *row])


def save_report(output_dir: Path, model: Model, generator, prefix: str) -> None:
    generator.reset()
    probabilities = model.predict(generator, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = generator.classes
    labels = [
        label
        for label, _index in sorted(generator.class_indices.items(), key=lambda item: item[1])
    ]
    report = classification_metrics(y_true, y_pred, labels)

    (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / f"{prefix}_classification_report.csv").open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=["label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(report["classes"])
    with (output_dir / f"{prefix}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, report["confusion_matrix"]):
            writer.writerow([label, *row])


def make_generators(args: argparse.Namespace):
    train_dir = Path(args.train_dir)
    validation_dir = Path(args.validation_dir) if args.validation_dir else None
    test_dir = Path(args.test_dir) if args.test_dir else None

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
        validation_split=0.0 if validation_dir else args.validation_split,
    )
    eval_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.0 if validation_dir else args.validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        subset=None if validation_dir else "training",
        shuffle=True,
        seed=args.seed,
    )

    if validation_dir:
        validation_generator = eval_datagen.flow_from_directory(
            validation_dir,
            target_size=IMAGE_SIZE,
            batch_size=args.batch_size,
            class_mode="categorical",
            shuffle=False,
            classes=list(train_generator.class_indices.keys()),
        )
    else:
        validation_generator = eval_datagen.flow_from_directory(
            train_dir,
            target_size=IMAGE_SIZE,
            batch_size=args.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=args.seed,
        )

    test_generator = None
    if test_dir:
        test_generator = eval_datagen.flow_from_directory(
            test_dir,
            target_size=IMAGE_SIZE,
            batch_size=args.batch_size,
            class_mode="categorical",
            shuffle=False,
            classes=list(train_generator.class_indices.keys()),
        )

    return train_generator, validation_generator, test_generator


def save_metadata(output_dir: Path, args: argparse.Namespace, class_indices: dict[str, int]) -> None:
    (output_dir / "class_indices.json").write_text(
        json.dumps(class_indices, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "architecture": args.architecture,
        "image_size": IMAGE_SIZE,
        "epochs": args.epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "validation_split": args.validation_split,
        "train_dir": str(args.train_dir),
        "validation_dir": str(args.validation_dir) if args.validation_dir else None,
        "test_dir": str(args.test_dir) if args.test_dir else None,
        "preprocessing": "rescale_1_255",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)
    validate_inputs(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_generator, validation_generator, test_generator = make_generators(args)

    model = build_model(
        class_count=len(train_generator.class_indices),
        architecture=args.architecture,
        learning_rate=args.learning_rate,
    )
    checkpoint_path = output_dir / "best_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]
    class_weights = compute_class_weights(train_generator)

    model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    if args.fine_tune_epochs > 0:
        model = load_model(checkpoint_path)
        unfreeze_top_layers(
            model,
            trainable_layers=args.fine_tune_layers,
            learning_rate=args.fine_tune_learning_rate,
        )
        model.fit(
            train_generator,
            epochs=args.fine_tune_epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weights,
        )

    model.save(output_dir / "final_model.keras")
    save_report(output_dir, model, validation_generator, "validation")
    if test_generator is not None:
        save_report(output_dir, model, test_generator, "test")
    save_metadata(output_dir, args, train_generator.class_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stronger ingredient classifier.")
    parser.add_argument("--train-dir", required=True, help="Folder containing one subfolder per class.")
    parser.add_argument("--validation-dir", default=None, help="Optional validation folder with matching class subfolders.")
    parser.add_argument("--test-dir", default=None, help="Optional test folder with matching class subfolders.")
    parser.add_argument("--output-dir", default="artifacts/better_model")
    parser.add_argument(
        "--architecture",
        choices=["efficientnetv2_b0", "mobilenet_v2"],
        default="efficientnetv2_b0",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
