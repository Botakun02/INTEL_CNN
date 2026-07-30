import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (224, 224)


def labels_from_class_indices(path: Path) -> list[str]:
    class_indices = json.loads(path.read_text(encoding="utf-8"))
    return [
        label
        for label, _index in sorted(class_indices.items(), key=lambda item: item[1])
    ]


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict:
    y_pred = np.argmax(probabilities, axis=1)
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

    top3 = np.argsort(probabilities, axis=1)[:, -3:]
    top3_accuracy = float(np.mean([actual in guesses for actual, guesses in zip(y_true, top3)]))
    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    return {
        "accuracy": accuracy,
        "top_3_accuracy": top3_accuracy,
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "classes": rows,
        "confusion_matrix": confusion.tolist(),
    }


def save_report(output_dir: Path, report: dict, labels: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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


def evaluate(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    labels = labels_from_class_indices(Path(args.class_indices))

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=labels,
    )

    model = load_model(model_path, compile=False)
    probabilities = model.predict(generator, verbose=1)
    report = classification_metrics(generator.classes, probabilities, labels)
    save_report(Path(args.output_dir), report, labels)

    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Top-3 accuracy: {report['top_3_accuracy']:.4f}")
    print("Weakest classes:")
    for row in sorted(report["classes"], key=lambda item: item["f1"])[:5]:
        print(f"- {row['label']}: f1={row['f1']:.4f}, support={row['support']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ingredient classifier on a held-out split.")
    parser.add_argument("--model", required=True, help="Path to .keras model.")
    parser.add_argument("--data-dir", required=True, help="Folder containing one class subfolder per label.")
    parser.add_argument("--class-indices", required=True, help="class_indices.json saved during training.")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
