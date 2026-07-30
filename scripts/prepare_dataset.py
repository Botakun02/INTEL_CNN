import argparse
import random
import shutil
import tempfile
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
SPLIT_ALIASES = {
    "train": {"train", "training"},
    "validation": {"valid", "validation", "val"},
    "test": {"test", "testing"},
}
LABEL_ALIASES = {
    "bell_pepper": "bell pepper",
    "chili pepper": "chilli pepper",
    "jalapeño": "jalapeno",
    "jalepeno": "jalapeno",
    "radish": "radish",
    "raddish": "radish",
    "soybean": "soy beans",
    "soybeans": "soy beans",
    "sweet corn": "sweetcorn",
    "sweet potato": "sweetpotato",
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return LABEL_ALIASES.get(normalized, normalized)


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(destination)


def contains_images(path: Path) -> bool:
    return any(
        item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        for item in path.rglob("*")
    )


def class_directories(path: Path) -> list[Path]:
    return [
        child
        for child in path.iterdir()
        if child.is_dir() and contains_images(child)
    ]


def find_split_root(root: Path, split: str) -> Path:
    aliases = SPLIT_ALIASES[split]
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and path.name.strip().lower() in aliases and class_directories(path):
            return path
    if class_directories(root):
        return root
    raise ValueError(f"Could not find a `{split}` split with class subfolders in {root}")


def copy_images(source_root: Path, output_dir: Path, max_per_class: int | None, seed: int) -> dict[str, int]:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_counts = {}

    for class_dir in class_directories(source_root):
        label = normalize_label(class_dir.name)
        target_dir = output_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        random.shuffle(image_paths)
        if max_per_class is not None:
            image_paths = image_paths[:max_per_class]

        for index, image_path in enumerate(image_paths, start=1):
            target_path = target_dir / f"{index:05d}_{image_path.name}"
            shutil.copy2(image_path, target_path)

        copied_counts[label] = copied_counts.get(label, 0) + len(image_paths)

    return dict(sorted(copied_counts.items()))


def prepare_dataset(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        if source.is_file() and source.suffix.lower() == ".zip":
            safe_extract(source, root)
        elif source.is_dir():
            root = source
        else:
            raise ValueError("Source must be a dataset folder or .zip file.")

        if args.all_splits:
            all_counts = {}
            for split in ("train", "validation", "test"):
                split_root = find_split_root(root, split)
                all_counts[split] = copy_images(
                    source_root=split_root,
                    output_dir=Path(args.output_dir) / split,
                    max_per_class=args.max_per_class,
                    seed=args.seed,
                )
        else:
            split_root = find_split_root(root, args.split)
            copied_counts = copy_images(
                source_root=split_root,
                output_dir=Path(args.output_dir),
                max_per_class=args.max_per_class,
                seed=args.seed,
            )

    print(f"Prepared dataset at: {args.output_dir}")
    if args.all_splits:
        for split, copied_counts in all_counts.items():
            print(f"{split}: {len(copied_counts)} classes, {sum(copied_counts.values())} images")
    else:
        print(f"Classes: {len(copied_counts)}")
        print(f"Images: {sum(copied_counts.values())}")
        for label, count in copied_counts.items():
            print(f"- {label}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert downloaded fruit/vegetable datasets into train.py folder format."
    )
    parser.add_argument("--source", required=True, help="Downloaded dataset folder or .zip file.")
    parser.add_argument("--output-dir", default="datasets/prepared", help="Prepared output folder.")
    parser.add_argument("--split", choices=sorted(SPLIT_ALIASES), default="train")
    parser.add_argument("--all-splits", action="store_true", help="Prepare train, validation, and test splits.")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_dataset(parse_args())
