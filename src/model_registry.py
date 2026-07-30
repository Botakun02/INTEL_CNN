import json
from pathlib import Path


def labels_from_class_indices(class_indices: dict[str, int]) -> list[str]:
    return [
        label
        for label, _index in sorted(class_indices.items(), key=lambda item: item[1])
    ]


def discover_trained_models(artifacts_dir: str | Path = "artifacts") -> list[dict]:
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        return []

    discovered = []
    for labels_path in artifacts_path.rglob("class_indices.json"):
        try:
            class_indices = json.loads(labels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        model_dir = labels_path.parent
        for model_name in ("best_model.keras", "final_model.keras"):
            model_path = model_dir / model_name
            if model_path.exists():
                discovered.append(
                    {
                        "name": f"{model_dir.name} - {model_name}",
                        "path": str(model_path),
                        "labels": labels_from_class_indices(class_indices),
                    }
                )

    return discovered


def available_models() -> list[dict]:
    return discover_trained_models()
