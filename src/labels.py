RECIPE_LABEL_ALIASES = {
    "bell_pepper": "bell pepper",
    "chilli pepper": "chili pepper",
    "raddish": "radish",
    "sweetpotato": "sweet potato",
}


def normalize_recipe_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return RECIPE_LABEL_ALIASES.get(normalized, normalized)
