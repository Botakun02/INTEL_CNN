import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import requests


SPOONACULAR_URL = "https://api.spoonacular.com/recipes/findByIngredients"
ENV_PATH = Path(".env")


@dataclass(frozen=True)
class Recipe:
    title: str
    url: str
    image_url: str | None
    source: str | None


def load_local_env(env_path: Path = ENV_PATH) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def has_recipe_credentials() -> bool:
    load_local_env()
    return bool(os.getenv("SPOONACULAR_API_KEY"))


def fallback_recipe_links(query: str) -> list[Recipe]:
    encoded_query = quote_plus(f"{query} recipe")
    return [
        Recipe(
            title=f"Search Google for {query} recipes",
            url=f"https://www.google.com/search?q={encoded_query}",
            image_url=None,
            source="Google",
        ),
        Recipe(
            title=f"Search YouTube for {query} recipes",
            url=f"https://www.youtube.com/results?search_query={encoded_query}",
            image_url=None,
            source="YouTube",
        ),
        Recipe(
            title=f"Search BBC Good Food for {query}",
            url=f"https://www.bbcgoodfood.com/search?q={quote_plus(query)}",
            image_url=None,
            source="BBC Good Food",
        ),
    ]


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "recipe"


def get_recipes(query: str, limit: int = 6) -> list[Recipe]:
    load_local_env()
    api_key = os.getenv("SPOONACULAR_API_KEY")
    if not api_key:
        return []

    ingredients = [item.strip() for item in query.split(",") if item.strip()]
    if not ingredients:
        return []

    response = requests.get(
        SPOONACULAR_URL,
        params={
            "ingredients": ",".join(ingredients),
            "number": limit,
            "ranking": 1,
            "ignorePantry": "true",
            "apiKey": api_key,
        },
        timeout=15,
    )
    response.raise_for_status()

    recipes = []
    for item in response.json()[:limit]:
        title = item.get("title", "Untitled recipe")
        recipe_id = item.get("id")
        slug = slugify(title)
        recipes.append(
            Recipe(
                title=title,
                url=f"https://spoonacular.com/recipes/{slug}-{recipe_id}" if recipe_id else "",
                image_url=item.get("image"),
                source="Spoonacular",
            )
        )
    return recipes
