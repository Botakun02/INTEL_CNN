import io
from pathlib import Path

import streamlit as st
from PIL import Image

from src.labels import normalize_recipe_label
from src.model_registry import available_models
from src.predict import load_keras_model, predict_top_k
from src.recipes import fallback_recipe_links, get_recipes, has_recipe_credentials


st.set_page_config(page_title="AI Kitchen", page_icon="🍳", layout="wide")

SAMPLES_DIR = Path("samples")
GITHUB_URL = "https://github.com/Botakun02/INTEL_CNN"
GRID_COLUMNS = 4


@st.cache_resource
def cached_model(model_path: str):
    return load_keras_model(model_path)


def format_confidence(value: float) -> str:
    return f"{value * 100:.1f}%"


def display_label(label: str) -> str:
    return label.title()


st.title("🍳 Intel Customised AI Kitchen")
st.caption(
    "Upload photos of your fruits and vegetables to identify them, then get recipe ideas "
    "for everything you found together."
)

model_options = {model_config["name"]: model_config for model_config in available_models()}
model_names = list(model_options.keys())
if not model_names:
    st.error("No models are available. Add a model file or train one with `python train.py`.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    if len(model_names) > 1:
        selected_name = st.selectbox("Classifier", model_names, index=0)
    else:
        selected_name = model_names[0]

    with st.expander("Advanced"):
        top_k = st.slider("Correction choices per image", min_value=1, max_value=5, value=3)
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            format="%.2f",
            help="Below this score, the app flags the prediction as uncertain.",
        )

    st.divider()
    if has_recipe_credentials():
        st.success("Recipe API connected.")
    else:
        st.info("No recipe API key configured, so search links are shown instead of full recipes.")

    model_config = model_options[selected_name]
    with st.expander("What can it recognize?"):
        st.write(", ".join(display_label(label) for label in sorted(model_config["labels"])))


def chunked(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


if "camera_shot_count" not in st.session_state:
    st.session_state.camera_shot_count = 0
if "camera_captures" not in st.session_state:
    st.session_state.camera_captures = []

uploaded_files = st.file_uploader(
    "Upload one or more ingredient images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

with st.expander("📷 Use your camera"):
    camera_photo = st.camera_input(
        "Take a photo", key=f"camera_input_{st.session_state.camera_shot_count}"
    )
    if camera_photo is not None and st.button("Add this photo"):
        st.session_state.camera_captures.append(
            (f"camera_{st.session_state.camera_shot_count}", camera_photo.getvalue())
        )
        st.session_state.camera_shot_count += 1
        st.rerun()

    if st.session_state.camera_captures:
        st.caption(f"{len(st.session_state.camera_captures)} photo(s) added from camera.")
        if st.button("Clear camera photos"):
            st.session_state.camera_captures = []
            st.rerun()

image_sources: list[tuple[str, str, Image.Image]] = []
for index, uploaded_file in enumerate(uploaded_files):
    image_sources.append((f"upload_{index}_{uploaded_file.name}", uploaded_file.name, Image.open(uploaded_file)))

for photo_key, photo_bytes in st.session_state.camera_captures:
    image_sources.append((photo_key, "Camera photo", Image.open(io.BytesIO(photo_bytes))))

sample_paths = sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
if sample_paths:
    st.write("No photos handy? Add an example:")
    sample_columns = st.columns(len(sample_paths))
    for column, sample_path in zip(sample_columns, sample_paths):
        with column:
            st.image(str(sample_path), use_container_width=True)
            if st.checkbox(sample_path.stem.title(), key=f"sample_selected_{sample_path.stem}"):
                image_sources.append((f"sample_{sample_path.stem}", sample_path.stem.title(), Image.open(sample_path)))

if not image_sources:
    st.info("Upload one or more images, or check an example above, to get started.")
else:
    model_path = Path(model_config["path"])
    if not model_path.exists():
        st.error(f"Missing model file: {model_path}")
        st.stop()

    model = cached_model(str(model_path))

    st.subheader(f"Identified ingredients ({len(image_sources)})")
    confirmed_labels: list[str] = []
    for row in chunked(image_sources, GRID_COLUMNS):
        columns = st.columns(GRID_COLUMNS)
        for column, (source_key, source_name, image) in zip(columns, row):
            with column:
                st.image(image, caption=source_name, use_container_width=True)
                with st.spinner("Classifying..."):
                    predictions = predict_top_k(model, model_config["labels"], image, k=top_k)
                top_prediction = predictions[0]
                if top_prediction.confidence < confidence_threshold:
                    st.caption(f"⚠️ Low confidence ({format_confidence(top_prediction.confidence)})")
                else:
                    st.caption(f"Confidence: {format_confidence(top_prediction.confidence)}")
                chosen_prediction = st.selectbox(
                    "Ingredient",
                    predictions,
                    format_func=lambda prediction: display_label(prediction.label),
                    key=f"label_choice_{source_key}",
                    label_visibility="collapsed",
                )
                confirmed_labels.append(normalize_recipe_label(chosen_prediction.label))

    unique_labels = sorted(set(confirmed_labels), key=confirmed_labels.index)
    combined_query = ", ".join(unique_labels)

    st.divider()
    st.write(f"**Your ingredients:** {combined_query}")

    if st.button("Find recipes", type="primary", use_container_width=True):
        with st.spinner("Searching recipes..."):
            try:
                recipes = get_recipes(combined_query)
            except Exception as exc:
                st.error(f"Recipe lookup failed: {exc}")
                recipes = []

        if not recipes:
            st.info("No Spoonacular recipes found. Showing recipe search links instead.")
            recipes = fallback_recipe_links(combined_query)

        if recipes:
            st.subheader(f"Recipes for {combined_query}")
            for recipe in recipes:
                with st.container(border=True):
                    columns = st.columns([1, 2.2])
                    with columns[0]:
                        if recipe.image_url:
                            st.image(recipe.image_url, use_container_width=True)
                    with columns[1]:
                        st.markdown(f"**{recipe.title}**")
                        if recipe.source:
                            st.caption(recipe.source)
                        st.link_button("Open recipe", recipe.url, use_container_width=True)

st.divider()
st.caption(
    "Fine-tuned EfficientNetV2 classifier over 33 ingredient classes, ~97% test accuracy. "
    f"[View source on GitHub]({GITHUB_URL})"
)
