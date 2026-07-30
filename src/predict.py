from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


IMAGE_SIZE = (224, 224)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class LegacyDepthwiseConv2D(DepthwiseConv2D):
    """Loads older MobileNetV2 H5 configs that include an unused `groups` key."""

    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array / 255.0


def load_keras_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    original_from_config = DepthwiseConv2D.from_config

    def compatible_from_config(cls, config):
        config.pop("groups", None)
        return original_from_config(config)

    try:
        DepthwiseConv2D.from_config = classmethod(compatible_from_config)
        return load_model(
            path,
            compile=False,
            custom_objects={
                "DepthwiseConv2D": LegacyDepthwiseConv2D,
                "keras.layers.DepthwiseConv2D": LegacyDepthwiseConv2D,
            },
        )
    finally:
        DepthwiseConv2D.from_config = original_from_config


def predict_top_k(model, labels: Iterable[str], image: Image.Image, k: int = 3) -> list[Prediction]:
    labels = list(labels)
    probabilities = model.predict(preprocess_image(image), verbose=0)[0]

    if len(probabilities) != len(labels):
        raise ValueError(
            f"Model output has {len(probabilities)} classes, but {len(labels)} labels were provided."
        )

    top_indices = np.argsort(probabilities)[::-1][:k]
    return [
        Prediction(label=labels[index], confidence=float(probabilities[index]))
        for index in top_indices
    ]
