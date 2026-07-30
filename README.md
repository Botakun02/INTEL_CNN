# Intel Customised AI Kitchen

This project classifies fruit or vegetable images with a trained TensorFlow/Keras CNN
model and can suggest recipes for the detected ingredients through the Spoonacular API.

The original Colab exports are still in `Code Base/` and `Final Model/INTEL.ipynb`.
For local use, start with the Streamlit app and reusable Python modules added in this
version.

## What It Does

1. Upload one or more ingredient images (or pick from the example gallery).
2. Load a saved Keras classifier.
3. Preprocess each image to `224x224` RGB and classify it, with a dropdown to correct
   any misclassification.
4. Combine every confirmed ingredient into one search.
5. Optionally search Spoonacular for recipes using all detected ingredients together.

## Project Layout

```text
.
+-- streamlit_app.py          # Local web app
+-- train.py                  # Local training script
+-- evaluate.py               # Held-out split evaluation script
+-- scripts/
|   +-- prepare_dataset.py    # Converts downloaded datasets into trainer format
+-- datasets/
|   +-- ingredients/          # Prepared local copy of the supplied dataset
+-- requirements.txt          # Python dependencies
+-- .env.example              # Recipe API environment variable template
+-- src/
|   +-- labels.py             # Recipe label aliases
|   +-- model_registry.py     # Auto-discovers trained models in artifacts/
|   +-- predict.py            # Model loading and prediction helpers
|   +-- recipes.py            # Spoonacular recipe lookup
+-- Final Model/
|   +-- INTEL.ipynb           # Original Colab-style notebook
|   +-- intel.py              # Original Colab-style inference export
+-- Code Base/                # Original Colab-style training exports
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Recipe lookup is optional. Sign up for a free API key at
[spoonacular.com/food-api](https://spoonacular.com/food-api) (no credit card required
for the free plan, 50 points/day) and set it as an environment variable:

```powershell
$env:SPOONACULAR_API_KEY="your_spoonacular_api_key"
```

Or create a local `.env` file:

```text
SPOONACULAR_API_KEY=your_spoonacular_api_key
```

Do not commit real API keys. `.env.example` shows the required variable name. If the
key is missing or Spoonacular returns no results, the app shows recipe search links
instead.

## Run The App

```powershell
streamlit run streamlit_app.py
```

Then open the local URL printed by Streamlit.

## Train A Better Model

Recommended starter dataset: Kaggle's **Fruits and Vegetables Image Recognition
Dataset** by Kritik Seth. It is small enough for local training and closely matches
the app's current ingredient labels:

```text
https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition
```

For a larger, more real-world dataset, use **Packed Fruits and Vegetables Recognition
Benchmark / PackEat**:

```text
https://www.kaggle.com/datasets/sergeynesteruk/packed-fruits-and-vegetables-recognition-benchmark
https://zenodo.org/records/16901177
```

The supplied dataset from `C:\Users\bryan\Downloads\archive (1)` has already been
prepared into this workspace:

```text
datasets/ingredients/
+-- train/        # 3,115 images, 33 classes
+-- validation/   # 351 images, 33 classes
+-- test/         # 359 images, 33 classes
```

The dataset originally shipped with 36 class folders, but `bell pepper`/`capsicum`/`paprika`
were the same vegetable under three names, and `corn`/`sweetcorn` were the same crop under
two. Those folders were merged into `bell pepper` and `corn` respectively, which removed
label noise the model was otherwise being penalized for.

If you need to rebuild that prepared copy, run:

```powershell
python scripts/prepare_dataset.py --source "C:\Users\bryan\Downloads\archive (1)" --output-dir datasets/ingredients --all-splits
```

For any future dataset, use a directory where each class has its own subfolder:

```text
dataset/
+-- apple/
+-- banana/
+-- tomato/
```

If you downloaded a zip from Kaggle, convert one split into the trainer format:

```powershell
python scripts/prepare_dataset.py --source "C:\path\to\archive.zip" --output-dir datasets/kaggle_fruitveg --split train
```

If the extracted dataset already has `train`, `validation`, and `test` folders,
prepare all splits at once:

```powershell
python scripts/prepare_dataset.py --source "C:\path\to\extracted_dataset" --output-dir datasets/kaggle_fruitveg --all-splits
```

Run:

```powershell
python train.py --train-dir datasets/ingredients/train --validation-dir datasets/ingredients/validation --test-dir datasets/ingredients/test --output-dir artifacts/ingredients_model
```

The script saves:

- `artifacts/ingredients_model/best_model.keras`
- `artifacts/ingredients_model/final_model.keras`
- `artifacts/ingredients_model/class_indices.json`
- `artifacts/ingredients_model/metadata.json`
- validation and test metrics/reports as JSON and CSV files

The currently trained model scores 96.7% accuracy on the held-out test split
(`artifacts/ingredients_model/test_metrics.json`).

The default trainer uses EfficientNetV2B0, class balancing, stronger augmentation,
and a short fine-tuning stage. After training, restart the Streamlit app. Any model
inside `artifacts/` with a matching `class_indices.json` is discovered automatically
and listed in the sidebar; the app has no bundled fallback model, so at least one
trained model must exist under `artifacts/` before it will run.

Useful options:

```powershell
python train.py --train-dir datasets/ingredients/train --validation-dir datasets/ingredients/validation --epochs 25 --fine-tune-epochs 10
python train.py --train-dir datasets/ingredients/train --validation-dir datasets/ingredients/validation --architecture mobilenet_v2
python train.py --train-dir datasets/ingredients/train --validation-dir datasets/ingredients/validation --batch-size 16
```

To evaluate an already trained model on the held-out test split:

```powershell
python evaluate.py --model artifacts/ingredients_model/best_model.keras --data-dir datasets/ingredients/test --class-indices artifacts/ingredients_model/class_indices.json --output-dir artifacts/ingredients_model/test_eval
```

## Notes

- `Code Base/intell.py` and `Code Base/intel (1).py` are Colab exports. They include
  notebook shell commands such as `!kaggle`, so they are kept as references rather
  than local Python entry points.
- The app no longer hardcodes recipe API credentials.
- The local app reports top-k confidence values instead of only a single class.
