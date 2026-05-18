"""Run the full local ReviewMind ML pipeline in order."""

import subprocess
import sys


STEPS = [
    ("Prepare IMDb dataset", "src/prepare_imdb_dataset.py"),
    ("Train TF-IDF + Logistic Regression baseline", "src/train_ml.py"),
    ("Compare multiple models", "src/compare_models.py"),
    ("Generate result visualizations", "src/visualize_results.py"),
]


def run_step(step_name: str, script_path: str) -> None:
    print(f"\n[START] {step_name}")
    print(f"Running: {sys.executable} {script_path}")

    subprocess.run([sys.executable, script_path], check=True)

    print(f"[DONE] {step_name}")


def main() -> None:
    print("ReviewMind pipeline started.")
    print("This will run dataset prep, training, model comparison, and visualization.")

    for step_name, script_path in STEPS:
        run_step(step_name, script_path)

    print("\nPipeline completed successfully.")
    print("Note: Streamlit app is not started by this script.")


if __name__ == "__main__":
    main()
