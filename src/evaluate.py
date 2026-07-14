"""Evaluate a trained CNN on a FER2013-style test dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64


def create_test_dataset(test_dir: Path):
    """Load the FER2013 test dataset.

    Expected directory structure:

    test_dir/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/

    Args:
        test_dir: Directory containing the seven test class folders.

    Returns:
        The TensorFlow test dataset and its class names.
    """
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = test_dataset.class_names

    test_dataset = test_dataset.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return test_dataset, class_names


def collect_predictions(model, test_dataset):
    """Collect true labels and model predictions."""
    true_labels = []
    predicted_labels = []

    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        true_labels.extend(labels.numpy())
        predicted_labels.extend(predictions)

    return np.array(true_labels), np.array(predicted_labels)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Create and save a confusion matrix figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_positions = np.arange(len(class_names))

    plt.xticks(
        tick_positions,
        class_names,
        rotation=45,
        ha="right",
    )
    plt.yticks(tick_positions, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate(
    model_path: Path,
    test_dir: Path,
    results_dir: Path,
) -> None:
    """Evaluate the trained model and save all evaluation outputs."""
    test_dataset, class_names = create_test_dataset(test_dir)

    model = tf.keras.models.load_model(model_path)

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    true_labels, predicted_labels = collect_predictions(
        model,
        test_dataset,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "classification_report.txt"
    summary_path = results_dir / "evaluation_summary.txt"
    matrix_path = results_dir / "confusion_matrix.png"

    report_path.write_text(report, encoding="utf-8")

    summary_path.write_text(
        (
            "FER2013 CNN Baseline Evaluation\n"
            "--------------------------------\n"
            f"Test accuracy: {test_accuracy:.4f}\n"
            f"Test loss: {test_loss:.4f}\n"
        ),
        encoding="utf-8",
    )

    save_confusion_matrix(
        matrix=matrix,
        class_names=class_names,
        output_path=matrix_path,
    )

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print("\nClassification report:")
    print(report)

    print(f"\nResults saved to: {results_dir}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FER2013 CNN model."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained .keras model.",
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Path to the FER2013 test directory.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory used to save evaluation outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Run model evaluation from the command line."""
    args = parse_arguments()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {args.model}"
        )

    if not args.test_dir.exists():
        raise FileNotFoundError(
            f"Test directory does not exist: {args.test_dir}"
        )

    evaluate(
        model_path=args.model,
        test_dir=args.test_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
