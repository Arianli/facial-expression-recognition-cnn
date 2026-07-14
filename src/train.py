"""Train the baseline CNN on a FER2013-style directory dataset."""

import argparse
from pathlib import Path

import tensorflow as tf

from model import build_model


IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42
DEFAULT_EPOCHS = 15


def create_datasets(data_dir: Path):
    """Create training and validation datasets.

    The expected folder structure is:

    data_dir/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/

    Args:
        data_dir: Directory containing the seven class folders.

    Returns:
        The training dataset, validation dataset, and class names.
    """
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_dataset.class_names
    autotune = tf.data.AUTOTUNE

    train_dataset = (
        train_dataset
        .cache()
        .shuffle(1000, seed=SEED)
        .prefetch(buffer_size=autotune)
    )

    validation_dataset = (
        validation_dataset
        .cache()
        .prefetch(buffer_size=autotune)
    )

    return train_dataset, validation_dataset, class_names


def train(data_dir: Path, output_path: Path, epochs: int) -> None:
    """Train the baseline CNN and save the trained model."""
    train_dataset, validation_dataset, class_names = create_datasets(data_dir)

    print("Class names:", class_names)

    model = build_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    print(f"Model saved to: {output_path}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a CNN baseline on the FER2013 dataset."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the FER2013 training directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fer_cnn_baseline.keras"),
        help="Path used to save the trained model.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum number of training epochs.",
    )

    return parser.parse_args()


def main() -> None:
    """Run model training from the command line."""
    args = parse_arguments()

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Training directory does not exist: {args.data_dir}"
        )

    train(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
