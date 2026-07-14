"""CNN model used for the FER2013 baseline experiment."""

from tensorflow.keras import layers, models


def build_model(
    input_shape: tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7,
):
    """Build and compile the baseline CNN.

    Args:
        input_shape: Height, width, and channels of each input image.
        num_classes: Number of facial-expression categories.

    Returns:
        A compiled Keras model.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    baseline_model = build_model()
    baseline_model.summary()
