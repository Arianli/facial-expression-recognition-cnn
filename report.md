# FER2013 CNN Baseline Experiment Report

## 1. Objective

The objective of this project was to build a reproducible convolutional neural network baseline for seven-class facial expression recognition using the FER2013 dataset.

The project focuses on implementing and evaluating a complete classification pipeline rather than proposing a new model architecture.

## 2. Dataset

The dataset contains 48 × 48 grayscale facial images classified into seven categories:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The training set contains 28,709 images, and the test set contains 7,178 images.

The dataset is imbalanced. The `happy` category contains 7,215 training images, while the `disgust` category contains only 436.

## 3. Model

The baseline CNN consists of:

- three convolutional layers
- max-pooling after each convolutional layer
- one dense hidden layer
- dropout with a rate of 0.5
- a seven-class softmax output layer

The model was trained with the Adam optimizer and sparse categorical cross-entropy loss.

## 4. Experimental Setup

- Image size: 48 × 48
- Image type: grayscale
- Batch size: 64
- Epochs: 15
- Training-validation split: 80/20
- Random seed: 42

## 5. Results

| Metric | Result |
|---|---:|
| Training accuracy | 70.83% |
| Validation accuracy | 55.95% |
| Test accuracy | 55.73% |
| Training loss | 0.7647 |
| Validation loss | 1.2840 |
| Test loss | 1.2875 |

The validation and test accuracy differ by only 0.22 percentage points, indicating that the test result is consistent with the validation result.

However, training accuracy is approximately 15.10 percentage points higher than test accuracy, showing that the model overfits the training data.

## 6. Error Analysis

The model performed relatively well on the `happy` and `surprise` categories.

Performance was weaker for `disgust`, which has substantially fewer training samples than the other categories.

The confusion matrix also shows that the model frequently confuses visually similar categories, including:

- fear and sad
- sad and neutral
- angry and sad
- fear and surprise

The low image resolution and possible label noise in FER2013 may also contribute to these errors.

## 7. Limitations

- The dataset is strongly imbalanced.
- The images have a low resolution of 48 × 48 pixels.
- No class weighting or oversampling was used.
- No extensive hyperparameter search was performed.
- Only one CNN architecture and one random seed were evaluated.
- The project does not claim state-of-the-art performance.

## 8. Future Work

Possible improvements include:

- data augmentation
- class-weighted loss
- batch normalization
- early stopping
- repeated experiments with multiple seeds
- deeper CNN architectures
- transfer learning
