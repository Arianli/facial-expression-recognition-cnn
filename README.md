# Facial Expression Recognition CNN Baseline

This repository contains a reproducible convolutional neural network baseline for facial expression recognition using the FER2013 dataset.

The purpose of this project is to build and evaluate a complete image classification pipeline rather than propose a new model architecture.

## Dataset

The project uses the FER2013 facial expression dataset downloaded through KaggleHub.

The dataset contains 48 × 48 grayscale facial images from seven emotion categories:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The training folder contains 28,709 images, while the test folder contains 7,178 images.

The dataset is imbalanced. For example:

| Emotion | Training Images | Test Images |
|---|---:|---:|
| angry | 3,995 | 958 |
| disgust | 436 | 111 |
| fear | 4,097 | 1,024 |
| happy | 7,215 | 1,774 |
| neutral | 4,965 | 1,233 |
| sad | 4,830 | 1,247 |
| surprise | 3,171 | 831 |

## Model

The baseline model uses:

- three convolutional layers
- max-pooling after each convolutional block
- one fully connected hidden layer
- dropout for regularization
- softmax output for seven-class classification

Input images are rescaled from pixel values in the range 0–255 to 0–1.

## Experimental Setup

- Image size: 48 × 48
- Color mode: grayscale
- Batch size: 64
- Training epochs: 15
- Optimizer: Adam
- Loss function: sparse categorical cross-entropy
- Training/validation split: 80/20
- Random seed: 42

## Results

| Metric | Result |
|---|---:|
| Final training accuracy | approximately 70.83% |
| Final validation accuracy | approximately 55.95% |
| Test accuracy | approximately 55.73% |

The model achieved similar validation and test accuracy, suggesting that the evaluation result is reasonably consistent.

The training accuracy continued to increase while validation performance began to level off, indicating some overfitting during later epochs.

## Error Analysis

The model performed best on the `happy` and `surprise` classes.

The `disgust` class was more difficult to classify. One likely reason is the severe class imbalance: only 436 disgust images were available in the training set.

The confusion matrix also shows frequent confusion among:

- fear and sad
- sad and neutral
- angry and sad
- fear and surprise

These categories may share similar facial features, especially in low-resolution 48 × 48 images.

## Results and Files

- [Complete Colab notebook](notebooks/fer_cnn_baseline.ipynb)
- [Accuracy curve](results/accuracy_curve.png)
- [Loss curve](results/loss_curve.png)
- [Confusion matrix](results/confusion_matrix.png)
- [Experiment summary](results/experiment_summary.txt)
- [Classification report](results/classification_report.txt)

## Limitations

- The dataset contains significant class imbalance.
- FER2013 images are low-resolution and may contain noisy labels.
- The current model is a simple CNN baseline.
- No class weighting or oversampling was used.
- No extensive hyperparameter search was performed.
- This project does not claim state-of-the-art performance.

## Possible Improvements

Future experiments could include:

- data augmentation
- class-weighted training
- early stopping
- batch normalization
- transfer learning
- comparison with deeper CNN architectures
- repeated runs using different random seeds

## Repository Structure

```text
.
├── notebooks/
├── results/
├── src/
├── README.md
├── report.md
└── requirements.txt
