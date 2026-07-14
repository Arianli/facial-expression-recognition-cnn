# Dataset

This project uses the FER2013 facial expression dataset.

The dataset is not included in this repository because of its size. It can be downloaded through KaggleHub using:

```python
import kagglehub

dataset_path = kagglehub.dataset_download("msambare/fer2013")

## Expected Directory Structure

```text
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```
