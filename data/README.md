
# Dataset Information

## Project Dataset

Rice Leaf Disease Detection dataset used for this project.

## Dataset Statistics

| Disease Class | Number of Images |
|---|---:|
| Bacterial Leaf Blight | 40 |
| Brown Spot | 40 |
| Leaf Smut | 39 |
| **Total** | **119** |

## Dataset Structure

The original dataset contains three disease-class folders:

- Bacterial Leaf Blight
- Brown Spot
- Leaf Smut

## Dataset Discrepancy

The institute project description refers to 120 images and mentions
Leaf Blast. However, the supplied dataset contains 119 images and
uses Leaf Smut as the third class.

The actual dataset labels were retained for model training.
No artificial image was created to compensate for the missing sample.

## Preprocessing

- Image decoding: JPEG, RGB
- Resizing: 128 x 448 pixels
- Aspect ratio: preserved using padding
- Pixel normalization: [0, 1]
- Feature extraction: ImageNet-pretrained MobileNetV2

## Data Availability

The original dataset is not included in this repository.
Place the dataset in the expected local folder structure before
running the notebook.
