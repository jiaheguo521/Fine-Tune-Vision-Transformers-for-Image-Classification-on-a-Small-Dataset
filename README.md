# Fine-Tuning Vision Transformers for Image Classification on a Small Dataset

> KTH Royal Institute of Technology | Course Project DD2424
> 

This repository contains the code for the KTH course project (DD2424, Deep Learning in Data Science) titled "Fine-Tune Vision Transformers for Image Classification on a Small Dataset."

The project investigates the effectiveness of fine-tuning Vision Transformers (ViT) on small-scale datasets, a common real-world scenario where data is limited. Using the **Oxford-IIIT Pet Dataset**, this study explores various fine-tuning strategies and compares the performance of a pre-trained ViT against established CNN baselines (ResNet18 and ResNet50).

**Author:** Jiahe Guo ([@jiaheguo521](https://www.google.com/search?q=https://github.com/jiaheguo521))

## Key Findings

The primary finding of this project is that despite implementing several advanced fine-tuning techniques, the **Vision Transformer did not outperform the ResNet baselines.**

- The best ViT configuration achieved **93.38%** accuracy.
- A fine-tuned ResNet50 reached **94.02%** on the same 37-class classification task.

This suggests that for small-scale datasets like Oxford-IIIT Pets, traditional CNN architectures (which have stronger inductive biases) may still be a more effective and efficient choice than the more data-hungry Transformer architecture.

## Dataset

- [**The Oxford-IIIT Pet Dataset**](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **Task 1 (Binary):** Cat vs. Dog classification.
- **Task 2 (Multi-class):** 37-category pet breed classification.
- **Size:** ~7,300 images (~200 per class).
- **Split:** 3,680 training images (split 80/20 for train/validation) and 3,669 test images.
- **Input Size:** All images resized to 224x224.

## Models & Methodology

### Models Compared

1. **Vision Transformer (ViT):** `google/vit-base-patch16-224` (pre-trained on ImageNet-21k).
2. **CNN Baselines:** `ResNet18` and `ResNet50` (pre-trained on ImageNet).

### Fine-Tuning Strategies Explored

This project implements and evaluates several fine-tuning strategies for the ViT model:

- **Baseline (Frozen):** Training only the final classification head.
- **Gradual Unfreezing:** Progressively unfreezing transformer encoder layers during training (e.g., unfreezing one layer every 2 epochs).
- **Differential Learning Rates:** Applying a smaller learning rate (e.g., `1e-5`) to the backbone layers and a larger one (e.g., `1e-4`) to the new classification head.
- **Learning Rate Schedulers:** Using a cosine annealing schedule with a warm-up period.
- **Data Augmentation:** Applying random horizontal flips, rotations (±15 degrees), and color jittering.
- **Optimizer:** AdamW.
- **Regularization:** Early stopping with a patience of 5 epochs.

## Summary of Results

| Task | Model | Best Accuracy |
| --- | --- | --- |
| Binary Classification (Cat vs. Dog) | ResNet18 | 99.2% |
| 37-Class Breed Classification | ResNet50 | **94.02%** |
| 37-Class Breed Classification | Vision Transformer (ViT) | 93.38% |

## How to Run

### 1. Setup

Clone the repository and install the required dependencies.

```
# 1. Clone the repository
git clone https://github.com/jiaheguo521/Fine-Tune-Vision-Transformers-for-Image-Classification-on-a-Small-Dataset.git
cd Fine-Tune-Vision-Transformers-for-Image-Classification-on-a-Small-Dataset

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
# (Assuming a requirements.txt file exists in the repo)
pip install -r requirements.txt

```

Key dependencies include `torch`, `torchvision`, `transformers`, `timm`, `numpy`, and `matplotlib`.

### 2. Training

The main scripts (e.g., `train.py`, `evaluate.py`) can be run from the command line. Please refer to the source code for specific arguments related to learning rate, model choice, and fine-tuning strategy.

## Reference

For a detailed analysis, methodology, and full results, please see the accompanying research paper for this project.

[**Fine-Tune Vision Transformers for Image Classification on a Small Dataset.pdf**](https://www.google.com/search?q=https(://github.com/jiaheguo521/Fine-Tune-Vision-Transformers-for-Image-Classification-on-a-Small-Dataset/blob/main/Fine_Tune_Vision_Transformers_for_Image_Classification_on_a_Small_Dataset.pdf))*(Note: You will need to upload the PDF to your repository for this link to work.)*
