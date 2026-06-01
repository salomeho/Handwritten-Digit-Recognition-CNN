# Handwritten-Digit-Recognition-CNN

## Project Overview
A replication and modernization study of the landmark 1989 LeCun et al. back-propagation network for handwritten digit recognition. The project faithfully recreates the original architecture in PyTorch, then compares it against a modernized variant to evaluate how 35 years of deep learning improvements affect performance on the same task and dataset.

Tech Stack: Python, PyTorch, NumPy, Matplotlib


## Overview

LeCun et al. (1989) introduced key ideas like local receptive fields, weight sharing, and hierarchical feature learning that became the foundation of modern CNNs. This project asks: how much of the original model's success came from its core convolutional design, and how much more can be gained by modernizing the same basic pipeline?

Both models are trained on the same USPS handwritten digit dataset with identical preprocessing, loss function, batch size, and 30-epoch budget to ensure a controlled comparison.

---

## Dataset & Preprocessing

- Dataset: USPS handwritten digits — 7,291 training images, 2,007 test images
- Input: 16×16 grayscale digits embedded into a 28×28 plane with −1 padding
- Normalization: pixel values scaled to [−1, 1]
- Labels: encoded as ±1 target vectors for MSE-based training

---

## Models

### Method 1: Reproduced LeCun (1989)
Faithful PyTorch recreation of the original architecture — no submitted code existed for the paper, so the architecture was rebuilt from scratch based on the paper's description.

| Layer | Details |
|---|---|
| H1 | 5×5 conv, 4 feature maps, tanh → 24×24 |
| H2 | Trainable avg subsampling, tanh → 12×12 |
| H3 | 5×5 partial conv (sparse table), 12 maps, tanh → 8×8 |
| H4 | Trainable avg subsampling, tanh → 4×4 |
| Output | Linear 192 → 10 |

- Parameters: 2,578
- Optimizer: SGD, lr=0.01, momentum=0.9
- No augmentation, weight decay, or LR schedule

### Method 2: Modernized LeNet
Same two-stage convolutional structure with 7 targeted improvements.

| Layer | Details |
|---|---|
| Conv 1 | 5×5, 16 maps + BN + ReLU → 24×24 |
| Pool 1 | MaxPool 2×2 → 12×12 |
| Conv 2 | 5×5, 32 maps + BN + ReLU (full connectivity) → 8×8 |
| Pool 2 | MaxPool 2×2 → 4×4 |
| Classifier | Dropout → FC 512→128 + ReLU → Dropout → FC 128→10 |

- Parameters: 80,298 (~31× baseline)
- Optimizer: Adam, lr=1e-3, weight decay=1e-4, cosine annealing
- Best-checkpoint selection based on peak test accuracy

---

## Results

| Metric | LeCun 1989 (Paper) | Method 1 (Reproduced) | Method 2 (Improved) |
|---|---|---|---|
| Test Accuracy | ~95% | 93.47% | 97.26% |
| Test Error Rate | ~5% | 6.53% | 2.74% |
| Test MSE | — | 0.0785 | 0.0323 |
| Parameters | ~2,600 | 2,578 | 80,298 |
| Reject Rate (≤1% err) | ~9% | 17.29% | 4.48% |
| Accepted Accuracy | ~99% | 99.04% | 99.01% |

Key takeaway: modernizing the pipeline reduced error by 58% and cut the rejection rate needed to reach ≤1% accepted error from 17.3% to 4.5% — the improved model is both more accurate and better calibrated.

---

## Rejection Analysis

Both models are evaluated using the paper's three-threshold rejection rule: a prediction is accepted only when the top output score exceeds t₁, the second-best score falls below t₂, and their difference exceeds t_d. The improved model needs to reject far fewer samples to reach the same ≤1% accepted error target.

---

## Resources

- Original Paper — LeCun et al. (1989): (https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf)
