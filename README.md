# Deep Learning: Image Classification using CNN and Transfer Learning

A multi-notebook project exploring image classification on the CIFAR-10 dataset, progressing from a custom CNN built from scratch through several transfer learning architectures, and concluding with data augmentation experiments.

---

## Project Structure

```
cnn_project/
├── cnn_transfer_learning.ipynb   # Main notebook: custom CNN + transfer learning comparison
├── data_augmentation.ipynb       # Extension: ResNet50 with augmented training data
└── test.ipynb                    # PyTorch experiment: custom CNN with GPU-cached data
```

---

## Dataset

**CIFAR-10** — 60,000 color images (32×32 px) across 10 classes:
`airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck`

- 50,000 training images (5,000 per class)
- 10,000 test images (1,000 per class)

---

## Notebook 1: `cnn_transfer_learning.ipynb`

This is the main notebook and covers the full project pipeline from baseline CNN to fine-tuned transfer learning.

### 1. Baseline CNN (Built from Scratch)

A simple sequential CNN was implemented in TensorFlow/Keras:

| Layer | Output Shape | Parameters |
|---|---|---|
| Conv2D (32 filters, 3×3, ReLU) | (None, 30, 30, 32) | 896 |
| MaxPooling2D (2×2) | (None, 15, 15, 32) | 0 |
| Conv2D (64 filters, 3×3, ReLU) | (None, 13, 13, 64) | 18,496 |
| MaxPooling2D (2×2) | (None, 6, 6, 64) | 0 |
| Flatten | (None, 2304) | 0 |
| Dense (128, ReLU) | (None, 128) | 295,040 |
| Dense (10, Softmax) | (None, 10) | 1,290 |

**Total parameters: 315,722 (~1.2 MB)**

**Training setup:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Early stopping: patience=3, monitoring `val_loss`, restoring best weights
- Max epochs: 10

**Results (Baseline CNN):**

| Metric | Score |
|---|---|
| Test Accuracy | **64.50%** |
| Precision (weighted) | 64.43% |
| Recall (weighted) | 64.50% |
| F1 Score (weighted) | 64.04% |

The model converged steadily but showed signs of overfitting from epoch 7 onward (training accuracy ~81%, validation ~64.5%).

---

### 2. Transfer Learning — MobileNetV2

**Source:** TensorFlow Hub (`tf2-preview/mobilenet_v2`)
**Input resized to:** 224×224
**Head added:** Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)
**Base model frozen** (feature extractor only)

**Results (MobileNetV2):**

| Epoch | Val Accuracy |
|---|---|
| 1 | 82.97% |
| 5 | 85.47% |
| Best | **~85.5%** |

MobileNetV2 already delivers a **+21 percentage point** improvement over the scratch CNN, despite being used only as a frozen feature extractor.

---

### 3. Transfer Learning — InceptionV3

**Source:** TensorFlow Hub (`imagenet/inception_v3`)
**Same head architecture** as MobileNetV2
**Base model frozen**

**Results (InceptionV3):**

| Epoch | Val Accuracy |
|---|---|
| 1 | 84.74% |
| 3 | 86.06% |
| Best | **~86.4%** |

InceptionV3 slightly outperforms MobileNetV2 despite being a heavier model, converging stably over 7 epochs before early stopping triggered.

---

### 4. Transfer Learning — ResNet50 (Two-Phase Fine-Tuning)

The most sophisticated approach in the notebook, using a **two-phase training strategy**:

**Phase 1 — Head Training Only:**
- Base ResNet50 fully frozen (weights from ImageNet)
- Input resized to 96×96 (less interpolation distortion from 32px originals)
- Head: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)
- Learning rate: 1e-3

| Epoch | Val Accuracy |
|---|---|
| 1 | 84.43% |
| 4 | 86.38% |
| Best (Phase 1) | **~86.77%** |

**Phase 2 — Fine-Tuning (Top 30 Layers Unfrozen):**
- Lower learning rate: 1e-5 (to avoid destroying pretrained weights)
- Only the last 30 ResNet50 layers made trainable

| Epoch | Val Accuracy |
|---|---|
| 1 | 87.60% |
| 3 | 89.08% |
| 6 | **89.61%** |

**Total ResNet50 parameters: 23,851,274 (~91 MB)**
**Trainable during Phase 1: 263,562 (~1 MB)**

---

## Notebook 2: `data_augmentation.ipynb`

This notebook extends the ResNet50 experiment by applying **online data augmentation** to the training pipeline, effectively doubling the training set size.

### Augmentation Techniques Applied

| Technique | Parameters |
|---|---|
| Random Horizontal Flip | 50% probability |
| Random Rotation | ±20° |
| Random Zoom | ±10% |
| Random Translation | ±10% horizontal and vertical |

### Data Pipeline

The original training dataset (50,000 images) and an augmented copy were both created and concatenated, yielding **100,000 training samples** per epoch. The test set was kept unaugmented.

### Training

The same two-phase ResNet50 strategy from Notebook 1 was applied on the combined dataset:

**Phase 1 (Head only, 1e-3 LR):** The model trained across all 10 epochs without early stopping, reaching ~86.65% validation accuracy.

**Phase 2 (Fine-tuning top 30 layers, 1e-5 LR):** Early stopping triggered mid-training after validation accuracy improved to ~88.0%.

The augmented pipeline trades faster convergence for slower per-epoch time (roughly 2× as many batches), but is expected to improve generalization and robustness to real-world image variation.

---

## Notebook 3: `test.ipynb` (PyTorch Experiment)

A parallel experiment reimplementing the pipeline in **PyTorch** with GPU-cached data for faster training iteration.

### Key Design Choices

**GPU Caching:** The full dataset is loaded onto GPU memory once as cached `.pt` tensors, eliminating CPU→GPU transfer overhead per batch:

```python
torch.save((train_images, train_labels), "cifar10_train_cached.pt")
```

**Custom Dataset Class (`CachedCIFAR`):** Implements online augmentation (random flip + rotation up to ±15°) directly on GPU tensors using `torchvision.transforms.functional`.

**Custom CNN (PyTorch):**

| Layer | Details |
|---|---|
| Conv2D → ReLU → MaxPool | 3→32 channels, 3×3 kernel |
| Conv2D → ReLU → MaxPool | 32→64 channels, 3×3 kernel |
| Flatten | 64×8×8 = 4,096 features |
| Linear → ReLU | 4,096 → 256 |
| Linear (output) | 256 → 10 |

- Optimizer: Adam (lr=1e-3)
- Loss: CrossEntropyLoss
- Batch size: 512 (leveraging GPU caching)

---

## Model Comparison Summary

| Model | Framework | Val Accuracy | Parameters | Notes |
|---|---|---|---|---|
| Custom CNN (scratch) | TensorFlow | 64.50% | 315K | Baseline |
| MobileNetV2 (frozen) | TensorFlow Hub | ~85.5% | ~3.4M | Feature extractor only |
| InceptionV3 (frozen) | TensorFlow Hub | ~86.4% | ~21.8M | Feature extractor only |
| ResNet50 (Phase 1) | TensorFlow/Keras | ~86.8% | 263K trainable | Head only |
| ResNet50 (Phase 2, fine-tuned) | TensorFlow/Keras | **~89.6%** | ~23.9M | Best result |
| ResNet50 + Augmentation | TensorFlow/Keras | ~88.0% | ~23.9M | 2× training data |
| Custom CNN (PyTorch) | PyTorch | N/A (error) | ~600K | GPU-cached pipeline |

---

## Key Findings

**Transfer learning dramatically outperforms training from scratch.** Even a frozen MobileNetV2 used purely as a feature extractor achieved 85.5% accuracy compared to 64.5% for the custom CNN — a gain of over 20 percentage points with the same head architecture and training time constraints.

**Two-phase fine-tuning is the most effective strategy.** First training only the classification head (Phase 1), then unfreezing the top layers of the base model with a very low learning rate (Phase 2), pushed ResNet50 to ~89.6% validation accuracy. The low LR in Phase 2 is critical — it preserves the pretrained feature representations while allowing task-specific adaptation.

**Input resolution matters for small images.** The CIFAR-10 images are only 32×32 pixels. Resizing to 96×96 (used for ResNet50) introduces less interpolation distortion than 224×224 (used for MobileNetV2 and InceptionV3), which may partly explain ResNet50's edge despite being trained on the same frozen backbone concept.

**Data augmentation adds robustness but slows convergence.** The augmented ResNet50 experiment shows slightly lower peak accuracy in the runs captured (~88%), likely because the augmented data introduces harder examples that require more epochs to converge. With sufficient training time, augmentation is expected to improve generalization on unseen data.

---

## Dependencies

```
tensorflow >= 2.x
tensorflow-hub
scikit-learn
matplotlib
numpy
torch (for test.ipynb)
torchvision (for test.ipynb)
```

Install via:

```bash
pip install tensorflow tensorflow-hub scikit-learn matplotlib numpy torch torchvision
```

---

## How to Reproduce

1. Open `cnn_transfer_learning.ipynb` and run all cells in order. The CIFAR-10 dataset is downloaded automatically via `tf.keras.datasets.cifar10.load_data()`.
2. Open `data_augmentation.ipynb` for the augmented ResNet50 experiment.
3. For the PyTorch pipeline in `test.ipynb`, run the caching cell first, then restart the kernel before running the training cell (or run all cells in a single session without interruption).

