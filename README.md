<<<<<<< HEAD
# UQCCNN: Unified Quantum–Classical CNN for Brain Tumor Classification

Hybrid deep learning framework integrating an 8-qubit Variational Quantum Circuit (VQC) within a classical CNN backbone for binary and multi-class brain tumor classification from MRI images.

---
=======
# UQCCNN: A Unified Quantum–Classical Convolutional Neural Network for Binary and Multi-Class Brain Tumor Classification
>>>>>>> 341df28504c1635b23140610583b0141be72a7b5

## Overview

This repository implements a Unified Quantum–Classical Convolutional Neural Network (UQCCNN) designed to perform:

- Binary classification (Tumor vs. No Tumor)
- Multi-class classification (Glioma, Meningioma, Pituitary, No Tumor)

The framework maintains a single unified architecture across tasks without structural modification.

Key research goals:
- Cross-dataset validation
- Hardware-aware noise robustness
- Hybrid quantum–classical integration
- Reproducible evaluation protocol

---

## Architecture

The model consists of:

### 1. Classical CNN Backbone
- Convolution + pooling layers
- Feature flattening
- Fully connected projection to 8-dimensional embedding

### 2. Quantum Module
- 8-qubit AngleEmbedding
- 3 layers of StronglyEntanglingLayers
- Pauli-Z expectation measurement
- Classical dense classification head

The 8-qubit design reflects a balance between expressive power and near-term NISQ hardware feasibility.

### Hybrid Pipeline

![Architecture](docs/architecture.png)

### Variational Quantum Circuit

![VQC Circuit](docs/vqc_circuit.png)

---

## Datasets

Two independent public brain MRI datasets were used:

| Dataset | Total Images | Tasks |
|----------|--------------|--------|
| Dataset-1 | 7,023 | Binary + Multi-class |
| Dataset-2 | 2,977 | Binary + Multi-class |

Binary grouping:
- Tumor: Glioma + Meningioma + Pituitary
- Non-Tumor: No Tumor

Original train/test splits were preserved.

---

## Experimental Configuration

- Image Size: 224 × 224
- Optimizer: AdamW
- Loss:
  - Focal Loss (Binary)
  - Weighted Cross-Entropy (Multi-class)
- Data Augmentation:
  - Geometric transforms
  - Intensity variation
  - Random erasing
  - Mixup
- Mixed precision training
- Test-Time Augmentation (TTA)

---

## Results

### Binary Classification

| Dataset | Accuracy | Precision | Recall | F1-score |
|----------|-----------|------------|--------|----------|
| Dataset-1 | 99.69% | 100.00% | 99.55% | 99.77% |
| Dataset-2 | 95.43% | 98.22% | 95.50% | 96.84% |

### Multi-Class Classification

| Dataset | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|----------|-----------|-----------------|--------------|----------|
| Dataset-1 | 98.93% | 98.89% | 98.88% | 98.88% |
| Dataset-2 | 78.68% | 85.16% | 78.43% | 76.44% |

---

## Noise-Aware Evaluation (NISQ-Inspired)

Simulated quantum perturbations:

- Quantum noise scale: 0.3
- Feature corruption rate: 20%
- Measurement error rate: 10%

Performance remained stable under noise-aware conditions.

Note: Experiments were conducted using a statevector simulator backend. This work does not claim execution on real quantum hardware.

---

## Repository Structure
UQCCNN-Brain-Tumor-Classification/
│
├── train.py
├── requirements.txt
├── models/
│ ├── hybrid_model.py
│ ├── classical_backbone.py
│ └── quantum_layer.py
├── utils/
├── results/
└── docs/

---

## Installation

```bash
pip install -r requirements.txt

## Running the Model

python train.py

## Research Contribution

This work emphasizes:

-Unified architecture for multiple diagnostic tasks

-Cross-dataset evaluation rather than single-benchmark optimization

-Hardware-aware robustness testing

-Integration of variational quantum circuits within classical CNNs

The study investigates feasibility and stability under realistic constraints rather than asserting quantum advantage.

## Citation

If you use this work, please cite:

Mitra, N., Kuiry, S.
A Unified Quantum–Classical Convolutional Neural Network for Binary and Multi-Class Brain Tumor Classification.
(Under IEEE Conference Resubmission)

## Status
<<<<<<< HEAD

Research manuscript under revision for IEEE conference submission.

=======
The corresponding paper is currently under review.
Citation details will be updated after acceptance.


## Note
Due to academic and confidentiality constraints, only the core implementation is shared.
>>>>>>> 341df28504c1635b23140610583b0141be72a7b5

