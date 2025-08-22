# DermieAI: Bias Mitigation in Dermatological AI Systems

A research project investigating fairness and bias mitigation techniques in AI-powered dermatological diagnosis systems across diverse skin tones.

## Overview

This repository contains the implementation and analysis of various bias mitigation strategies for dermatological AI systems. The research addresses the critical issue of performance disparities across Fitzpatrick skin types (FST I-VI) in two key diagnostic tasks:

- **Cancer Detection**: Malignant vs. Benign lesion classification
- **Inflammatory Conditions**: Eczema vs. Psoriasis classification

## Problem Statement

Despite achieving dermatologist-level accuracy, current AI systems in dermatology exhibit systematic bias against darker skin tones. With an estimated 86.26% of the global population belonging to non-Caucasian racial groups, ensuring equitable performance across all skin tones is essential for responsible clinical deployment.

## Models & Techniques

> **Note**: The following models are implementations based on published research. Original implementations and papers are referenced below.

### 1. Baseline Model
-  ResNet-152 with transfer learning
  
### 2. TABE (Turning a Blind Eye)
- **Architecture**: ResNet-152 feature extractor with two classification heads
- **Key Features**:
  - Confusion loss to encourage uniform skin tone predictions
  - Gradient Reversal Layer (GRL) for adversarial training
  - Skin tone-invariant feature learning
- **Original Paper**: [*"Detecting Melanoma Fairly: Skin Tone Detection and Debiasing for Skin Lesion Classification"*](https://github.com/pbevan1/Detecting-Melanoma-Fairly/blob/main/train.py)

### 3. VAE (Variational Autoencoder)
- **Architecture**: ResNet-152 encoder with mirrored decoder
- **Key Features**:
  - Latent space regularization (120-dimensional)
  - Adaptive resampling based on latent variable rarity
  - Combined loss: classification + reconstruction + KL divergence
- **Original Paper**: [*"Towards Ethical Dermatology: Mitigating Bias in Skin Condition Classification"*](https://ieeexplore.ieee.org/document/10650487)

### 4. FairDisCo (Fair Disentanglement Contrastive Learning)
- **Architecture**: ResNet-152 with multiple specialized branches
- **Key Features**:
  - Adversarial training for skin tone suppression
  - Supervised contrastive learning
  - Multi-branch design (target, sensitive attribute, contrastive)
- **Original Paper**: [*"FairDisCo: Fairer AI in Dermatology via Disentanglement Contrastive Learning"*](https://github.com/siyi-wind/FairDisCo)

### 5. LesionCLIP
- Domain-specific foundation model used as alternative feature extractor for improved generalization
- **From**: [WhyLesionCLIP](https://huggingface.co/yyupenn/whylesionclip)

## Datasets
The following datasets were used in this study, the metadata for the publicly available is also found in this repo.
- Fitzpatrick17k
- PADUFES
- SCIN
- Dermie: Private dataset used for external validation 

## Experimental Design
- In-domain: train and testing on Fitzpatrick17k dataset
- Generalisation: testing on Dermie, training on the following dataset combinations
   - Fitzpatrick17k
   - Dataset Augmentations: Fitzpatrick17k + PADUFES, Fitzpatrick17k + SCIN, Fitzpatrick17k + PADFUES + SCIN
   
## Key Findings

### FairDisCo Superior In-Domain Performance

#### Cancer Detection (Fitzpatrick17k)
| Metric | FairDisCo | Baseline |
|--------|-----------|----------|
| **Balanced Accuracy** | **84.6 ± 2.3%** | 82.4 ± 0.8% |
| **EOM Fairness Score** | **0.814** | 0.773 |
| **PQD Score** | **0.871** | 0.833 |

#### Eczema/Psoriasis Classification (Fitzpatrick17k)
| Metric | FairDisCo | Baseline | 
|--------|-----------|----------|
| **Balanced Accuracy** | **81.6 ± 2.1%** | 77.7 ± 4.2% | 
| **EOM Performance** | 0.640 | 0.530 |

### FairDisCo Superior Generalisation without Dataset Augmentation

- **Cancer Detection**: Achieved **64.6%** balanced accuracy on external test set, being the only architecture to outperform baseline (63.3%) without augmentation
- **Eczema/Psoriasis**: Demonstrated highest generalization performance before dataset augmentation

### Optimal Performance-Fairness Trade-offs

- **Cancer Detection**: VAE with Fitzpatrick17k + PADFUES + SCIN
- **Eczema/Psoriasis**: FairDisCo with Fitpatrick17k

### LesionCLIP Foundation Model Benefits
- Improved generalisation performance accuracy in cancer detection

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{diaz2025bias,
  title={Bias in AI Dermatology Systems},
  author={Diaz, Rocio Mexia},
  year={2025},
  school={University College London},
  department={Department of Computer Science}
}
