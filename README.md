# MIRAGE: Mutual Inference for Risk-Aware Disinformation Detection and Collusive Group DiscovEry

Official implementation for the paper: **"Who Is Spreading Disinformation? Collusive User Detection via Information Propagation Modeling in Social Media."**

## Table of Contents

- [Introduction](#Introduction)
- [Directory Structure](#Directory_Structure)
- [Dataset Preparation](#Dataset_Preparation)
- [Environment Setup](#Environment_Setup)
- [Run Experiments](#Run_Experiments)

## Introduction

Existing disinformation detection studies usually treat news events as independent samples and focus on event-level classification based on content semantics, propagation structures, or fixed social contexts. However, disinformation diffusion on social media often involves repeated cross-event participation, synchronized spreading, and coordinated group behaviors. Meanwhile, existing collusive behavior detection methods can identify suspicious user groups, but they are usually not constrained by news veracity, making it difficult to distinguish ordinary active communities from high-risk groups oriented toward disinformation spreading.

To address this gap, we propose **MIRAGE**（Mutual Inference for Risk-Aware Disinformation Detection and Collusive Group DiscovEry）, a mutual inference framework that jointly models event-level disinformation risk and group-level collusive behavior. Event-level risk provides weak supervision for discovering high-risk user groups across events, while inferred group risk is fed back into event-level detection as cross-event behavioral evidence. Specifically, MIRAGE introduces user risk modeling and risk-modulated propagation learning to distinguish trustworthy diffusion paths from high-risk spreading noise, and further constructs a risk-guided cross-event user collusion graph to capture repeatedly emerging coordinated behaviors.

Experiments on the PolitiFact and GossipCop datasets show that MIRAGE achieves state-of-the-art performance in disinformation detection. Beyond event-level accuracy, behavioral validation, future infiltration analysis, and cross-event case studies demonstrate that the detected groups are not merely highly active communities, but exhibit stronger fake-oriented coordination patterns and higher future reappearance tendencies.
<img width="1278" height="516" alt="{755A92F6-70CC-4631-B524-5744B8F8F187}" src="https://github.com/user-attachments/assets/a8fb08d8-2755-4236-9bc7-d0b83c9db3ba" />

------

## Directory_Structure

```text
MIRAGE/
├── checkpoints/               # Saved model weights and intermediate files
├── data/                      # Dataset directory
├── models/                    # Core model architecture
├── training/                  # Training and evaluation procedures
├── utils/                     # Data loading and utility functions
├── results/                   # Output directory for reproduced results
├── config.py                  # Hyperparameters and configuration settings
├── main.py                    # Main entry point for PolitiFact experiments
└── retest_saved_checkpoint.py # Script for reproducing results with saved checkpoints
```

---

## Dataset_Preparation

The project primarily utilizes the **UPFD (User Preference-aware Fake News Detection)** dataset:

* **GossipCop**: Entertainment/Celebrity news containing 5,464 news propagation graphs.
* **PolitiFact**: Political news containing 314 news propagation graphs.

You can obtain the processed data from: [dataset](https://drive.google.com/uc?export=download&id=12ihCH3eE-V6dnNHmQGxMz6OVW2uqVBFs)

After downloading, place the dataset under the following directory:

```
./data
```

---

## Environment_Setup

Install the required dependencies with:

```
pip install -r requirement.txt
```

**Experiments are conducted on a NVIDIA RTX 4090.**

---

## Run_Experiments

### 1. Run with the best hyperparameters

For **PolitiFact**:

```
python main.py \  --dataset politifact \  --config ultra \  --phase all \  --seed 42 \  --lr 0.002 \  --dropout 0.4 \  --hidden_dim 128 \  --lambda2 0.2 \  --dead_hours 12
```

For **GossipCop**:

```
python main1.py \  --dataset gossipcop \  --config ultra \  --phase all \  --seed 42 \  --lr 0.0015 \  --dropout 0.25 \  --hidden_dim 256 \  --lambda2 0.15 \  --dead_hours 48
```

### 2. Reproduce results using released checkpoints

We provide the released checkpoints and intermediate files for result reproduction.  
They can be downloaded from: [checkpoints](https://drive.google.com/uc?export=download&id=16752iI7yd5p4X6_eJXvUi7F6cAcNLKxb)


After downloading, place or extract the files into the following directory:

```text
./checkpoints
```

The checkpoint directory should contain files such as:

```
checkpoints/
├── phase3_best_pol.pth
├── phase2_collusion_pol.pkl
├── aligner_pol.pkl
├── phase3_best_gos.pth
├── phase2_collusion_gos.pkl
└── aligner_gos.pkl
```

For **PolitiFact**:

```
python retest_saved_checkpoint.py \
  --dataset politifact \
  --config ultra \
  --data_root ./data \
  --checkpoint ./checkpoints/phase3_best_pol.pth \
  --collusion_pkl ./checkpoints/phase2_collusion_pol.pkl \
  --aligner_pkl ./checkpoints/aligner_pol.pkl \
  --seed 42 \
  --lr 0.002 \
  --dropout 0.4 \
  --hidden_dim 128 \
  --lambda2 0.2 \
  --dead_hours 12 \
  --save_json results/retest_politifact_pol.json
```

For **GossipCop**:

```
python retest_saved_checkpoint.py \
  --dataset gossipcop \
  --config ultra \
  --data_root ./data \
  --checkpoint ./checkpoints/phase3_best_gos.pth \
  --collusion_pkl ./checkpoints/phase2_collusion_gos.pkl \
  --aligner_pkl ./checkpoints/aligner_gos.pkl \
  --seed 42 \
  --lr 0.0015 \
  --dropout 0.25 \
  --hidden_dim 256 \
  --lambda2 0.15 \
  --dead_hours 48 \
  --save_json results/retest_gossipcop_gos.json
```

The reproduced results will be saved under:

```
./results
```
