# Hand Sign Classification

Hand Sign Classification 모델의 구현 과정을 정리하였다.

학습에는 Google Cloud Platform에서 NVIDIA Tesla V100 GPU 2개가 있는 VM instance를 생성하여 수행하였다.

## Environments

- Linux Ubuntu 18.04
- tensorflow 2.1.0
- python 3.7.6

## Contents

- [Dataset](./dataset)
    1. Dataset Preparation
    2. Augment and Resize data
- [Training](./training)
    1. Simple CNN Model
    2. Fine Tune Pre-trained Model : VGG16
    3. Model Selection
    4. Train all data