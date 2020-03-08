# Training

## Data Augmentation

학습에 사용할 데이터 수가 너무 적기 때문에 Keras의 ImageDataGenerator를 사용해 상하좌우 이동, zoom, 밝기 등의 변환을 적용해 data augmentation을 하였습니다.

아래의 구현 코드를 통해 학습 전에 미리 인자들의 값을 조정해가며 변환된 이미지를 확인한 후 학습 데이터로 주입하였습니다.

구현 코드는 `show_augmentation_result.ipynb`에 있습니다.



## Training

적은 학습 데이터로 좋은 성능을 내기 위해 augmentation과 더불어 fine tuning을 사용하였습니다.

Fine tuning에 사용한 모델은분류기 학습 시 Xception, ResNet에 비해 성능이 좋았던 VGG16을 사용하였습니다.

### Training process

1. 마지막 분류기를 제외한 VGG16 모델을 불러와 전체 모델을 동결
2. 동결 시킨 VGG16 모델의 출력을 입력으로 받는 분류기를 학습
3. 동결 시킨 VGG16 모델의 마지막 Convolution Block을 동결 해제시킨 후 분류기와 함께 재 학습

구현 코드는 `training.ipynb`에 있습니다.