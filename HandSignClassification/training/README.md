# Training

## Data Augmentation

학습에 사용할 데이터 수가 너무 적기 때문에 Keras의 ImageDataGenerator를 사용해 Augmentation을 하였습니다.

Augmentation시 적용한 변환은 가로 및 세로 이동, zoom, 밝기 등입니다.

학습 전에 미리 인자들의 값을 조정해가며 변환된 이미지를 테스트해보았습니다.

[구현 코드](./show_augmentation_result.ipynb)

## Training

### Fine Tuning

적은 학습 데이터로 좋은 성능을 내기 위해 fine tuning을 사용하였습니다.

Fine tuning에 사용한 모델은 Xception, ResNet에 비해 분류기 학습 결과가 좋았던 VGG16을 사용하였습니다.

### 학습 방법

1. 마지막 분류기를 제외한 VGG16 모델을 불러와 전체 모델을 동결
2. 동결 시킨 VGG16 모델의 출력을 입력으로 받는 분류기를 학습
3. 동결 시킨 VGG16 모델의 마지막 Convolution Block을 동결 해제시킨 후 분류기와 함께 재 학습

[구현 코드](./training.ipynb)



