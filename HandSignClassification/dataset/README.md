# Dataset

Dataset은 여러 배경에서 손 모양을 직접 촬영한 사진을 사용하였습니다.

총 7가지 손모양을 촬영하였으며 이미지와 레이블은 다음과 같습니다.

<img src="./class_images/0_front.JPG" alt="0_front" width="150" height="150"/>	0_front

<img src="./class_images/1_back.JPG" alt="1_back" width="150" height="150"/>	1_back

<img src="./class_images/1_front.JPG" alt="1_front" width="150" height="150"/>	1_front

<img src="./class_images/2_back.JPG" alt="2_back" width="150" height="150"/>	2_back

<img src="./class_images/2_front.JPG" alt="2_front" width="150" height="150"/>	2_front

<img src="./class_images/5_front.JPG" alt="5_front" width="150" height="150"/>	5_front

<img src="./class_images/ilu.JPG" alt="ilu" width="150" height="150"/>	ILU



## Dataset 분리

전체 사진은 각 class별로 약 250장이며 이중 50장은 검증용 나머지는 학습용으로 두었습니다.

테스트 데이터셋을 따로 두지 않은 이유는 데이터 수가 워낙 적은데다 모델의 성능은 OpenCV를 통해 실시간으로 받는 이미지를 대상으로 평가할 계획이었기 때문입니다.

따라서 다음과 같이 directory를 구성하였습니다.

분리에 사용한 코드는 `./split_datasets.ipynb`에 있습니다.

```
train/
	0_front
	1_back
	1_front
	2_back
	2_front
	5_front
	ILU
val/
	0_front
	1_back
	1_front
	2_back
	2_front
	5_front
	ILU
```

