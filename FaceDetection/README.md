# Face Detection

Face Detection 모델의 구현 과정을 정리하였다.

학습에는 <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow Object Detection API</a>의 pretrained model(ssd mobilenet-v1)을 사용하였고, Google Cloud Platform에서 NVIDIA Tesla V100 GPU 2개가 있는 VM instance를 생성하여 수행하였다.

또한, 과정이 복잡하여 추후 다시 참고하기 쉽도록 tutorial 형식으로 하나하나 문서를 작성하였다.

## Environments
- Linux Ubuntu 18.04
- Tensorflow 1.14
- Python 3.7.3

## Contents
- <a href='./installation'>Object Detection API 및 Tensorflow, Python Library 설치</a>
- <a href='./dataset'>Dataset</a>
- <a href='./training'>Training</a>

## References

- https://github.com/tensorflow/models/tree/master/research/object_detection

- https://medium.com/@victor.dibia/how-to-build-a-real-time-hand-detector-using-neural-networks-ssd-on-tensorflow-d6bac0e4b2ce

- https://github.com/tensorflow/models/issues/4467
