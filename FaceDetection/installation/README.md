# Object Detection API 및 Tensorflow, Python Library 설치

## Tensorflow, Python Library 설치

자신의 환경에 맞게 pip를 통해 tensorflow와 python library들을 설치합니다.

```bash
# CPU
pip install tensorflow
# GPU (GPU 버전의 자세한 설치는 다른 인터넷 자료를 참고)
pip install tensorflow-gpu
```

```bash
# 아래의 파이썬 라이브러리들을 설치
pip install Cython
pip install contextlib2
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
```

## Tensorflow Object Detection API 설치

설치를 원하는 경로로 이동 후, git clone을 이용해 Tensorflow Object Detection API를 git clone을 이용해 내려받습니다.

경로가 꼬이면 안되므로 여기서는 /home/user 경로에 tensorflow라는 디렉토리를 만들고 이곳에 내려받는다고 가정하고 진행하겠습니다.

```bash
user@ubuntu:~$ mkdir tensorflow
user@ubuntu:~$ cd ./tensorflow
user@ubuntu:~/tensorflow$ git clone https://github.com/tensorflow/models.git
```

## COCO API 설치

Tensorflow의 학습된 모델은 COCO dataset에 대해 학습이 된 모델이므로 우리의 데이터를 학습시키는 과정에서도 COCO dataset을 학습하는 과정에서 사용한 여러 함수들을 사용하게 됩니다.

따라서 아래와 같이 git clone을 통해 COCO API를 내려받습니다.

COCO API를 clone하는 위치는 상관이 없지만 내려받은 후 make를 통해 설치된 pycocotools 디렉토리의 경로는 반드시 위에서 내려받은 tensorflow/models/research/ 위치해야합니다.

```bash
user@ubuntu:~/tensorflow$ git clone https://github.com/cocodataset/cocoapi.git
user@ubuntu:~/tensorflow$ cd ./cocoapi/PythonAPI
user@ubuntu:~/tensorflow/cocoapi/PythonAPI$ make
# <path_to_tensorflow>/models/research/
user@ubuntu:~/tensorflow/cocoapi/PythonAPI$ cp -r pycocotools ~/tensorflow/models/research/
```

## Protobuf 컴파일

Tensorflow Object Detection API는 모델과 학습 파라미터들을 설정하는데에 Protobuf를 사용합니다. 따라서, 학습 전에 이를 컴파일하는 과정이 필요합니다.

따라서, tensorflow/models/research 경로로 이동 후, 다음과 같이 컴파일합니다.

```bash
user@ubuntu:~/tensorflow/cocoapi/PythonAPI$ cd ~/tensorflow/models/research
user@ubuntu:~/tensorflow/models/research$ protoc object_detection/protos/*.proto --python_out=.
```

## PYTHONPATH 환경변수 설정

Tensorflow Object Detection API를 사용하기 위해서는 tensorflow/models/research/와 tensorflow/models/research/slim/ 디렉토리 경로를 PYTHONPATH라는 환경변수에 설정해주어야 합니다.

따라서 ~/.bashrc 파일에 환경변수 설정을 추가한 후, source를 통해 적용해줍니다. (zsh이나 다른 shell을 쓰는 경우 각 shell의 해당하는 파일)

```bash
# 반드시 tensorflow/models/research 경로에서 실행
user@ubuntu:~/tensorflow/models/research$ echo export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim >> ~/.bashrc
user@ubuntu:~/tensorflow/models/research$ source ~/.bashrc
```

~/.bashrc에 추가하지 않고 매 터미널 실행 시 마다 다음과 같이 export를 해 주어도 됩니다.

```bash
# 반드시 tensorflow/models/research 경로에서 실행
user@ubuntu:~/tensorflow/models/research$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim >> ~/.bashrc
user@ubuntu:~/tensorflow/models/research$ source ~/.bashrc
```

# 설치 확인

마지막으로 tensorflow/models/research/object_detection/builders/model_builder_test.py 파일을 실행시켜 설치가 잘 되었는지 확인합니다.

```bash
user@ubuntu:~/tensorflow/models/research$ python ./object_detection/builders/model_builder_test.py
```

다음과 같은 출력이 나온다면 설치가 잘 된 것입니다.

```bash
[ RUN      ] ModelBuilderTest.test_create_faster_rcnn_model_from_config_with_example_miner
[       OK ] ModelBuilderTest.test_create_faster_rcnn_model_from_config_with_example_miner
[ RUN      ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
[       OK ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
[ RUN      ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
[       OK ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
[ RUN      ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
[       OK ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
[ RUN      ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
[       OK ] ModelBuilderTest.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
[ RUN      ] ModelBuilderTest.test_create_rfcn_model_from_config
[       OK ] ModelBuilderTest.test_create_rfcn_model_from_config
[ RUN      ] ModelBuilderTest.test_create_ssd_fpn_model_from_config
[       OK ] ModelBuilderTest.test_create_ssd_fpn_model_from_config
[ RUN      ] ModelBuilderTest.test_create_ssd_models_from_config
[       OK ] ModelBuilderTest.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTest.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTest.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTest.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTest.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTest.test_invalid_model_config_proto
[       OK ] ModelBuilderTest.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTest.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTest.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTest.test_session
[  SKIPPED ] ModelBuilderTest.test_session
[ RUN      ] ModelBuilderTest.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTest.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTest.test_unknown_meta_architecture
[       OK ] ModelBuilderTest.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTest.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTest.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 16 tests in 0.129s

OK (skipped=1)
```
