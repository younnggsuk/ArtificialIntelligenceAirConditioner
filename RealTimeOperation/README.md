# Real time operation

OpenCV를 사용해 실시간으로 읽어들인 영상에서 앞서 구현한 모델을 사용해 명령을 받아들이도록 구현하였습니다.

아래의 각 파일에 구현된 함수 및 코드에 대한 설명은 주석으로 달아놓았습니다.

- `my_utils.py`
  - 모든 얼굴 감지, 얼굴 추적, 손모양 인식 등의 함수들이 구현된 파일입니다.

- `run.py`
  - 메인함수가 있는 파일입니다.



## Environments

- Linux Ubuntu 18.04
- tensorflow 2.1.0
- python 3.7.6



## Commands

7가지 손모양이 의미하는 명령의 역할은 다음과 같이 가정하고 구현하였습니다.

여기서는 해당 class label을 출력하도록 구현하였습니다.

![hand_signs](/home/young/ML/Project/CapstoneDesign/RealTimeOperation/hand_signs.png)

5_front : 명령을 내리겠다는 신호

0_front : ON / OFF

1_back : 온도 감소

1_front : 온도 증가

2_back : 풍량 감소

2_front : 풍량 증가

ILU : 회전



## Flow chart

main 함수에서 동작하는 프로그램의 흐름은 다음과 같습니다.

![flow_chart](/home/young/ML/Project/CapstoneDesign/RealTimeOperation/flow_chart.jpg)