# Real time operation

OpenCV를 사용해 실시간 영상에서 앞서 구현한 모델을 사용해 명령을 받아들이도록 구현하였다.

원래는 Jetson Nano에서 모든 연산을 처리하려고 하였지만 성능이 부족해 제대로 동작하지 않아서 딥러닝 연산은 다른 PC에서 처리하도록 TCP 소켓 통신으로 구현하였다.

따라서, 아래 구현 내용의 client는 Jetson Nano가 되고 server는 딥러닝 연산용 PC가 된다.


## [Client](./Client)
`run_client.py`
- Webcam 영상 이미지를 server로 전송 후 처리된 결과 및 명령을 받아 display에 그리는 main() 함수

`client_utils.py`
- 에어컨의 동작 및 현재 상태가 표시된 control panel을 관리하는 AirConditioner 클래스
- Server와 데이터를 주고 받는 함수들

## [Server](./Server)
`run_server.py`
- Client의 영상 이미지를 받아 face detection & tracking을 하며, hand area(face의 오른쪽 영역)에 대해 classification을 수행
- 위 과정에서 hand area의 bounding box가 그려진 영상 이미지를 계속해서 client로 전송하고, classification 결과(command)가 나오면 영상 이미지와 같이 전송

`server_utils.py`
- Face detection, tracking
- Hand sign classification
- Client와 데이터를 주고 받는 함수들