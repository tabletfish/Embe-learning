# Object

인지 학습 테스트하기 위한 목적임.

# Dataset

https://drive.google.com/file/d/1u-MRZrUtB1MAKlQTRKYnn4oIJacdslQl/view?usp=drive_link
(동영상 촬영 후, 1초 프레임단위로 사진저장한것 ㅇㅇ)

# Labeling

roborflow웹으로 labeling 및 학습데이터 분류까지 진행해야됨 ㅇㅇ --> 이게 진짜 노가다

# 데이터 테스트

  yolo predict task=detect model=/home/dydlz/Downloads/best.onnx source=0
  imgsz=640 device=cpu show=True

  종료는 카메라 창에서 q.


