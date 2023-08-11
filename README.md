# AutoDriveAI
자율주행 인공지능 알고리즘 개발 챌린지

[한국안전교통공단 데이터셋](http://challenge.gcontest.co.kr/template/m/12709#)

### CustomDataset 클래스
목적: 지정된 디렉토리에서 사용자 지정 데이터셋 이미지와 레이블을 로드합니다.

#### 속성:
root_dir: 데이터셋의 루트 디렉토리.
image_dir: 이미지가 저장된 디렉토리.
label_dir: 레이블 JSON 파일이 위치한 디렉토리.
image_files: 이미지 파일 이름 목록.
label_map: 문자열 레이블을 정수 값에 매핑하는 dictionary.
#### 메소드
__len__: 데이터셋에 있는 이미지의 총 개수를 반환합니다.
__getitem__: 인덱스가 주어지면 객체 검출 작업에 필요한 형식의 이미지와 해당 대상을 반환합니다.

### 데이터 로더
훈련, 검증 및 테스트 데이터셋에 대한 세 개의 데이터 로더가 생성됩니다:
train_loader: 훈련 데이터셋을 위한 DataLoader.
val_loader: 검증 데이터셋을 위한 DataLoader.
test_loader: 테스트 데이터셋을 위한 DataLoader.

### 모델 생성
모델은 ResNet-50 backbone과 Feature Pyramid Network (FPN)을 갖는 Faster R-CNN입니다.
get_model 함수는 Faster R-CNN 모델을 생성하고 이 부분에 classifier 코드를 짤 예정입니다.
사전 학습된 Faster R-CNN 모델이 로드되고, 우리 데이터셋의 10개 클래스를 처리하기 위해 그 머리 부분(박스 예측기)이 교체됩니다.

### 훈련 루프
Optimizer: 학습률 0.001, 모멘텀 0.9 및 가중치 감소 0.0005로 SGD입니다.
조기 종료: 모델의 평균 손실을 기반으로 조기 종료 로직이 훈련 루프에 포함되어 있습니다. 손실이 지정된 에폭(patience) 동안 개선되지 않으면 훈련이 조기에 중단됩니다.

### 객체 검출 모델 평가

1. 검증 데이터셋에 대한 정확도 계산
model.eval(): 모델을 평가 모드로 전환합니다. 이렇게 하면 드롭아웃 및 배치 정규화와 같은 계층이 변경되지 않습니다.
torch.no_grad(): 기울기 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높입니다.

이 코드는 예측 라벨과 실제 라벨을 비교하여 검증 세트에서의 정확도를 계산합니다.

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, targets in val_loader:
        ...
        for index, output in enumerate(outputs):
            ...
accuracy = 100 * correct / total
print(f"Accuracy on validation set: {accuracy:.2f}%")
```
2. 유틸리티 함수
xywh_to_xyxy(boxes): bounding box의 좌표 형식을 [x, y, width, height]에서 [x_min, y_min, x_max, y_max]로 변환합니다.
box_iou(boxes1, boxes2): 두 세트의 bounding boxes 간의 IoU (Intersection over Union) 값을 계산합니다.

3. AP (Average Precision) 및 mAP (mean Average Precision) 계산
calculate_AP(det_boxes, det_scores, true_boxes, true_labels, class_idx): 주어진 클래스에 대한 AP 값을 계산합니다.
calculate_mAP(model, data_loader, device, num_classes=10): 모델의 전체 mAP 값을 계산합니다.

```python
def calculate_AP(det_boxes, det_scores, true_boxes, true_labels, class_idx):
    ...

def calculate_mAP(model, data_loader, device, num_classes=10):
    ...
mAP = calculate_mAP(model, val_loader, device)
print(f"mAP: {mAP:.4f}")
```
[mAP, ioU 식 설명](https://velog.io/@blublue_02/IoU-mAP)
