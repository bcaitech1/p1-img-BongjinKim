# 마스크 착용 상태 분류
사람의 사진을 입력으로 받아 마스크의 유무, 나이, 성별을 18개의 클래스로 분류하는 이미지 분류 Task

- `input` : 마스크 착용 사진, 미착용 사진, 혹은 이상하게 착용한 사진(코스크, 턱스크)


- `output` : 총 18개의 class를 예측해야합니다. 결과값으로 0~17에 해당되는 숫자가 각 이미지 당 하나씩 나와야합니다.
        
    |Class|Mask|Gender|Age|
    |---|---|---|---|
    |0|Wear|Male|<30|
    |1|Wear|Male|>=30 and <60|
    |2|Wear|Male|>=60|
    |3|Wear|Female|<30|
    |4|Wear|Female|>=30 and <60|
    |5|Wear|Female|>=60|
    |6|Incorrect|Male|<30|
    |7|Incorrect|Male|>=30 and <60|
    |8|Incorrect|Male|>=60|
    |9|Incorrect|Female|<30|
    |10|Incorrect|Female|>=30 and <60|
    |11|Incorrect|Female|>=60|
    |12|Not Wear|Male|<30|
    |13|Not Wear|Male|>=30 and <60|
    |14|Not Wear|Male|>=60|
    |15|Not Wear|Female|<30|
    |16|Not Wear|Female|>=30 and <60|   
    |17|Not Wear|Female|>=60|
   

- `metric` : f1-score

## Dataset

- 전체 사람 명 수 : 4,500


- 한 사람당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]


- 이미지 크기: (384, 512)
## Code
### Files
- `dataset.py`

    - 마스크 데이터셋을 읽고 전처리를 진행한 후 데이터를 하나씩 꺼내주는 Dataset 클래스를 구현한 파일입니다.

    - 이 곳에서, 나만의 Data Augmentation 기법 들을 구현하여 사용할 수 있습니다.


- `loss.py`
    - 이미지 분류에 사용될 수 있는 다양한 Loss 들을 정의한 파일입니다

    - 이외에, 성능 향상을 위한 다양한 Loss 를 정의할 수 있습니다.


- `model.py`

    - 데이터를 받아 연산을 처리한 후 결과 값을 내는 Model 클래스를 구현하는 파일입니다.

    - 이 곳에서, 다양한 CNN 모델들을 구현하여 학습과 추론에서 사용할 수 있습니다.


- `train.py`

    - 실제로, 마스크 데이터셋을 통해 CNN 모델 학습을 진행하고 완성된 모델을 저장하는 파일입니다.

    - 다양한 hyperparameter 들과 커스텀 model, data augmentation 등을 통해 성능 향상을 이룰 수 있습니다.


- `inference.py`

    - 학습 완료된 모델을 통해 test set 에 대한 예측 값을 구하고 이를 .csv 형식으로 저장하는 파일입니다.
    

- `f1.py`

    - f1 스코어를 계산하는 파일입니다.

### Install Libraries
```python
pip install -r requirements.txt
```
### Trainng
```
python train.py
```
- 기본으로 설정된 hyperparameter로 train.py 실행합니다.

### Inference
```python
python inference.py --model_dir=./results/checkpoint-500
```
- 학습된 모델을 추론합니다.

- 제출을 위한 csv 파일을 만들고 싶은 model의 경로를 model_dir에 입력해 줍니다.

- 오류 없이 진행 되었다면, submission.csv 파일이 생성 됩니다.

## Competition
### Final Score
![image](https://media.vlpt.us/images/loulench/post/77f75ac3-1b67-4d2d-98cf-2672179d931e/image.png)
### Work flow
|Model|Tecnics|F1-Score|
|---|---|---|
|resnet|params not tuned|0.39|
|vgg19|params not tuned|0.50|
|vgg19|params tuned|0.51|
|vgg11|params tuned|0.56|
|efficientnet_b4|params not tuned|0.60|
|efficientnet_b4|params tuned, weighted sample|0.62|
|efficientnet_b4|params tuned, weighted sample, focal|0.63|
|efficientnet_b4|params tuned, loss weight, focal, augmentation|0.71|
|ensemble|efficientnet_b4, vgg11, resnet|0.73|

### References


