# **BARCODE LESS**
![barcodeless](https://user-images.githubusercontent.com/89384669/146715669-a4b7d5cf-f225-428b-a8a7-d743db8969bc.png) 

----

## **yoloV4** (Object Detection)
 - ### [Darknet yoloV4](https://github.com/AlexeyAB/darknet)
 - #### 학습 환경 설정   
 
| 환경 | 버전 |
| :------------: | :------------: |
| Ubuntu | 18.04 |
| GPU | Nvidia-A100 |
| Driver | 470.86 |
| Cuda | 11.4 |
| Cuda Toolkit | 11.1.105 |
| CUDnn | 8.0.5 |
| Python | 3.8.8 |
| Cmake | 3.22.0 |
| OpenCV | 4.5.4 |

### yoloV4 를 사용한 이유
 - 간단한 처리 과정으로 속도가 매우 빠르며 기존의 실시간 Object Detection 모델들과 비교하면 2배 정도 높은 mAP를 보여준다. 
 - 이미지 전체를 한 번에 바라보는 방식을 이용하므로 class에 대한 맥락적 이해도가 다른 모델에 비해 높아 낮은 False-Positive를 보여준다.
 - 일반화된 Object 학습이 가능하여 자연 이미지로 학습하고 이를 그림과 같은 곳에 테스트 해도 다른 모델에 비해 훨씬 높은 성능을 보여준다.

### yoloV4의 단점
 - 다른 모델에 비해 낮은 정확도
 - 작은 객체애 대한 정확도가 낮은 단점 (상품 이미지를 찾는 프로젝트에서 작은 객체를 탐지할 일이 없기 때문에 사용)


<br><br>

## **yoloV4 학습 방법**
----
### 1. 데이터 라벨링
 - 이미지 - bar_8694.jpeg 

<img src="https://user-images.githubusercontent.com/70836261/147017717-33bdec15-09ef-48ae-98c1-6dabc33eecfc.jpeg" width="480"/>

 - 레이블 - bar_8694.txt
 ```
0 0.547443 0.508528 0.231844 0.690241
 ```

### 2. train.txt, valid.txt 파일 만들기
 - train 과 valid 에 쓰일 이미지의 경로를 저장하는 파일
 ```
data/obj/bar_5515.jpeg
data/obj/corn_3818.jpeg
data/obj/bar_8576.jpeg
data/obj/corn_5557.jpeg
data/obj/bar_11174.jpeg
data/obj/drink_451.jpeg
 ```


### 3. obj.names, obj.data 파일 만들기
 - **obj.names** - 클래스 이름 정의 파일 
 ```
bar
bucket
corn
drink
sand
snack
 ```
 - **obj.data** - train, valid 이미지 경로 설정 파일
 ```
classes= 6
train  = data/train.txt
vaild = data/valid.txt
names = data/obj.names
backup = backup/
 ```
### 4. cfg 파일 만들기
 - git clone 해온 **darknet/cfg/yolov4-obj.cfg** 파일을 수정 <br>
    학습관련 파라미터들을 설정하는 파일 
 ``` 
aumentation을 위해 추가
angle=30
blur=1
mosaic=1

6번째 줄 batch = 32 (디폴트값은 64였으나 cuda memory 부족으로 32로 변경)
7번째 줄 subdivisions = 16 (디폴트값은 32였는데 16으로 변경)
8,9번째 줄 width, height 학습할 이미지 크기. 416으로 설정
20번째 줄 max_batches = 12000 (클래스수 * 2000으로 설정)
22번쨰 줄 steps = 9600, 10800 (max_batches의 80%, 90% 로 설정)
963번째 줄 filters = 33 (클래스수+5) * 3 으로 설정
970번째 줄 classes = 6 클래스수
1051번째 줄 filters = 33 (클래스수+5) * 3 으로 설정
1058번째 줄 classes = 6 클래스수
1139번째 줄 filters = 33 (클래스수+5) * 3 으로 설정
1146번째 줄 classes = 6 클래스수

 ```
### 5. 사전학습된 yolov4.conv.137 파일을 받아와서 학습을 진행 한다
```bash
# darknet/
./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -mjpeg_port 9999 -map
# ./darknet detector train {obj.data 파일경로} {yolov4-obj.cfg 파일경로} {volov4.weights 파일 경로}
# -dont_show            콘솔 환경에서 이미지를 보지 않기 위해 샤용
# -mjpeg_port 9999      loss 값과 mAP 값을 웹페이지로 보기 위해 사용
# -map                  loss값과 함께 mAP 값을 보기 위해 사용
```
 - 결과물이 **backup/yolov4-obj.weights** 파일로 저장된다 (best, last 값이 따로 저장된다)


### 6. 학습된 결과 테스트
 ```bash
 ./darknet detector test data/obj.data cfg/yolov4-obj.cfg yolov4-obj.weights -thresh 0.25 {테스트 이미지 파일}
 ```


<br><br><br>

----
## **EfficientNet** (Image Classification)
 - ### [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)
 - #### 학습 환경 설정   
 
| 환경 | 버전 |
| :------------: | :------------: |
| GPU | RTX2060 |
| Cuda | 10.4 |
| Cuda Toolkit | 10.2.89 |
| CUDnn | 7.6.5 |
| Python | 3.7.7 |
| Cmake | 3.17.2 |
| OpenCV | 4.1.0 |
| Pytorch | 1.10.0 |
| Torchvision | 0.11.1 |
| EfficientNet-Pytorch | 0.7.1 |


### EfficientNet 모델의 특징
 - 적은 파라미터로 효율적인 성능을 내려던 **Inception** 과 쉬운 아키텍쳐 구성으로 높은 성능을 내려던 **ResNet** 의 결합으로 만들어진 모델
 - 기존의 CNN 모델의 3가지 Scaling 방법 (Width, depth, resolution)을 EfficientNet은 3가지 Scaling을 동시에 고려하는 **Compound Scaling** 을 통해 모델의 정확도를 높여주었다

 ![Compound Scaling](https://blog.kakaocdn.net/dn/b0YwHo/btqDVWmQTYg/kz4f74uAUQyHk0KJK6j4k0/img.png)


### EfficientNet 을 사용한 이유
 - ConvNet의 성능을 높이기 위해 Scaling 을 시도 하는데 EfficientNet은 위의 3가지 방법에 대한 최적의 조합을 찾는 연구를 했고 **Compound Scaling** 을 사용하고 다른 모델들과 비교해서 파라미터는 1/5 수준이고 Inference 시간도 11배 빠른 속도를 보여준다
 - 이미지의 빠른 처리를 위해 **Image Classification** 에서 매우 높은 속도와 성능을 보여주는 EfficientNet을 사용했다 


### EfficientNet의 단점
 - ResNet 계열의 모델에 비해 메모리 사용량이 높은 단점이 있다

### [추가] EfficientNetV2(2021)
 - EfficientNet의 단점인 높은 메모리 사용량을 줄이고
 4배 빠른 학습 시간과 6.8배 적은 파라미터로 비슷한 정확도를 보여준다 
 - 기존 방식보다 좀더 나은 학습 방법인 **Prograssive Learning** 을 사용 한다
 빠른 학습속도가 장점 이지만 정확도가 감소하는 문제점이 있었지만
 EfficientNetV2 에서는 이미지 크기에 따라 정규화 방법을 다르게 하여 정확도가 감소하는 문제점을 해결했다 

<br><br>

## **EfficientNet 학습 방법**

### 1. 데이터 레이블링
 - 클래스별 폴더로 이미지 분류를 통해 레이블링
   - /bar
   - /cone
   - /sand
   - /bucket
   - /snack
   - /drink

### 2. 사전학습된 efficientnet-b0 모델을 불러온다
 - **pip install efficientnet_pytorch** 명령어로 설치하고
 ```python
from efficientnet_pytorch import EfficientNet
model_name = 'efficientnet-b0' 
model = EfficientNet.from_pretrained(model_name, num_classes=10)  # num_classes 원하는 클래스 개수 만큼 설정!
 ```

### 3. Train, valid 데이터 셋을 만들고 Augmentation 한다
 - **sklearn** 의 **train_test_split** 을 이용해서 **train**, **valid** 데이터셋을 나누고
 - train 데이터셋을 **torchvision** 의 transforms 을 이용해 **Augmentation** 했다
   - **RandomRotation(30)**
   - **RandomHorizontalFlip()**

### 4. 클래스명 정의 
 - Snack 데이터 예시
```python
class_names = {
    '0':'롤리폴리',
    '1':'롯데샌드',
    '2':'마가렛트',
    '3':'몽쉘',
    '4':'빅파이',
    '5':'빠다코코낫',
    '6':'찰떡파이',
    '7':'초코쿠키',
    '8':'쿠크다스',
    '9':'포키'
}
```

### 5. EfficientNet 학습 진행
 - 각 클래스별로 6개의 모델 학습 진행 
   - 모델 : efficientnet-b0
   - criterion : torch.nn.CrossEntropyLoss (다중 분류를 위한 손실함수) 
   - optimizer : torch.optim.SGD (최적화 함수)
   - scheduler : torch.optim.lr_scheduler.MultiplicactiveLR
   - epochs : 10

### 6. 테스트 
 - opencv 를 통해 불러온 이미지를 다음 과정을 통해 프로세스진행 
 ```python
items = []

model.eval()
with torch.no_grad():
    outputs = model(img)

# Efficient 결과로 나온 예측 % 값중에 가장 큰 값을 items 리스트로 저장     
for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():    
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('[', class_names[str(idx)], ': {p:.2f}% ]'.format(p=prob*100))
    items.append(class_names[str(idx)])
print('-----')
 ```

 - 테스트 결과
 
<image src="https://user-images.githubusercontent.com/70836261/147040222-2f1fbe00-b7a6-47ff-8729-392e6c5a7336.png" width="360">



<br><br><br>


## BARCODELESS [DJANGO]

### 장고에서 이미지 파일을 YOLO, EfficientNet 결과로 산출하기  

Step 1: **[Django, HTML, Java Script]** Input Images

Step 2: **[Darknet-yoloV4]** Object detection

Step 3: **[OpenCV]** Image Crop, Merge Backdrop

Step 4: **[Pytorch EfficientNet-b0]** Image Classification 

<br><br>

----

<br><br>

### 웹 서비스 

<br>
<div>
<image src="https://user-images.githubusercontent.com/70836261/147031140-3387202d-ed2e-4f8c-b11f-e139da38f8fc.png" width="360">
<image src="https://user-images.githubusercontent.com/70836261/147031207-46ccd28c-b4cb-45cb-9e49-cb37487622b1.png" width="360">
</div> 


- 카메라 기능 
   - **`cam.html`** 에서 사진 촬영 또는 업로드 버튼을 통해 Image Input을 할 수 있습니다.
   - 사진 촬영 시, 이미지 파일은 장고의 미디어 폴더에 저장됩니다.  

- 전송 기능
   - **`cam.html`** 의 '촬영하기' 버튼을 누르면 사진 파일을 서버로 전송하여 입력 데이터로 가공합니다. 

- 모델 기능
   - **`Darknet-yoloV4`** 에서 객체 감지를 한 후, 
   OpenCV에서 이미지 크롭 및 머지 백드롭(Merge Backdrop)을 한 후 **`Pytorch EfficientNet-b0`** 에서 이미지 분류가 진행됩니다. 
   - 연산 결과 반환
    ```python
   [ # [상품명, 가격, 수량], 
       ["주물러",500,2],
       ["캐나다드라이",1100,2],
       ["구구크러스터",4500,1],
       ["누가바",500,2],
   ]
   ```

- 장바구니 기능
   -  **`main.html`** 화면에서 모델 연산 후의 결과가 상품 리스트로 출력됩니다.
   -  상품명, 수량, 가격 합계 등의 기능을 제공합니다. 

<br><br><br>
 
----

<br><br><br>
 
 ### yoloV4 학습된 weights 파일 다운로드
  - 다음 경로에 저장 `/barcodeless/prediction/pred_yolo/yoloV4-obj_best.weights`
 
 [yoloV4-obj_best.weights](https://drive.google.com/file/d/1VtDGNmMHr4N_5y4Ng0wFneTFLCkooTKr/view?usp=sharing)
 
<br><br><br>
 
----

<br><br><br>

## 절대 경로 수정 필요

**`barcodeless\views.py`**
39 번째 줄


**`barcodeless\prediction\pred_yolo\darknet_cv.py`**
82 번째 줄


**`barcodeless\prediction\pred_yolo\yolo_effi.py`**
108, 131, 168, 170 번째 줄 

