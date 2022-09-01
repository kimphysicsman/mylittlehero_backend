# 🦸‍♂️ MyLittleHero

나와 닮은 마블 히어로 캐릭터 찾기  

<br />

# 📃 프로젝트 정보

### 1. 제작기간

> 2022.05.18 ~ 05.25

### 2. 참여 인원

> |                    Name                    |  Position   |
> | :----------------------------------------: | :---------: |
> | [김동우](https://github.com/kimphysicsman) | Back, Front |
> |   [김진수](https://github.com/creamone)    | Back, Front |
> |     [이윤지](https://github.com/J1NU2)     | Back, Front |
> |    [최민기](https://github.com/mankic)     | Back, Front |

### 3. 역할 분담

> - 김동우 : 메인페이지 + 결과 보여주기 + InceptionV3 모델 학습
> - 김진수 : 메인페이지 - 모달창 / 헤더 디자인  + 이미지 업로드 (모달창) + MobileNetV2 모델 학습
> - 이윤지 : 히스토리페이지 + 결과 보여주기 + ResNet50 모델 학습
> - 최민기 : 로그인 / 회원가입 + VGG16 모델 학습

<br />

# 📚 기술 스택

### 1. Back-end

> python3  
> Flask  
> Keras  
> mongo DB  

<br />

# 📊 ERD 

<details>
<summary>ERD</summary>
<div markdown="1" style="padding-left: 15px;">
<img src="https://user-images.githubusercontent.com/104434422/186101629-0fe314ae-6ab8-4145-a6e7-4121325268e9.png" width="800px"/>
</div>
</details>

<br />

<!-- <details>
<summary>Structure</summary>
<div markdown="1" style="padding-left: 15px;">
<img src="" />
</div>
</details>

<br /> -->

# 🔑 구현내용

### 1. CNN 모델별 학습 및 성능비교

<details>
<summary>VGG16 (최민기)</summary>
<div markdown="1" style="padding-left: 15px;">

- 간단 설명 :3X3의 작은 사이즈의 필터를 고정으로 사용해서 레이어를 깊게 만든다.  
- 장점 : 필터 사이즈가 작아서 파라메터 개수가 줄어든다 ⇒ 학습효율성, 정규화할때 이점  
- 단점 : 레이어가 깊어 메모리 차지를 많이하고 학습속도가 느리다.  
- 학습결과 : 27%…
- 추가 레이어 없이
- kernel_regularizer=l2(.001)
- Adam(lr=.0001)
- 이미지증강 0.2

<img width="400" src="https://user-images.githubusercontent.com/68724828/187684340-636ced7d-4aa7-417e-bdcd-a25f621b51b4.png" />
</div>
</details>

<details>
<summary>MobileNet V2 (김진수)</summary>
<div markdown="1" style="padding-left: 15px;">

- 간단 설명 : 모바일이나 리소스가 제약된 환경을 위한 것(가벼운 모델을 만들기 위해 만들어진 기법)
- 장점 : 파라미터값이 타모델들에 비해 상대적으로 적게 사용되고 같은 정확도를 가지면서도 연산수를 크게줄이고 사용되는 메모리를 줄여준다.
- 단점 : 네트워크가 매우 복잡하다.
- 학습결과 : 0.65

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2 


input = Input(shape=(224, 224, 3))

base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input, pooling='max')

x = base_model.output
x = Dropout(rate=0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(rate=0.5)(x)
output = Dense(8, activation='softmax', kernel_regularizer=l2(.1))(x)

model = Model(inputs=base_model.input, outputs=output)

opt = Adam(lr=.0001, beta_1=.99, beta_2=.999, epsilon=1e-8)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

model.summary()
```

<img width="800" src="https://user-images.githubusercontent.com/68724828/187684917-6313a0e0-c6ce-4e90-8beb-72f98c071aab.png" />
</div>
</details>

<details>
<summary>Inception V3 (김동우)</summary>
<div markdown="1" style="padding-left: 15px;">

- 간단 설명 : 1x1 filter로 채널 수를 줄여 연산량을 줄이고 여러크기의 filter의 output을 concat하는 구조
- 장점 : 여러크기의 filter를 사용하여 다양한 크기의 이미지 특징을 찾을 수 있다. 
- 단점 :  네트워크 구조가 매우 깊고 복잡하다.
- 학습결과 : 최대 val_cc : 0.6430

```python
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.regularizers import l2 


input = Input(shape=(224, 224, 3))

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input, pooling='max')

x = base_model.output
x = Dropout(rate=0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(rate=0.5)(x)
output = Dense(8, activation='softmax', kernel_regularizer=l2(.1))(x)

model = Model(inputs=base_model.input, outputs=output)

opt = Adam(lr=.0001, beta_1=.99, beta_2=.999, epsilon=1e-8)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

from tensorflow.keras.callbacks import ModelCheckpoint

history = model.fit(
    train_gen,
    validation_data=test_gen, # 검증 데이터를 넣어주면 한 epoch이 끝날때마다 자동으로 검증
    epochs=10, # epochs 복수형으로 쓰기!
    callbacks=[
      ModelCheckpoint('drive/MyDrive/mylettlehero_model/model.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)
```

<img width="800" src="https://user-images.githubusercontent.com/68724828/187685280-201d4da1-895f-49c2-9370-036fb22b0094.png" />

```python

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['acc'])

history = model.fit(
    train_gen,
    validation_data=test_gen, # 검증 데이터를 넣어주면 한 epoch이 끝날때마다 자동으로 검증
    epochs=10, # epochs 복수형으로 쓰기!
    callbacks=[
      ModelCheckpoint('model_2.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)

```

<img width="800" src="https://user-images.githubusercontent.com/68724828/187686502-4a2f51b6-244e-42c4-b81f-1d644212a934.png" />

</div>
</details>

<details>
<summary>ResNet 50 (이윤지))</summary>
<div markdown="1" style="padding-left: 15px;">

- 간단 설명 : https://velog.io/@arittung/CNN-ResNet50
- 장점 :  스킵 연결(Skip connection)은 층이 깊어져도 학습을 효율적으로 할 수 있다.
- 단점 : input과 output의 dimension을 통일해줘야한다.
- 학습결과 : 최대 val_acc: 0.6563

```python
from tensorflow.keras.callbacks import ModelCheckpoint

opt = Adam(lr=.0001, beta_1=.99, beta_2=.999, epsilon=1e-8)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

history = model.fit(
    train_gen,
    validation_data=test_gen, # 검증 데이터를 넣어주면 한 epoch이 끝날때마다 자동으로 검증
    epochs=10, # epochs 복수형으로 쓰기!
    callbacks=[
      ModelCheckpoint('drive/MyDrive/mylittlehero/model.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)
```
<img width="800" src="https://user-images.githubusercontent.com/68724828/187685890-0c42a46f-32e6-49d0-a2b9-c2c9179d4e83.png" />

```python
model = load_model('drive/MyDrive/mylittlehero/model.h5')

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['acc'])

history = model.fit(
    train_gen,
    validation_data=test_gen, # 검증 데이터를 넣어주면 한 epoch이 끝날때마다 자동으로 검증
    epochs=10, # epochs 복수형으로 쓰기!
    callbacks=[
      ModelCheckpoint('model_2.h5', monitor='val_acc', verbose=1, save_best_only=True)
    ]
)
```
<img width="800" src="https://user-images.githubusercontent.com/68724828/187686093-67c97221-0fb9-4544-bcd9-314a1fb4fb18.png" />
</div>
</details>

<br />

### 2. 학습 결과에 따른 MobileNet 모델 선정

- **선정 이유**  
팀원별 CNN모델 학습 결과 VGG16을 제외한 모델들에서 validation accuracy가 0.65정도로 비슷한 성능을 냄.  
따라서 그 중에서 가볍고 학습속도가 빠른 MobileNetV2를 이용  

> <img width="600" src="https://user-images.githubusercontent.com/68724828/187687041-ac7db1a3-5b86-47fd-93b4-63600e931519.png" />  <br/>
> MobileNetV2 학습 그래프, epoch 20 이후부터는 overfitting이 일어나서 그전까지만 학습한 모델을 이용했다.


<br />

### 3. 이미지 저장
> 사용자 이메일과 저장 시간을 파일 이름으로 하여 이미지 파일을 저장  
> [코드 보러가기](https://github.com/kimphysicsman/mylittlehero_backend/blob/master/functions/common.py#L50)

<br />

### 4. 닮은 마블 캐릭터 추천
> 입력받은 이미지 주소로 저장한 이미지를 불러와서 전처리 후 학습 시킨 모델로 예측, 예측 확률이 높은 순으로 정렬하여 반환  
> [코드 보러가기](https://github.com/kimphysicsman/mylittlehero_backend/blob/master/functions/model.py#L21)


<br />

# 📕 기타 자료

### 1. 기획문서

> [MyLittleHero - Notion](https://www.notion.so/kimphysicsman/My-Little-Hero-13b315a07f1940c79ddc81ad06c79fd0)

### 2. 히어로 데이터셋

> [Marvel Heroes - Kaggle](https://www.kaggle.com/datasets/hchen13/marvel-heroes)

### 3. 발표영상

<table>
  <tbody>
    <tr>
      <td>
        <p align="center"> 22.05.25 발표 </p>
        <a href="https://youtu.be/EVSkMMqKbns" title="MyLittleHero 최종발표">
          <img align="center" src="https://user-images.githubusercontent.com/104434422/186104467-2ef162b7-fe34-4d8c-abb3-a0c6858a5fea.png" width="300" >
        </a>
      </td>
    </tr>
  </tbody>
</table>

