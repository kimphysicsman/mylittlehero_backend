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

# 🔑 핵심기능

### 1. 여행장소 검색

> 사용자가 여행장소를 검색하면 DB에서 여행장소를 검색하고  
> DB에 없는 장소이면 네이버지도에서 검색하여 최상단의 장소의 정보를 가져오고 DB에 저장합니다.  
> [코드 보러가기](https://github.com/nbcamp-AI-2-fantastic4/MyLittelTrip_backend/blob/d9eba0efc4567cbaef9ec19eea76e76495190a69/recommend/functions/parsing.py#L70)

### 2. 최단 여행경로 찾기 & 여행일정 만들기

> 사용자가 입력한 여행장소들을 바탕으로 여행일정을 만듭니다.  
> [코드 보러가기](https://github.com/nbcamp-AI-2-fantastic4/MyLittelTrip_backend/blob/d9eba0efc4567cbaef9ec19eea76e76495190a69/recommend/functions/schedule.py#L14)

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

