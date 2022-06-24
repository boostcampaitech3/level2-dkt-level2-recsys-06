 

# 📕 DKT
><p align="center"><img src=https://user-images.githubusercontent.com/58590260/168202720-1a9ac668-d1e3-4fd9-ad6a-c5d83e841a9f.png width=50%></p><br>
>DKT를 활용하면 우리는 학생 개개인에게 수학의 이해도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능하여 DKT는 맞춤화된 교육을 제공하기 위해 아주 중요한 역할을 맡게 된다. 시험을 보는 것은 동일하지만 단순히 우리가 수학을 80점을 맞았다고 알려주는 것을 넘어서 우리가 수학이라는 과목을 얼마만큼 이해하고 있는지를 측정해주고, 이런 이해도를 활용하여 우리가 아직 풀지 않은 미래의 문제에 대해서 우리가 맞을지 틀릴지 예측이 가능하다. 이런 DKT를 활용하면 우리는 학생 개개인에게 수학의 이해도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능하다.

## ❗ 주제 설명
- 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는 일보다는 각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 최종 문제를 맞출지 틀릴지 예측한다.

## 👋 팀원 소개
|[강신구](https://github.com/Kang-singu)|[김백준](https://github.com/middle-100)|[김혜지](https://github.com/h-y-e-j-i)|[이상연](https://github.com/qwedsazxc456)|[전인혁](https://github.com/inhyeokJeon)|
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
|  [![Avatar](https://user-images.githubusercontent.com/58590260/163955612-1e3c1752-9c68-4cb1-af8f-c99b99625750.jpg)](https://github.com/Kang-singu) |  [![Avatar](https://user-images.githubusercontent.com/58590260/163910764-69f88ef5-5d66-4cec-ab17-a53b12463c7d.jpg)](https://github.com/middle-100) | [![Avatar](https://user-images.githubusercontent.com/58590260/163910721-c067c68a-9612-4e70-a464-a4bb84eea97e.jpg)](https://github.com/h-y-e-j-i) | [![Avatar](https://user-images.githubusercontent.com/58590260/163955925-f5609908-6984-412f-8df6-ae490517ddf4.jpg)](https://github.com/qwedsazxc456) | [![Avatar](https://user-images.githubusercontent.com/58590260/163956020-891ce159-3233-469d-a83c-4c0926ec438a.jpg)](https://github.com/inhyeokJeon) |


## 🔨 Tools
```python
Python 3.8.5
PyTorch torch 1.10.2
Scikit-Learn 1.0.2
Wandb 0.12.15
```

## 🏢 Structure
```bash
├── EDA
│   ├── hyeji_EDA.ipynb
│   └── inhyeok_EDA.ipynb
├── Ensemble
│   └── ensemble.ipynb
├── NMF
│   ├── NMF.ipynb
│   └── readme.md
├── README.md
└── SaintPlus
    ├── README.md
    ├── args.py
    ├── data_generator.py
    ├── elapsed.png
    ├── model.py
    ├── pre_process.py
    ├── structure.png
    ├── submission.py
    ├── sweep.yaml
    ├── train.py
    └── utils.py
```


## 🔎 EDA
**userID : 사용자 고유 번호**
 - train : `6698`명의 고유 사용자
 - test : `744`명의 고유 사용자

**assessmentItemID : 문항의 고유 번호**

 - `9454`개의 고유 문항

**testID : 시험지 고유 번호**

 - `1537`개의 고유한 시험지

**answerCode : 정답 여부**

 - 틀린 경우 `0`, 맞는 경우 `1`

**Timestamp : 문제 풀기 시작한 시간**

 - train : `2019-12-31 15:08:01`~`2020-12-29 16:46:21`
 - test : `2019-12-31 23:43:18`~ `2020-12-29 16:44:10`

**KnowledgeTag : 문제의 중분류 태그**

 - `912`개의 태그
 
 ### ❗ EDA 결과
- **사용자 별 정답률**\
  ![image](https://user-images.githubusercontent.com/58590260/175435536-6c690007-057b-48e9-958a-70d90725773f.png)
 - 사용자 별 정답률 평균은 `0.628`
 - 가장 낮은 정답률은 `0.000000`
 - 가장 높은 정답률은 `1.000000`

- **문항 별 정답률**\
![image](https://user-images.githubusercontent.com/58590260/175435565-32385c8b-f035-4703-bcae-45ef8f105708.png)
    - 문향별  정답률 평균은 `0.6542`
    - 가장 낮은 정답률은 `0.04943`
    - 가장 높은 정답률은 `0.99631`
 
- **시험지 별 정답률**\
![image](https://user-images.githubusercontent.com/58590260/175435587-9b5d65e2-6780-4002-90f4-d410fac13ec2.png)
    - 시험지 별 평균은   `0.667982`
    - 가장 낮은 정답률은 `0.327186`
    - 가장 높은 정답률은 `0.955474`
        

- **태그 별 정답률**\
![image](https://user-images.githubusercontent.com/58590260/175435610-16af0791-49e4-40e4-b109-e90dc1deecd6.png)
    - 태그 별 정답률 평균은 `0.615524`
    - 가장 낮은 정답률은 `0.188940`
    - 가장 높은 정답률은 `0.977778`

- **정답률과 문제를 푼 개수 사이 인과관계 : `0.168`**\
![image](https://user-images.githubusercontent.com/58590260/175435713-af99fd04-904d-46b9-a210-256a6a34da55.png)

    - 평균보다 문항을 많이 푼 학생들이 낮은 학생들 보다 높은 정답률을 보이는 경향이 있다.
    
- **태그를 풀었던 사용자의 수와 정답률 사이 상관관계 : `0.376`**\
![image](https://user-images.githubusercontent.com/58590260/175435735-eedecdc8-c7a6-44cf-a4ed-ff263b91e19b.png)

    - 평균보다 많이 노출된 태그가 높은 정답률을 보이는 경향이 있다.


## 🏢 Models
![Untitled](https://user-images.githubusercontent.com/58590260/168202680-43b12e86-a2bb-4051-a3a5-25c53472a6ab.png)
- [Saint +](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/main/SaintPlus)
- [NMF](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/main/NMF)
- [CatBoost](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/CatBoost/dkt)
- [GRU](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/GRU/dkt/dkt)
- [Gradient Boosting](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/GradientBoosting/dkt)
- [HistGradient Boosting](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/HistGradientBoosting/dkt)
- [LGBM](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/LGBM/dkt)
- [RandomForest](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/RandomForest/dkt)
- [XGBoost](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/XGBoost/dkt)
- [cl4kt](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/cl4kt)
- [Ensemble](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/tree/Ensemble/dkt/ensemble)
### 1️⃣ Model

- 마지막 문제의 정답여부를 맞추는 것이기에 **앞의 문제 풀이 이력들이 영향을 끼칠 것으로 예측**되어 sequential 문제를 풀기위한 모델인 BERT 와 LSTM을 적용
- **데이터의 수가 적어** 데이터의 수가 많이 필요한 딥러닝 모델보다는 딥러닝이 아닌 **머신러닝의 모델**들이 더 좋을 것이라 예측되어 LGBM , Catboost, XGBoost, HistGradeintBoosting 적용
- **단순 행렬 분해를 통해 특성을 구하는 것도 좋은 결과가 나올 것이라 예측**되어 SVD, NMF 사용
- **유저의 수준**과 **문제 난이도**를 고려하는 Feature를 추가.
- **DKT를 위한 모델**인 SAINT+ (riiid) , CL4KT(upstage) ****를 적용
### 2️⃣ Ensemble
- **서로 다른 방식의 모델들** 위주로 앙상블하였음. ( SAINT+NMF , SAINT+NMF+Boost, ...)
### ✨ Final Model : SAINT+ 와 NMF 앙상블

<p align="center"><img src="https://user-images.githubusercontent.com/58590260/175435804-c5c3f381-b87e-4233-8d0d-361ebc7c59bc.png" width=1000></p>

- **SAINT+ 와 NMF 의 결과 평균을 사용.**
- NMF 분석으로 생성된 행렬 변환 행렬을 통해 정답을 추론하여 결과 값 생성하고, Saint+ encoder / decoder에 attention이 사용.
- SAINT+ 는 **정답 여부에 문제 푼 시간이 중요한데 시간을 임베딩하여 사용**하였기에 좋은 결과가 나온 것으로 예측.
- **정답 여부를 0 / 1 로 나타내고 특성들도 음수가 없을 것이기**에 SVD보다 NMF의 결과가 더 좋은 것으로 예측.

## 🏆 최종 결과
|Model|Final Rank|Final score|
|:---:|:---:|:---:|
|Saint Plus + NMF|6|0.8494|


## 📒 보고서
[Report Notion Link](https://thundering-astronomy-d23.notion.site/RecSys-06-Level-2-P-Stage-DKT-f4ce25796e6a454b9bb916d94161347b)


## 📜 참고자료
- [SAINT+: Integrating Temporal Features for EdNet Correctness Prediction](https://arxiv.org/abs/2010.12042)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Contrastive Learning for Knowledge Tracing](https://dl.acm.org/doi/pdf/10.1145/3485447.3512105)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- [UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation](https://arxiv.org/abs/2110.15114)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/pdf/1706.09516.pdf)
