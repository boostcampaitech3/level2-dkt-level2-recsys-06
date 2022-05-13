 

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
