# SaintPlus-Knowledge-Tracing-Transformer
paper : https://arxiv.org/abs/2002.07033

## Introduction
기본 아이디어는 현재 문제와 과거 문제를 인코더에 입력으로 주면 문제 푼 경험의 어떤 부분을 `Attention` 하는지 찾을 수 있다.
다음으로 인코더 값 벡터의 가중치 합을 encoder-decoder의 attention layer에 key와 value 값으로 설정한다.

학생들의 과거이력 칼럼이 디코더 입력으로 사용되고, 디코더의 첫 번째 layer는 응답 간의 관계, 질문 작업에 소요된 시간 및 사용자가 응답한 다른 문제간의 시간 간격을 학습한다. 
첫 번째 디코더의 output sequence는 query로 다음 layer에 전달한다. 

직관적인 설명은 `지식 경험(query)`에 대한 과거 경험을 가지고 있으며, `문제풀이(key, value)`에 대해 `학생들이 어떻게 수행할 것인지(weighted value vector)`로 표현할 수 있다.

## SaintPlus
Saint+는 **Transformer**기반 지식 추적 모델로 학생들의 문제 이력 정보를 바탕으로 미래 정답을 예측.
![image](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/blob/main/SaintPlus/elapsed.png)

## Structure of model
![image](https://github.com/boostcampaitech3/level2-dkt-level2-recsys-06/blob/main/SaintPlus/structure.png)

## Parameters
- `num_layers`: int. 
  - number of multihead attention layer
- `num_heads`: int. 
  - number of head in one multihead attention layer
- `d_model`: int. 
  - dimension of embedding size
- `n_questions`: int 
  - number of different question
- `seq_len`: int 
  - sequence length
- `warmup_steps`: int 
  - warmup_steps for learning rate
- `dropout`: float 
  - dropout ratio
- `epochs`: int 
  - number of epochs
- `patience`: int 
  - patience to wait before early stopping
- `batch_size`: int 
  - batch size
- `optimizer`: str 
  - optimizer
- `lr`: float 
  - learning rate

## How to train
1. Run `python pre_process.py` to fit data for training but you have to concatenate train/validation dataframe from dataset
2. Adjust hyperparameter in `args.py`
3. Setup `wandb` (optional)
4. Run `python train.py`
5. Run `python submission.py`

## Training Result
| |valid auroc|competition auroc|competition accuracy|
|---|---|---|---|
|Basic Dataset|0.8398|0.8260|0.7500|
|with Feature Engineering|0.8510|0.8114|0.7554|

## Reference
https://arxiv.org/abs/2010.12042 - SAINT+: Integrating Temporal Features for EdNet Correctness Prediction   
https://arxiv.org/abs/1706.03762 - Attention Is All You Need

