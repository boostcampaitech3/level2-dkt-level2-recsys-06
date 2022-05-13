# SaintPlus-Knowledge-Tracing-Transformer
paper : https://arxiv.org/abs/2002.07033

## Introduction
Thanks to BoostCamp and a lot of amazing data enthusiasm people sharing their info so I had a chance to learn Transformer and really use it to a real-world task!   
    
Saint+ is a **Transformer** based knowledge-tracing model which takes students' exercise history information to predict future performance. As classical Transformer, it has an Encoder-Decoder structure that Encoder applied self-attention to a stream of exercise embeddings; Decoder applied self-attention to responses embeddings and encoder-decoder attention to encoder output.

## SaintPlus
The basic idea is that we fed current question and stream of past exerices into encoder, it will find which parts of exercises experience should be noticed. Then fed the weighted sum of encoder value vector as key and value to encoder-decoder attention layer.    
    
How students performed in past is used as decoder input. The first layer of decoder will learn relationship between responses, how long they took for a question task and time gap between different task user answered. The output sequence from first decoder layer is forward to second layer as query. The intuitive explanation is right now we have past experince of knowledge (query), how will a student perform (weighted value vector) for a sequence of questions (key, value).     

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

