# Contrastive Learning for Knowledge Tracing
This is our implementation for the paper [Contrastive Learning for Knowledge Tracing](https://dl.acm.org/doi/abs/10.1145/3485447.3512105) (TheWebConf 2022).

To run CL4KT, please prepare the configuration file (`configs/example.yaml`) and the raw dataset (e.g., `datatset/algebra05/data.txt`, `datatset/assistments09/data.csv`, etc.).

For example, the `algebra05` dataset comes from the [KDD Cup 2010 EDM Challenge](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp). Datasets need to be downloaded and put inside each corresponding data folder in `dataset`.

Please use the following script to run data preprocessing:

```
python preprocess_data.py --data_name algebra05 --min_user_inter_num 5
```

Please use the following script to run the CL4KT model:

```
CUDA_VISIBLE_DEVICES=0 python main.py --model_name cl4kt --data_name iscream --mask_prob 0.5 --crop_prob 0.3 --permute_prob 0.5 --replace_prob 0.5 --reg_cl 0.1
```


'''
1. data_preprocessing.ipynb 파일 먼저 실행

2. configs/example.yaml 파일을 통해서 하이퍼 파라미터 튜닝 가능

3. 아래의 커맨드 입력. (세부 경로 설정은 본인이 조절해야 할 듯)
저는 workspace에 cl4kt 폴더를 만들어서 돌렸습니다.
CUDA_VISIBLE_DEVICES=0 python main.py
CUDA_VISIBLE_DEVICES=0 nohup python main.py &

4. submission 폴더에 돌린 파일의 example.yaml, 성능, submission 파일 저장됨
'''