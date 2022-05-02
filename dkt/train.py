import os

import torch
import wandb
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess, custom_train_test_split, feature_engineering, train_test
from dkt.utils import setSeeds
import pandas as pd
import datetime


def main(args):
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if args.model == 'LGBM':
        data_dir = '/opt/ml/input/data' # 경로는 상황에 맞춰서 수정해주세요!
        csv_file_path = os.path.join(data_dir, 'train_data.csv') # 데이터는 대회홈페이지에서 받아주세요 :)
        df = pd.read_csv(csv_file_path) 
        
        df = feature_engineering(df)
        train, test = custom_train_test_split(df)
        train, y_train, test, y_test = train_test(train, test)
        trainer.run(args, [train, y_train], [test, y_test])

    else:
        preprocess = Preprocess(args)
        preprocess.load_train_data(args.file_name)
        train_data = preprocess.get_train_data()
        train_data, valid_data = preprocess.split_data(train_data)
        if args.wandb:
            wandb.login()
            wandb.init(project="dkt", config=vars(args))
        trainer.run(args, train_data, valid_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)