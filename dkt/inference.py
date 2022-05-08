import os
from regex import E

import torch
import pandas as pd
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess, feature_engineering, custom_train_test_split, train_test


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if args.model == "RandomForest":
        # LOAD TESTDATA
        data_dir = '/opt/ml/input/data'
        test_csv_file_path = os.path.join(data_dir, 'test_data.csv')
        test_df = pd.read_csv(test_csv_file_path)
        
        df = feature_engineering(test_df)
        trainer.inference(args, df)
    else:
        preprocess = Preprocess(args)
        preprocess.load_test_data(args.test_file_name)
        test_data = preprocess.get_test_data()

        trainer.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)