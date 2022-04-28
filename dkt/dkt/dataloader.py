import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
import datetime
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df):
        # TODO
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct = row[0], row[1], row[2], row[3]

        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader


def feature_engineering():

    data_dir = '/opt/ml/input/data' # 경로는 상황에 맞춰서 수정해주세요!
    csv_file_path = os.path.join(data_dir, 'train_data.csv') # 데이터는 대회홈페이지에서 받아주세요 :)

    df = pd.read_csv(csv_file_path) 
    df = feature_engineering(df)
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    
    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    
    # df.insert(3, "solveTime", np.NaN)
    # train_np = df.to_numpy()

    # idx_list = list()

    # for i in range(len(train_np)-1):
    #     # 현재 문제의 timestamp을 가져온다
    #     current_date_data, current_time_data = train_np[i][5].strip().split(" ")
    #     current_year, current_month, current_day = map(int, list(current_date_data.split("-")))
    #     current_hour, current_minute, current_second = map(int, list(current_time_data.split(":")))

    #     # 다음 문제의 timestamp를 가져온다.
    #     next_date_data, next_time_data = train_np[i+1][5].strip().split(" ")
    #     next_year, next_month, next_day = map(int, list(next_date_data.split("-")))
    #     next_hour, next_minute, next_second = map(int, list(next_time_data.split(":")))


    #     # 같은 유저가 다음 문제도 같은 시험지를 풀고 았거나, 
    #     # 다른 시험 문제지를 같은 날짜에 풀었을 경우
    #     # 문제 푸는 시간 = 다음 문제가 시작 시간 - 현재 문제가 시작한 시간
    #     if train_np[i][0]==train_np[i+1][0] and \
    #         (train_np[i][2]==train_np[i+1][2] or (train_np[i][5]!=train_np[i+1][5] and current_date_data == next_date_data)):         
    #         train_np[i][3] = datetime.datetime(next_year, next_month, next_day, next_hour, next_minute, next_second) - datetime.datetime(current_year, current_month, current_day, current_hour, current_minute, current_second)
    #         train_np[i][3] = train_np[i][3].total_seconds() # 초로 변환
    #         #if train_np[i][3]>150 : train_np[i][3] = 150

    #     else :
    #         # 마지막으로 푼 문제인 경우는 60으로 통일
    #         train_np[i][3] = 60.0

    #     df.iloc[i,3] = train_np[i][3]

    #     #if train_np[i][3] == 0  : idx_list.append(i)
   
    # train_np[-1][3] = 60.0
    # df.iloc[-1,3] = train_np[i][3]

    
    # test_mean_solveTime = df.groupby('testId')['solveTime'].mean()
    # test_mean_solveTime.columns = ['test_mean_solveTime']
    # # user_mean_solveTime = df.groupby('userID')['solveTime'].mean()
    # # user_mean_solveTime.columns = ['user_mean_solveTime']
    # tag_mean_solveTime = df.groupby('KnowledgeTag')['solveTime'].mean()
    # tag_mean_solveTime.columns = ['tag_mean_solveTime']
    # #assessmentItem_mean_solveTime = df.groupby('assessmentItemID')['solveTime'].mean()
    # #assessmentItem_mean_solveTime.columns = ['assessmentItem_mean_solveTime']

    # df = pd.merge(df, test_mean_solveTime, on=['testId'], how="left")
    # #df = pd.merge(df, user_mean_solveTime, on=['userID'], how="left")
    # df = pd.merge(df, tag_mean_solveTime, on=['KnowledgeTag'], how="left")
    # #df = pd.merge(df, assessmentItem_mean_solveTime, on=['assessmentItemID'], how="left")

    df.insert(1, "testType", np.NaN)
    df.insert(2, "testID", np.NaN)
    df.insert(3, "questionID", np.NaN)

    train_np = df.to_numpy()

    for i in range(len(df)):
        assessmentItemID = train_np[i][4]
        df.iloc[i,1] = int(assessmentItemID[2])
        df.iloc[i,2] = int(assessmentItemID[4:7])
        df.iloc[i,3] = int(assessmentItemID[8:])

    #df = pd.DataFrame(train_np, columns=df.columns)

    
    # df = df.drop(idx_list, axis=0)
    # df = df.drop(['solveTime'], axis=1)

    return df