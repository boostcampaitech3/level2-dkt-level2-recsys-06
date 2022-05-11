import numpy as np
import pandas as pd
import os

def make_submission(
    submission_path
):
    df0 = pd.read_csv(os.path.join(submission_path, "submission_0.csv"), sep=',')
    df1 = pd.read_csv(os.path.join(submission_path, "submission_1.csv"), sep=',')
    df2 = pd.read_csv(os.path.join(submission_path, "submission_2.csv"), sep=',')
    df3 = pd.read_csv(os.path.join(submission_path, "submission_3.csv"), sep=',')
    df4 = pd.read_csv(os.path.join(submission_path, "submission_4.csv"), sep=',')

    df = pd.concat([df0['prediction'],df1['prediction'],df2['prediction'],df3['prediction'],df4['prediction']], axis=1)
    df.columns = ['p1','p2','p3','p4','p5']
    df['prediction'] = df.mean(axis=1)

    save_df = df[['prediction']]
    save_df.index.name = 'id'
    save_df.to_csv(os.path.join(submission_path, "submission.csv"), sep=",")