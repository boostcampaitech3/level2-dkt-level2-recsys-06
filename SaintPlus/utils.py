import random
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_time_lag(df):
    """
    Compute time_lag feature, same task_container_id shared same timestamp for each user
    """
    time_dict = {}
    time_lag = np.zeros(len(df), dtype=np.float32)
    for idx, row in enumerate(df[["userID", "Timestamp", "testId"]].values):
        if row[0] not in time_dict:
            time_lag[idx] = 0
            time_dict[row[0]] = [row[1], row[2], 0] # last_timestamp, last_task_container_id, last_lagtime
        else:
            if row[2] == time_dict[row[0]][1]:
                time_lag[idx] = time_dict[row[0]][2]
            else:
                time_lag[idx] = row[1] - time_dict[row[0]][0]
                time_dict[row[0]][0] = row[1]
                time_dict[row[0]][1] = row[2]
                time_dict[row[0]][2] = time_lag[idx]

    df["time_lag"] = time_lag/60 # convert to miniute
    df["time_lag"] = df["time_lag"].clip(0, 1440) # clip to 1440 miniute which is one day => 문제푼지 하루가 지났다면 1400(60*24)로 만들어주고, 아니라면 그대로, 0보다 작다면 0으로 만들어줌
    return time_dict