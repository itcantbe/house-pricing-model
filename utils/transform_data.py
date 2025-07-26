import numpy as np
import pandas as pd
from utils.config import FRAC_VAL,STATE_VAL


def split_data(df):
    random_input  = df.sample(frac=FRAC_VAL, random_state=STATE_VAL)
    random_test = df[~df.index.isin(random_input.index)]
    random_input_target = random_input[["median_house_value"]].copy()
    random_input_features = random_input.drop(columns=["median_house_value"]).copy()
    random_test_target = random_test[["median_house_value"]].copy()
    random_test_features = random_test.drop(columns=["median_house_value"]).copy()
    return random_input_features,random_input_target,random_test_features,random_test_target

def skew_in_data(df):
    return df.skew()

def one_hot_transform(df):
    return pd.get_dummies(df,dtype=int)

def log_transformation(data,*args):
    for i in args:
        data[i] = np.log(data[i])

def zScoreRegularization(data):
    return (data - data.mean()) / data.std()
