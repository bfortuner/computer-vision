import numpy as np
import pandas as pd


def get_key_int_maps(keys):
    key_to_int = {name: i for i, name in enumerate(keys)}
    int_to_key = {i: name for i, name in enumerate(keys)}
    return key_to_int, int_to_key

def onehot_encode_class(class_to_idx, classname):
    n_classes = len(class_to_idx.keys())
    onehot = np.zeros(n_classes)
    idx = class_to_idx[classname]
    onehot[idx] = 1
    return onehot   

def encode_column(df, column):
    df = df.copy()
    unique_names = df[column].unique()
    key_to_int, int_to_key = get_key_int_maps(unique_names)
    encoded = [key_to_int[c] for c in df[column].values]
    df[column+'_code'] = encoded
    return df