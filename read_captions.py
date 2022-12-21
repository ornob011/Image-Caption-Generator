import pandas as pd


def read_captions(caption_path):

    data = pd.read_csv(caption_path)
    
    return data
