import pandas as pd

def load_dataset(path, target):
    df = pd.read_csv(path)
    return df.drop(columns=[target]), df[target]
