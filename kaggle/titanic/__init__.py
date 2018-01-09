
import pandas as pd

if __name__ == '__main__':
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    combine = [train_df,test_df]

    print(train_df.columns.values)