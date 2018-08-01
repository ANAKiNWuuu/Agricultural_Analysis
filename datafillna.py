import pandas as pd

df = pd.read_csv('weatherday二崙鄉產銷履歷.csv', header=0, index_col=0)
df.fillna(0, inplace=True)
print(df.head(5))
df.to_csv('weatherday二崙鄉產銷履歷.csv')