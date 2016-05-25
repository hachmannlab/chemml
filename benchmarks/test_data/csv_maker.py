import pandas as pd

file = 'sample_data_0.csv'
df = pd.DataFrame()
df['a'] = [0]*5
df['b'] = [1]*5
df['c'] = [-1]*5
df['d'] = range(5)

df.to_csv(file,index=False)
