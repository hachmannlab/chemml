import pandas as pd
from tabulate import tabulate

df = pd.read_excel('contents.xlsx',sheetname=0,header=0) 
print tabulate(df, headers='keys', tablefmt='psql')
