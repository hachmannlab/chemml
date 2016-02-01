import pandas as pd
import numpy as np
from data import Trimmer
from data import Uniformer

df = pd.read_csv('test/sample_data.csv')
tf = pd.read_csv('test/sample_target.csv')

trim = Trimmer(type="margins", sort = False, cut = 0.5, shuffle = True)
df, tf = trim.fit_transform(df,tf)

clf = Uniformer(bins=22, bin_pop=2, include_lowest=True, substitute = 'mean')
d,t = clf.fit_transform(df,tf)
