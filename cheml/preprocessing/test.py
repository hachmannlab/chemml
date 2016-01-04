import pandas as pd
import numpy as np
data = pd.read_csv('cheml/preprocessing/test/test_missing_values.csv',header=None)
target=pd.DataFrame([1,2,3,np.nan,4])
print 'data:\n'
print data
print 'target:\n'
print target

###
from cheml import preprocessing
missval = preprocessing.missing_values(strategy = 'zero',
                                       string_as_null = True,
                                       inf_as_null = True,
                                       missing_values = False)
data = missval.fit(data)
target = missval.fit(target)

data, target = missval.transform(data, target)
###

data=pd.DataFrame({0:[1,np.nan,3],1:[3,5,np.nan]})
mean_values = pd.DataFrame({col:[np.mean(data[col])] for col in data.columns})