import pytest
import pandas as pd
import numpy as np
from chemml.autoML import ModelScreener

x1=[]
y1=[]
for i in range(0, 10):
    x1.append(i)
    y1.append(i*2)

df = pd.DataFrame(list(zip(x1,y1)), columns=["x","y"])

def test_screener_types():
    for i in ["regressor", "classifier"]:
        scores = ModelScreener(df=df, target="y", screener_type=i).screen_models()
        #if scores is not empty, everything is okay
        assert scores.empty != True

def test_target_format_types():
    for i in ["y", 1]:
        scores = ModelScreener(df=df, target=i, screener_type="regressor").screen_models()
        #if scores is not empty, everything is okay
        assert scores.empty != True
