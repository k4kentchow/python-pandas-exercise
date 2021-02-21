import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas import DataFrame
data = pd.read_csv("data1.csv")
print("1: ")
print(data)

data.fillna(-999,inplace=True)
print("2: ")
print(data)

max_ovd_days_set = data.groupby("cust_id").agg("max")

print(max_ovd_days_set)

max_ovd_days_set["y"]=None
max_ovd_days_set.y[max_ovd_days_set.max_ovd_days>30]=1
max_ovd_days_set.y[max_ovd_days_set.max_ovd_days<=30]=0
print("3: ")
print(max_ovd_days_set)

max_ovd_days_set=DataFrame(max_ovd_days_set)
max_ovd_days_set['intercept'] = 1.0
train_cols = max_ovd_days_set.columns[1:3]
y=np.array(max_ovd_days_set['y'],dtype=np.float64)
x=np.array(max_ovd_days_set[train_cols],dtype=np.float64)
logit = sm.Logit(y,x)
# logit = sm.Logit((max_ovd_days_set['y']), (max_ovd_days_set[train_cols]))
logit.raise_on_perfect_prediction = False
result = logit.fit()