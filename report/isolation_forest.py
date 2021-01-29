import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

data = pd.read_csv(r"C:\Users\wuc\Desktop\detection.csv")
# 建立孤立森林模型
model = IsolationForest()
# 训练数据
model.fit(data[["visitNumber"]])
# 计算异常的评分
data["score"]=model.decision_function(data[["visitNumber"]])
# 判断是否异常
data["ifnormal"]=model.predict(data[["visitNumber"]])

print(data[["visitNumber","score","ifnormal"]].query("ifnormal==-1"))

print(len(data[["visitNumber","score","ifnormal"]].query("ifnormal==-1")))
