import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
year = np.arange(2009,2020)
salse = np.array([0.52,9.36,33.6,132,352,571,912,1207,1682,2135,2684])
df = pd.DataFrame([year,salse]).T
# plt.scatter(x=year,y=salse)
# plt.show()
# 初步判断：多项式回归
"""
y= -0.20964258*x**3 + 34.42433566*x**2 + -117.85390054*x +90.1206060606047
"""

# 数据预处理：准备x、y对应的值，方便后续建模并计算系数、截距；
model_y = salse
model_x = (year-2008).reshape(len(year-2008),1)
model_x = np.concatenate([model_x**3,model_x**2,model_x],axis=1)

#创建回归模型
model = LinearRegression()
model.fit(model_x,model_y)
#将值带入方程式
# print("系数：",model.coef_)
# print("截距：",model.intercept_)

#添加趋势线
trend_x = np.linspace(1,12,100)
func = lambda x : -0.20964258*x**3 + 34.42433566*x**2 + -117.85390054*x +90.1206060606047
trend_y = func(trend_x).round(2)
axes = plt.subplot(111)
"""方法二"""
# axes.plot(trend_x,trend_y,color='r')
# axes.scatter(x=year-2008,y=salse,color='g')
# axes.scatter(12,func(12),color='y')
# plt.xticks([2,4,6,8,10,12])
# x = list(range(1,13))
# y = np.append(salse,func(12)).round(2)
# for x,y in zip(x,y):
#     plt.text(x-0.3,y+30,"%s"%y,ha='center',va='bottom',fontsize=10)
# axes.set_xticklabels(["2010","2012","2014","2016","2018","2020"])
# plt.show()

"""方法一"""
salse = np.append(salse,func(12)).round(2)
yr = pd.Series(range(1,13))
df2 = pd.DataFrame([yr,salse]).T
df3 = pd.DataFrame([trend_x,trend_y]).T
ax = df3.plot(x=0,y=1)
df2.plot.scatter(x=0,y=1,ax=ax)
plt.show()