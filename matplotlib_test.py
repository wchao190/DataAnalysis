import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import test11
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一(替换sans-serif字体)
plt.rcParams['axes.unicode_minus'] = False # 步骤二(解决坐标轴负数的负号显示问题)
# 电影年产量
# data = test11.movie_count_by_year()
# data["year"] = data.index
# data["year"] = data["year"].dt.year
# data.set_index(["year"],drop=True,inplace=True)
# dt = data["release_date"]
# ax1 = plt.subplot(111)
# dt.plot.bar(xticks=[5,25,45,65,85,105],ax=ax1,rot=0)
# ax1.set_xticklabels([1920,1940,1960,1980,2000,2020])
# plt.show()

#每年各国电影产量 -- 折线图
# ax2=plt.subplot(111)
# data2 = test11.count_by_year_country()
# data2[["中国","美国","日本","韩国","英国","德国"]].plot(xlim=("1970","2020"),ax=ax2)
# plt.show()

# 各国电影总产量 -- 直方图
# data3 = test11.count_by_year_country()
# data3 = data3.T
# data3 = data3.sum(axis="columns")
# data3 = data3.drop("release_date")
# data3 = data3.sort_values(ascending=False)
# # data3[:10].plot.bar()
# # plt.show()

# 评分与人数 - 散点图
# data4 = test11.average_vote()
# data4.plot.scatter(x='average',y='votes')
# plt.show()

# 电影类型热力图

# data5 = test11.gener_average()
# data5["avg"] = round(data5["total"].div(data5["count"]),2)
# result = data5.loc[:,"total"]
# print(result)
# sns.heatmap(result,xticklabels=data5.index,yticklabels=data5["avg"])
# plt.show()

# 散点图
# data6 = test11.gener_average()
# # data6["avg"] = round(data6["number"].div(data6["total"]),2)
# data6.index.names=["gener","score"]
# data6 = data6["total"]
# dt = data6.unstack(0)
# sns.heatmap(dt,annot=True,cmap="YlGn",fmt="000000",vmin=100,vmax=2000)
# plt.show()

# 每年电影评分 箱型图
# dt = pd.DataFrame([["123","456",None,"789"],["one","two","three",None],["a","c",None,"e"]])
# print(dt)


from sklearn.cluster import KMeans
file = r"C:\Users\wuc\Desktop\score.csv"
dt = pd.read_csv(file,encoding="gbk")

kmodels = KMeans(n_clusters=4)
for col in dt.columns:
    kmodels.fit(dt[[col]])
    print(col)
    centers = kmodels.cluster_centers_
    for c in centers:
        print(c[0])
