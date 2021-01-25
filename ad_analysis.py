import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score
from openpyxl.workbook import Workbook
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一(替换sans-serif字体)
plt.rcParams['axes.unicode_minus'] = False # 步骤二(解决坐标轴负数的负号显示问题)

#显示全部行
pd.set_option("display.max_rows",None)
#显示全部列
pd.set_option("display.max_columns",None)

col = ["渠道代号","日均UV","平均注册率","平均搜索量","访问深度","平均停留时间","订单转化率","投放总时间"]
path  = r"C:\Users\wuc\Desktop\ad_performance.csv"
dt = pd.read_csv(path,index_col=0)
dt.fillna({"平均停留时间":round(dt["平均停留时间"].mean(),4)},inplace=True)
# print(dt.describe().round(4))

# 计算、合并：相关性
result = dt.corr().round(4) #计算相关性
dt = dt.drop(["平均停留时间"],axis=1) #删除相关性高的其中一列
# print(dt)
# pd.DataFrame(result).to_excel(r"C:\Users\wuc\Desktop\ad.xlsx")

# 对数据标准化：使用minmax_sacler方法
scaler_dt = dt.iloc[:,1:7]
model = MinMaxScaler(feature_range=(0,1))
scaler_result = model.fit_transform(scaler_dt)
# print(scaler_result)

# 特征化数字：独热编码One-Hot
oneHot_model = OneHotEncoder()
onehot_result = oneHot_model.fit_transform(dt[["素材类型","广告类型","合作方式","广告尺寸","广告卖点"]]).toarray()
# print(onehot_result)

# 合并标准化数据、独热编码
new_data = np.hstack((scaler_result,onehot_result))
# print(new_data)
# KMeans建模，基于平均轮廓系数，找到最佳K值；
max_val = -1 # 设置一个轮廓系数初始值
best_k = 2   # 设置一个聚类中心数量初始值
for k in range(2,6):
    kmeans_model = KMeans(n_clusters=k)  #初始化聚类函数
    temp = kmeans_model.fit_predict(new_data)  #对数据聚类，并返回每个数据样本所属聚类的索引
    val = silhouette_score(new_data,temp) # 计算所有样本的轮廓系数
    # print(val)
    if val > max_val: # 如果轮廓系数 > 初始值，
        max_val = val   # 替换原来的系数值
        best_k = k    # 同时替换原来的聚类中心数量
        label = temp
# print(best_k,max_val)
# print(label)

# 聚类合并结果
# 1.合并数据与聚类标签
cluster_label = pd.DataFrame(label,columns=["cluster"])
# dt["cluster"] = label
merge_data = pd.concat([dt,cluster_label],axis=1)
# print(dt.head())

# 2.各聚类下的样本量
count = merge_data["渠道代号"].groupby(merge_data["cluster"]).count()
cluster_count = pd.DataFrame(count).T.rename(index={"渠道代号":"counts"})
# 3.各聚类下的样本占比
cluster_percent = (cluster_count/count.sum()).round(4).rename(index={"counts":"percent"})

label_describe = []
for label in range(best_k):
    # 4.数值类特征的均值，前6列
    label_data = merge_data[merge_data["cluster"] == label] #  获取每个簇的数据
    p1_dt = label_data.iloc[:,1:7].describe() # 对前6列数值数据做聚合运算
    p1_mean = p1_dt.loc["mean",:] # 取均值
    # 5.字符类特征的众数
    p2_dt = label_data.iloc[:,7:-1].describe() # 对分类数据做聚合运算
    p2_top = p2_dt.loc["top"] # 取众数
    merge_describe = pd.concat([p1_mean,p2_top])
    label_describe.append(merge_describe) # 合并数值、分类聚合数据

# 6.数据合并与展示
label_pd = pd.DataFrame(label_describe).T
all_label_result = pd.concat([cluster_count,cluster_percent,label_pd],axis=0)

#绘制雷达图：对数值特征对比分析
#获取个簇/集群的数值特征均值，并且标准化
matplotlib_dt = label_pd.iloc[0:6,:].T
matplotlib_model = MinMaxScaler(feature_range=(0,1)) # 初始化标准化
matplotlib_result = matplotlib_model.fit_transform(matplotlib_dt).round(4) # 标准化数据，返回一个二维数组
#绘制画布，准备数据，x轴角度、y轴数据、类别对应的数据
labels = matplotlib_dt.columns.values
first_dt = matplotlib_result[:,0]  # 获取数据的首列值
rader_dt = np.concatenate((matplotlib_result, first_dt.reshape(4, 1)), axis=1)  # 合并数据，形成闭环数据
#分割圆周长
n = len(labels)
angle = np.linspace(0,2*np.pi,n,endpoint=False) #分割圆周长
angle = np.concatenate((angle,[angle[0]])) #将雷达图一圈封闭起来
#绘制各簇对应的点线图
fig = plt.figure()  #创建画布
axes = fig.add_subplot(111,polar=True)  #绘制雷达图
for i in range(len(rader_dt)): #根据指标数量，循环画图
    axes.plot(angle,rader_dt[i],label=i,marker=".") #设置各个区的轮廓，x是角度，y是半径
axes.set_thetagrids(angle[:6]*180/np.pi,labels)  #设置类别名称，剔除最后一个，因为与第一个重复
axes.set_rlim(-0.2,1.2)  #设置半径刻度范围；
plt.legend(loc="lower right")  #设置图例标签
plt.show()
