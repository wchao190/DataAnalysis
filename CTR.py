import numpy as np
import pandas as pd
from openpyxl.workbook import Workbook
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\wuc\Desktop\ABtest_actions.csv")
# 总数据量
size = len(data)
# 参与用户数
user_num = data.drop_duplicates(subset=["id"])

# 测试时长
days = pd.to_datetime(data["timestamp"].tail(1).values) - pd.to_datetime(data["timestamp"].head(1).values)

# 试验组A
experiment_group = data[ data["group"]=='experiment']
experiment_click = experiment_group.query("action=='click'")["id"].nunique()
experiment_view =experiment_group.query("action=='view'")["id"].nunique()
experiment_crt = round((experiment_click / experiment_view),4)

# 控制组B
control_click = data[(data["group"]=='control')&(data["action"]=='click')]["id"].nunique()
control_view = data[(data["group"]=='control')&(data["action"]=='view')]["id"].nunique()
control_crt = round((control_click/control_view),4)


# 方案A、B的差异
diff_crt = round(experiment_crt - control_crt,4)
print(diff_crt)

# p_calue <0.05
#　对样本抽样调查
diffs = []
for n in range(1000):
    sample = data.sample(size,replace=True)
    # 试验组A
    experiment_group = sample[sample["group"] == 'experiment']
    experiment_click = experiment_group.query("action=='click'")["id"].nunique()
    experiment_view = experiment_group.query("action=='view'")["id"].nunique()
    experiment_crt = round((experiment_click / experiment_view), 4)

    # 控制组B
    control_click = sample[(sample["group"] == 'control') & (sample["action"] == 'click')]["id"].nunique()
    control_view = sample[(sample["group"] == 'control') & (sample["action"] == 'view')]["id"].nunique()
    control_crt = round((control_click / control_view), 4)
    # 方案A、B的差异
    diff_crt = round(experiment_crt - control_crt, 4)
    diffs.append(diff_crt)
diffs = np.array(diffs)

# 正态分布
normal = np.random.normal(0,diffs.std(),size)

p_value = (normal > diff_crt).mean()
print(p_value)
