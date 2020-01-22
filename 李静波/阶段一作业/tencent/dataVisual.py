from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams["font.family"] = 'Arial Unicode MS'

fd = open('Data.csv', 'r')
alg = pd.read_csv(fd, usecols=['course_organization', 'course_link'])
group = alg.groupby('course_organization')
org = group.count()

# 统计开设的课程最多的前十个机构，并用matplotlib柱状图可视化
df = org.sort_values('course_link',ascending = False)
ab = df[:10]
ab.plot(kind='bar')
plt.show()
