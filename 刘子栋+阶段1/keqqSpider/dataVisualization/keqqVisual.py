# -*- coding: gbk -*-
import pandas
import matplotlib.pyplot as plt
import numpy as np
alg = pandas.read_csv('../keqqInfo.csv')

for i,v in enumerate(alg['price']):
    if v == '免费' or v == None:
        alg['price'][i] = 0.00
price = list(alg['price'])
course_price = [float(i) for i in price]
course_status = list(alg['course_status'])

figure = plt.figure()
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False


#价格与人数散点分布图
ax1 = figure.add_subplot(221,title='课程价格与报名人数散点分布图')
ax1.scatter(alg['price'],alg['course_status'])
plt.xlabel('价格')
plt.ylabel('报名人数')
plt.show()

#培训机构及课程数量图
group_course_organization = alg.groupby('course_organization')
sum = group_course_organization.count()
sum = sum ['course_name']
# ax2 = figure.add_subplot(222,title='培训机构数量图')
sum.plot()
plt.xlabel('培训机构')
plt.ylabel('数量')
plt.show()

# 免费课程和付费课程占比饼状图
pay_course = 0
free_course = 0
for i in alg['price']:
    if i == '免费':
        free_course += 1
    else:
        pay_course += 1
pay_course_perc = pay_course/(pay_course+free_course)
free_course_perc = free_course/(pay_course+free_course)
labels = ["付费","免费"]
color = ["yellow","green"]
plt.pie([pay_course_perc,free_course_perc],labels=labels,colors=color,explode=(0,0.05),autopct="%0.2f%%")
plt.show()
