# -*- coding: gbk -*-
import pandas
import matplotlib.pyplot as plt
import numpy as np
alg = pandas.read_csv('../keqqInfo.csv')

for i,v in enumerate(alg['price']):
    if v == '���' or v == None:
        alg['price'][i] = 0.00
price = list(alg['price'])
course_price = [float(i) for i in price]
course_status = list(alg['course_status'])

figure = plt.figure()
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False


#�۸�������ɢ��ֲ�ͼ
ax1 = figure.add_subplot(221,title='�γ̼۸��뱨������ɢ��ֲ�ͼ')
ax1.scatter(alg['price'],alg['course_status'])
plt.xlabel('�۸�')
plt.ylabel('��������')
plt.show()

#��ѵ�������γ�����ͼ
group_course_organization = alg.groupby('course_organization')
sum = group_course_organization.count()
sum = sum ['course_name']
# ax2 = figure.add_subplot(222,title='��ѵ��������ͼ')
sum.plot()
plt.xlabel('��ѵ����')
plt.ylabel('����')
plt.show()

# ��ѿγ̺͸��ѿγ�ռ�ȱ�״ͼ
pay_course = 0
free_course = 0
for i in alg['price']:
    if i == '���':
        free_course += 1
    else:
        pay_course += 1
pay_course_perc = pay_course/(pay_course+free_course)
free_course_perc = free_course/(pay_course+free_course)
labels = ["����","���"]
color = ["yellow","green"]
plt.pie([pay_course_perc,free_course_perc],labels=labels,colors=color,explode=(0,0.05),autopct="%0.2f%%")
plt.show()
