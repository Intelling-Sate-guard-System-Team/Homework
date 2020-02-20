import csv
import pandas as pd
import matplotlib.pyplot as plt

#
# filename = 'fee_course.csv'
# fd = pd.read_csv(filename, usecols=['course_name', 'course_organization', 'course_price'])
# #fd.sort_index()
# fd.sort_values(by=['course_price'], ascending=False)
# print(fd.head(10))


figure = plt.figure(num='hist')

ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_xlim(0,12)

fields = ['course_name', 'course_organization', 'course_price']

data = []
course_name = set()
with open('fee_course.csv', 'r') as fd:

    reader = csv.DictReader(fd, fieldnames=fields)
    next(reader)

    for row in reader:
        data.append(row['course_price'])
        course_name.add(row['course_name'])

r = ax.hist(
    x=data,
    bins=len(course_name),  # 箱子个数
    density=False,
    label='直方图',
    range=(0, 12),
    cumulative=False,
    bottom=[-1],
    histtype='bar',
    align='left',
    orientation='vertical',
    rwidth=0.5,
    color='r',
    stacked=True,
)
print(r)
ax.legend()

figure.show(warn=False)
plt.show()

