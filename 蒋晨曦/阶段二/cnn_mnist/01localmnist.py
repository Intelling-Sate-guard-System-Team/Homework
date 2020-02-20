import struct
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

# 加载本地 MNIST数据集
# 拼接数据集路径
def dataPath(filename):
    root = "/Users/macbookpro/PycharmProjects/data/mnist_dataset"
    return os.path.join(root, filename)

# 读取图片
def load_image_fromfile(filename):
    with open(filename, 'br') as fd:
        # 读取图像的信息
        header_buf = fd.read(16)   # 16字节，4个int整数
        # 按照字节解析头信息（具体参考python SL的struct帮助）
        magic_, nums_, width_, height_ = struct.unpack('>iiii', header_buf)  # 解析成四个整数：>表示大端字节序，i表示4字节整数
        # 保存成ndarray对象
        imgs_ = np.fromfile(fd, dtype=np.uint8)
        imgs_ = imgs_.reshape(nums_, height_, width_)
    return imgs_

# 读取标签
def load_label_fromfile(filename):
    with open(filename, 'br') as fd:
        header_buf = fd.read(8)
        magic, nums = struct.unpack('>ii' ,header_buf)
        labels_ = np.fromfile(fd, np.uint8)
    return labels_

# 读取训练集
train_x = load_image_fromfile(dataPath('train-images-idx3-ubyte'))
train_y = load_label_fromfile(dataPath('train-labels-idx1-ubyte'))
train_x = train_x.astype(np.float64)
train_y = train_y.astype(np.int64)

# 读取测试集
test_x = load_image_fromfile(dataPath('t10k-images-idx3-ubyte'))
test_y = load_label_fromfile(dataPath('t10k-labels-idx1-ubyte'))

# # 可视化验证读取的数据
# ax1 = plt.subplot(121, title=F"train_picture, label: {train_y[0]}")
# ax1.imshow(train_x[0], cmap="gray")
# ax2 = plt.subplot(122, title=F"test_picture, label: {test_y[0]}")
# ax2.imshow(test_x[0], cmap="gray")
# plt.show()

# 使用Torch的数据集管理工具管理
# 转换为Tensor
x = torch.tensor(train_x).view(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]) # 样本数，通道数，宽，高
y = torch.LongTensor(train_y)

t_x =  torch.tensor(test_x).view(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2])   # N,C,W,H
t_y =  torch.LongTensor(test_y)

# 使用TensorDataSet封装数据与标签
train_dataset = torch.utils.data.TensorDataset(x, y)
test_dataset = torch.utils.data.TensorDataset(t_x, t_y)

# 数据随机与切分器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=2000)   # 批次数量1000
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=10000)  # 一个批次直接预测



class LeNet_5(torch.nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        # 卷积层1 :1 @ 28 * 28 - > 6 @ 28 * 28 -> 6 @ 14 * 14
        # 卷积层2 :6 @ 14 * 14  -> 16 @ 10 * 10 -> 16 @ 5 * 5
        # 卷积层3 :16 @ 5 * 5 -> 120 @ 1 * 1
        self.layer_1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2)
        self.layer_2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), padding=0)
        self.layer_3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), padding=0)
        # 连接层1 : 120 -> 84
        # 链接层2 : 84 -> 10
        self.layer_4 = torch.nn.Linear(120, 84)
        self.layer_5 = torch.nn.Linear(84, 10)

    def forward(self, input):
        # 预测模型实现
        # 卷积层
        t = self.layer_1(input)
        t = torch.nn.functional.relu(t)
        t = torch.nn.functional.max_pool2d(t, kernel_size=(2, 2))

        t = self.layer_2(t)
        t = torch.nn.functional.relu(t)
        t = torch.nn.functional.max_pool2d(t, kernel_size=(2, 2))

        t = self.layer_3(t)
        t = torch.nn.functional.relu(t)

        t = t.squeeze()  # 长度为1的维数直接降维
        # 链接层
        t = self.layer_4(t)
        t = torch.nn.functional.relu(t)

        t = self.layer_5(t)
        t = torch.nn.functional.log_softmax(t, dim=1)
        return t

# 模型
model = LeNet_5()
parameters = model.parameters()
# 巡视函数
criterion = torch.nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 学习率

# 训练参数
epoch = 500
for e in range(epoch):
    # 批次处理
    for data, target in train_loader:
        # 清空梯度
        optimizer.zero_grad()
        # 计算输出
        out = model(data.float())
        # 计算损失
        loss  = criterion(out, target)
        # 计算梯度
        loss.backward()
        # 更新梯度
        optimizer.step()
    # 一轮结束，可以使用测试集测试准确率
    if e % 100 == 0:
        with torch.no_grad():   # 关闭梯度计算跟踪
            for data, target in test_loader:
                y_ = model(data)
                predict = torch.argmax(y_, dim=1)
                correct_rate = (predict == target).float().mean()
                print(F"\t损失度：{loss:8.6f},\t准确率：{correct_rate * 100: 5.2f}%")
