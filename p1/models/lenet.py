import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5) # channels表示输入图像的通道数（灰度图为1，RGB图为3） 同时因为我们用六个卷积核，所以输出的通道数是6，kernel_size表示卷积核的大小是5*5
        self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2) # 池化层，kernel_size表示池化窗口的大小，stride表示步长(窗口滑动的步长)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)
        self.pool2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size = 5)
        self.fc1 = nn.Linear(120, 84) # 全连接层，输入特征数为120，输出特征数为84
        self.fc2 = nn.Linear(84, num_classes) # 全连接层，输入特征数为84，输出特征数为num_classes

    def forward(self, x):
        x = F.tanh(self.conv1(x)) # 使用tanh激活函数,对卷积结果做 tanh 激活，使数据范围变成 [-1, 1]，增加非线性表达能力。
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        x = x.view(x.size(0), -1) # 展平多维输入数据为二维，-1表示自适应计算该维度大小
        x = F.tanh(self.fc1(x))
        x = self.fc2(x) # 最后一层不使用激活函数，直接输出原始分数（logits）
        return x

# LeNet-5 是 Yann LeCun 等人提出的经典卷积神经网络，主要用于手写数字识别。其结构如下：

#     输入层：单通道（灰度）图像 32x32
#     C1：卷积层，6 个 5x5 卷积核，输出 6x28x28
#     S2：池化层（亚采样），6x14x14
#     C3：卷积层，16 个 5x5 卷积核，输出 16x10x10
#     S4：池化层，16x5x5
#     C5：卷积层，120 个 5x5 卷积核，输出 120x1x1（全连接）
#     F6：全连接层，输出 84
#     输出层：全连接层，输出 10（数字类别）
