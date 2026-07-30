from torch import nn
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 最基础的NN模型
class NN(nn.Module):
    """
    最基本的线性神经网络（多层感知机）
    输入 时间t、热流H、倾角d、年龄Y、俯冲速度v（深度D、长度L暂时不做输入，当前数据深度为600km，长度为660km）
    输出 温度场，数据数量（分辨率）为 nx × nz
    """

    # 构造函数
    def __init__(self, nx: int = 111, nz: int = 101):
        super().__init__()

        # 较简单的前向神经网络
        self.model01 = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(),
        )   # 输出 512
        self.model02 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
        )   # 输出 512
        self.model03 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, nx * nz)
        )

    # 前向计算
    def forward(self, x):
        out1 = self.model01(x)
        out2 = self.model02(out1)
        out = self.model03(out1 + out2)     # 跳跃连接
        return out


# 最基础的NN模型
class FNN(nn.Module):
    """
    最基本的线性神经网络（多层感知机）
    输入 时间t、热流H、倾角d、年龄Y、俯冲速度v（深度D、长度L暂时不做输入，当前数据深度为600km，长度为660km）
    输出 温度场，数据数量（分辨率）为 nx × nz
    """

    # 构造函数
    def __init__(self, in_features: int = 10, out_features: int = 3):
        super().__init__()

        # 较简单的前向神经网络
        self.model = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.Sigmoid(),   # 因为能量守恒公式存在温度对时间空间的二阶导，所以最后一层使用sigmoid激活函数，否则会导致二阶导为0
            # nn.Dropout(0.1),
            nn.Linear(100, out_features),
            # nn.Sigmoid()        
        )

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


# 最基础的DCNN模型
class DCNN(nn.Module):
    """
    最基本的线性神经网络（多层感知机）
    输入 时间t、热流H、倾角d、年龄Y、俯冲速度v、横坐标x、纵坐标z（深度D、长度L暂时不做输入，当前数据深度为600km，长度为660km）
    输出 温度场，数据数量（分辨率）为 nx × nz (111 × 101)
    """

    # 构造函数
    def __init__(self, nd: int = 5):
        super().__init__()

        # 基于DCGAN的生成网络，输出128×128×3，分别为温度场、x方向上的速度、z方向上的速度
        self.model = nn.Sequential(
            # Conv_1
            nn.ConvTranspose2d(in_channels=nd, out_channels=512, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Conv_2
            nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Conv_3
            nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Conv_4
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Conv_5
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Conv_6
            nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid()
        )

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


# DCGAN 判别器神经网络搭建
class DCGAN_discriminator(nn.Module):

    # 构造函数
    def __init__(self, ngpu):
        super(DCGAN_discriminator, self).__init__()

        self.ngpu = ngpu
        self.add_module("model", nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=(4, 4), bias=False),
            nn.Sigmoid()
        ))

    # 前向计算函数
    def forward(self, x):
        return self.model(x)


# 最基础的LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=32 * 32 * 6):
        super().__init__()
        self.hidden_size = hidden_size

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True  # 输入形状为(batch, seq, feature)
        )

        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        # 初始化隐藏状态和细胞状态
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)
            c0 = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)
            hidden = (h0, c0)

        # LSTM前向传播
        out, (hn, cn) = self.lstm(x, hidden)  # out形状: (batch, seq, hidden_size)

        # 只取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层预测
        out = self.fc(out)
        return out, (hn, cn)


# 最基础的NN模型
class PINN(nn.Module):
    """
    最基本的线性神经网络（多层感知机）
    输入 时间t、热流H、倾角d、年龄Y、俯冲速度v（深度D、长度L暂时不做输入，当前数据深度为600km，长度为660km）
    输出 温度场，数据数量（分辨率）为 nx × nz
    """

    # 构造函数
    def __init__(self, nx: int = 111, nz: int = 101):
        super().__init__()

        # 较简单的前向神经网络
        self.model = nn.Sequential(
            nn.Linear(116, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, nx * nz)
        )

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x

    # 计算PDE
    def pde_loss(self, x, vx):
        # dT/dt + vx * dT/dx - d²T/d²t = 0
        pass

