from torch import nn


# ConvAE的
class ConvAE(nn.Module):
    def __init__(self, middle_channels: int, final_channels: int, kernel_size: int = 5, stride: int = 3):
        super(ConvAE, self).__init__()
        # 编码层
        self.add_module('encoder', nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=middle_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride)),
            nn.Tanh(),
            nn.Conv2d(in_channels=middle_channels, out_channels=final_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride)),
            nn.Tanh()
        ))
        # 解码层
        self.add_module('decoder', nn.Sequential(
            nn.ConvTranspose2d(in_channels=final_channels, out_channels=middle_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride)),
            nn.Tanh(),
            # 最后一层，直接输出完整的图片的通道
            nn.ConvTranspose2d(in_channels=middle_channels, out_channels=1, kernel_size=(kernel_size, kernel_size), stride=(stride, stride)),
            nn.Sigmoid()
        ))

    # 前向计算函数
    def forward(self, x):
        z = self.encode(x)
        output = self.decode(z)
        return output

    # 编码处理
    def encode(self, x):
        z = self.encoder(x)
        return z

    # 解码处理
    def decode(self, z):
        output = self.decoder(z)
        return output


class ConvAE_5Layer(nn.Module):
    def __init__(self):
        super(ConvAE_5Layer, self).__init__()
        # 编码层
        self.add_module('encoder', nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(5, 5), stride=(3, 3)),
            nn.Tanh(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(5, 5), stride=(3, 3)),
            nn.Tanh(),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(5, 5), stride=(3, 3)),
            nn.Tanh(),
        ))
        # 解码层
        self.add_module('decoder', nn.Sequential(
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=(5, 5), stride=(3, 3)),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=(5, 5), stride=(3, 3)),
            nn.Tanh(),
            # 最后一层，直接输出完整的图片的通道
            nn.ConvTranspose2d(in_channels=12, out_channels=1, kernel_size=(5, 5), stride=(3, 3)),
            nn.Sigmoid()
        ))

    # 前向计算函数
    def forward(self, x):
        z = self.encode(x)
        output = self.decode(z)
        return output

    # 编码处理
    def encode(self, x):
        z = self.encoder(x)
        return z

    # 解码处理
    def decode(self, z):
        output = self.decoder(z)
        return output

