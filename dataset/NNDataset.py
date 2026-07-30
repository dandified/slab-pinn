import numpy
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision
import numpy as np
import torch


# 温度数据限制，用于归一化温度场
T_MIN = 0.0
T_MAX = 1320.0

# 时间模拟步数间隔
STEP_SIZE = 50
SINGLE_LENGTH = int(15000 / 50 + 1)


class NNDataset(Dataset):
    def __init__(self, root_dir: str, tag: str):
        """
        初始化数据集加载
        :param root_dir: 数据集根目录
        :param tag: 当前加载数据集标签
        """
        self.root_dir = root_dir
        self.data_dir = f"{root_dir}/{tag}"
        self.input_data = os.listdir(f"{self.data_dir}/input")

    def __getitem__(self, item):
        """
        获取单一数据
        :param item: 目标数据索引
        :return: item索引对应的一组数据
        """
        input_file = f"{self.data_dir}/input/{self.input_data[item]}"
        output_file = f"{self.data_dir}/output/{self.input_data[item]}"
        input_data = np.loadtxt(input_file, delimiter=',')
        output_data = np.loadtxt(output_file, delimiter=',')
        # 对温度场进行归一化
        output_data = np.clip(output_data, 0.0, 1320.0) / 1320.0
        return torch.from_numpy(input_data).float(), torch.from_numpy(output_data).float(), self.input_data[item]

    def __len__(self):
        """
        获取数据集总长度
        :return: 返回当前数据长度
        """
        return len(self.input_data)


class FNNDataset(Dataset):
    def __init__(self, root_dir: str, tag: str, nx: int = 111, nz: int = 101, output_index_start: int = 0, output_index_end: int = 3):
        """
        初始化数据集加载
        :param root_dir: 数据集根目录
        :param tag: 当前加载数据集标签
        """
        self.output_index_start = output_index_start
        self.output_index_end = output_index_end
        self.root_dir = root_dir
        self.data_dir = f"{root_dir}/{tag}"
        self.input_data = os.listdir(f"{self.data_dir}/input")
        self.single_num = nx * nz

    def __getitem__(self, item):
        """
        获取单一数据
        :param item: 目标数据索引
        :return: item索引对应的一组数据
        """
        file_index, data_index = divmod(item, self.single_num)
        input_file = f"{self.data_dir}/input/{self.input_data[file_index]}"
        output_file = f"{self.data_dir}/output/{self.input_data[file_index]}"

        # print(f"开始读取数据【{item}】：{self.input_data[file_index]}，数据索引{data_index}")

        with open(input_file, 'rt', encoding="utf-8") as inputf:
            content = inputf.read().split(',')
            # print(len(content))
            input_params = content[data_index]

        with open(output_file, 'rt', encoding="utf-8") as outputf:
            content = outputf.read().split(',')
            output_params = content[data_index]

        input_data = np.fromstring(input_params, dtype=float, sep=' ')
        # 对坐标进行归一化
        input_data[0] = input_data[0]
        input_data[1] = input_data[1]
        output_data = np.fromstring(output_params, dtype=float, sep=' ')
        # print(output_data)
        # 对温度场进行归一化
        output_data[0] = np.clip(output_data[0], 0.0, 1320.0) / 1320.0
        # 对速度场进行归一化，方便训练收敛
        output_data[1] = (output_data[1] * np.sqrt(2.0) / input_data[6] + 1.0) / 2.0
        output_data[2] = (output_data[2] * np.sqrt(2.0) / input_data[6] + 1.0) / 2.0
        # print(output_data)
        return torch.from_numpy(input_data).float(), torch.from_numpy(output_data[self.output_index_start:self.output_index_end]).float(), self.input_data[file_index]

    def __len__(self):
        """
        获取数据集总长度
        :return: 返回当前数据长度
        """
        return len(self.input_data) * self.single_num


# 用于整体测试的NN数据集
class FNNTestDataset(FNNDataset):
    def load_data_from_string(self, content):
        """从字符串加载数据"""
        # 按逗号分割
        lines = content.strip().split(',')
        # 过滤空行
        lines = [line for line in lines if line.strip()]
        
        # 处理每行数据
        data = []
        for line in lines:
            # 按空格分割（支持多个空格）
            values = list(map(float, line.strip().split()))
            data.append(values)
        
        # 转换为numpy数组
        return np.array(data)

    def load_data_from_file(self, file_path):
        """从文件加载数据"""
        with open(file_path, 'r') as f:
            content = f.read().strip()
        return self.load_data_from_string(content)

    def __getitem__(self, item):
        """
        获取单一数据
        :param item: 目标数据索引
        :return: item索引对应的一组数据
        """
        # file_index, data_index = divmod(item, self.single_num)
        input_file = f"{self.data_dir}/input/{self.input_data[item]}"
        output_file = f"{self.data_dir}/output/{self.input_data[item]}"

        input_data = self.load_data_from_file(input_file)
        output_data = self.load_data_from_file(output_file)

        # print(input_data[:, 6])
        # print(output_data[:, 1])

        # 对温度场进行归一化
        output_data[:, 0] = np.clip(output_data[:, 0], 0.0, 1320.0) / 1320.0
        # 对速度场进行归一化，方便训练收敛
        output_data[:, 1] = (output_data[:, 1] * np.sqrt(2.0) / input_data[:, 6][0] + 1.0) / 2.0
        output_data[:, 2] = (output_data[:, 2] * np.sqrt(2.0) / input_data[:, 6][0] + 1.0) / 2.0
        # print(output_data)
        return torch.from_numpy(input_data).float(), torch.from_numpy(output_data).float(), self.input_data[item]


    def __len__(self):
        """
        获取数据集总长度
        :return: 返回当前数据长度
        """
        return len(self.input_data)


# 结合自编码器的NN数据集
class AENNDataset(NNDataset):
    def __init__(self, root_dir: str, tag: str, nx: int = 111, nz: int = 101):
        """
        初始化数据集加载
        :param root_dir: 数据集根目录
        :param tag: 当前加载数据集标签
        """
        super().__init__(root_dir, tag)
        self.nx = nx
        self.nz = nz

    def __getitem__(self, item):
        """
        获取单一数据
        :param item: 目标数据索引
        :return: item索引对应的一组数据
        """
        input_file = f"{self.data_dir}/input/{self.input_data[item]}"
        output_file = f"{self.data_dir}/output/{self.input_data[item]}"
        input_data = np.loadtxt(input_file, delimiter=',')
        output_data = np.loadtxt(output_file, delimiter=',')
        # 对温度场进行归一化
        output_data = np.clip(output_data, 0.0, 1320.0) / 1320.0
        output_data = torch.from_numpy(output_data).float()
        output_tensor = torch.reshape(output_data, (1, self.nx, self.nz))
        Resize = torchvision.transforms.Resize((128, 128))
        output_tensor = Resize(output_tensor)
        return torch.from_numpy(input_data).float(), output_tensor, self.input_data[item]


class DCNNDataset(NNDataset):
    def __init__(self, root_dir: str, tag: str, nx: int = 111, nz: int = 101):
        super().__init__(root_dir, tag)
        self.nx = nx
        self.nz = nz

    def __getitem__(self, item):
        input_file = f"{self.data_dir}/input/{self.input_data[item]}"
        output_file = f"{self.data_dir}/output/{self.input_data[item]}"
        input_data = torch.from_numpy(np.loadtxt(input_file, delimiter=',')).float()
        output_data = np.loadtxt(output_file, delimiter=',')
        # 对温度场进行归一化
        output_data = np.clip(output_data, 0.0, 1320.0) / 1320.0
        output_data = torch.from_numpy(output_data).float()
        # 将输入转为图片类型的张量
        input_tensor = torch.reshape(input_data, (len(input_data), 1, 1))
        output_tensor = torch.reshape(output_data, (1, self.nx, self.nz))
        Resize = torchvision.transforms.Resize((128, 128))
        output_tensor = Resize(output_tensor)
        return input_tensor, output_tensor, self.input_data[item]


class DCPINNDataset(DCNNDataset):
    def __getitem__(self, item):
        input_file = f"{self.data_dir}/input/{self.input_data[item]}"
        output_file = f"{self.data_dir}/output/{self.input_data[item]}"

        with open(input_file) as inf:
            in_datas = inf.read()
            input_data = [float(in_data) for in_data in in_datas.split(',')]
        input_data = torch.Tensor(input_data)
        output_data = np.loadtxt(output_file, delimiter=',')
        # 对温度场进行归一化
        output_data = np.clip(output_data, 0.0, 1320.0) / 1320.0
        output_data = torch.from_numpy(output_data).float()
        # 将输入转为图片类型的张量
        input_tensor = torch.reshape(input_data, (len(input_data), 1, 1))
        output_tensor = torch.reshape(output_data, (1, self.nx, self.nz))
        Resize = torchvision.transforms.Resize((128, 128))
        output_tensor = Resize(output_tensor)
        return input_tensor, output_tensor, self.input_data[item]



class AEDataset(DCNNDataset):
    def __getitem__(self, item):
        input_tensor, output_tensor, _ = super().__getitem__(item)
        # 放大到128方便提取特征
        Resize = torchvision.transforms.Resize((128, 128))
        output_tensor = Resize(output_tensor)
        return output_tensor, self.input_data[item]


class LSTMDataset(Dataset):
    def __init__(self, root_dir: str, tag: str, train_window: int = 5):
        """
        初始化数据集加载
        :param root_dir: 数据集根目录
        :param tag: 当前加载数据集标签
        :param train_window: 训练窗口长度
        """
        self.root_dir = root_dir
        self.data_dir = f"{root_dir}/{tag}"
        self.input_data = os.listdir(f"{self.data_dir}/input")
        self.train_window = train_window
        # 每组模拟数据的最大索引值
        self.single_seq_max = SINGLE_LENGTH - train_window
        # 一共多少组数据
        self.seq_num = len(self.input_data) / SINGLE_LENGTH

    def __getitem__(self, item):
        """
        获取单一数据
        :param item: 目标数据索引
        :return: item索引对应的一组数据
        """
        output_file = f"{self.data_dir}/output/{self.input_data[item]}"
        output_data = np.loadtxt(output_file, delimiter=',')
        # 对温度场进行归一化
        output_data = np.clip(output_data, 0.0, 1320.0) / 1320.0
        # 抽取训练窗口的数据
        input_seq = []
        for i in range(self.train_window):
            # 获取时间步数
            file_name, ext = os.path.splitext(self.input_data[item])
            input_file = f"{self.data_dir}/input/{file_name}.{int(ext[1:]) + int(i * 50)}"
            input_data = np.loadtxt(input_file, delimiter=',')
            input_seq.append(torch.from_numpy(input_data).float())
        return (input_seq, torch.from_numpy(output_data).float()), self.input_data[item]

    def __len__(self):
        """
        获取数据集总长度
        :return: 返回当前数据长度
        """
        return self.seq_num * self.single_seq_max


if __name__ == "__main__":
    file_name, ext = os.path.splitext("TempHeat0045.0Age01Vel01.50")
    print(int(ext[1:])/STEP_SIZE)
    print(file_name)
    print(SINGLE_LENGTH)
