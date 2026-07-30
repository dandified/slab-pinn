import math
import os
import random
from pathlib import Path
import shutil
import tqdm


# 待处理的原始数据文件夹
data_dir = "D:\\地球动力学与PINN\\ProcessedData"
target_dir = "C:\\GeoScience\\俯冲带与PINN\\俯冲板片\\train\\PINN\\data\\fnn"


# 复制文件到目标路径
def copy_file(src_file: str, dest_file: str):
    # print(f"复制文件：{src_file} -> {dest_file}")
    shutil.copy2(src_file, dest_file)


# 将数据集路径进行复制
def copy_dataset(tag: str, data: list[str]):
    # 创建文件夹
    final_dir = f"{target_dir}/nn/{tag}"
    os.makedirs(f"{final_dir}/input", exist_ok=True)
    os.makedirs(f"{final_dir}/output", exist_ok=True)

    for data_file in data:
        # 将输入输出分开
        file_path = Path(data_file)
        output_file = data_file.replace('Elaps', 'Dat_uniT')
        dir_name = os.path.basename(file_path.parent)
        filename = file_path.name
        name, ext = filename.split('.')
        copy_file(data_file, f"{final_dir}/input/{dir_name}.{ext}")
        copy_file(output_file, f"{final_dir}/output/{dir_name}.{ext}")


# 将数据集路径进行复制
def copy_dataset2(tag: str, data: list[str]):
    # 创建文件夹
    final_dir = f"{target_dir}/{tag}"
    os.makedirs(f"{final_dir}/input", exist_ok=True)
    os.makedirs(f"{final_dir}/output", exist_ok=True)

    for data_file in tqdm.tqdm(data):
        # 将输入输出分开
        copy_file(f"{target_dir}/origin/input/{data_file}", f"{final_dir}/input/{data_file}")
        copy_file(f"{target_dir}/origin/output/{data_file}", f"{final_dir}/output/{data_file}")


# 将数据集路径进行复制
def copy_dataset_fnn(tag: str, data: list[str]):
    # 创建文件夹
    orgin_data_path = f"{target_dir}/origin"
    final_dir = f"{target_dir}/{tag}"
    os.makedirs(f"{final_dir}/input", exist_ok=True)
    os.makedirs(f"{final_dir}/output", exist_ok=True)

    for data_dirs in data:
        data_files = os.listdir(f"{orgin_data_path}/{data_dirs}")
        for data_file in data_files:
            # 将输入输出分开
            if "Elaps" in data_file:
                data_path = f"{orgin_data_path}/{data_dirs}/{data_file}"
                file_path = Path(data_path)
                filename = file_path.name
                name, ext = filename.split('.')

                # 读取速度和温度
                with open(
                        f"{orgin_data_path}/{data_dirs}/Elaps.{ext}") as inputf:
                    input_params = inputf.read().replace(',', ' ')
                with open(
                        f"{orgin_data_path}/{data_dirs}/Dat_uniXV.{ext}") as xf:
                    xv = xf.readlines()
                with open(
                        f"{orgin_data_path}/{data_dirs}/Dat_uniXV.{ext}") as zf:
                    zv = zf.readlines()
                with open(
                        f"{orgin_data_path}/{data_dirs}/Dat_uniT.{ext}") as tf:
                    T = tf.readlines()

                input_param = []
                output_param = []

                for i in range(len(T)):
                    # 拼接输入
                    Temperature = T[i].replace('\n', '').split(',')
                    X_Velocity = xv[i].replace('\n', '').split(',')
                    Z_Velocity = zv[i].replace('\n', '').split(',')

                    input_param.append(f"{Temperature[0]} {Temperature[1]} {input_params}")
                    output_param.append(f"{Temperature[2]} {X_Velocity[2]} {Z_Velocity[2]}")

                with open(f"{final_dir}/input/{data_dirs}.{ext}", "wt", encoding="utf-8") as wf:
                    wf.write(','.join(input_param))

                with open(f"{final_dir}/output/{data_dirs}.{ext}", "wt", encoding="utf-8") as wf:
                    wf.write(','.join(output_param))


# 将数据集路径进行复制
def copy_dataset_fnn2(tag: str, data: list[str], data_tag: str = "20260607"):
    # 创建文件夹
    orgin_data_path = f"{target_dir}/origin_{data_tag}"
    final_dir = f"{target_dir}/{tag}"
    os.makedirs(f"{final_dir}/input", exist_ok=True)
    os.makedirs(f"{final_dir}/output", exist_ok=True)

    for data_dirs in data:
        data_files = os.listdir(f"{orgin_data_path}/{data_dirs}")
        for data_file in data_files:
            # 将输入输出分开
            if "Elaps" in data_file:
                data_path = f"{orgin_data_path}/{data_dirs}/{data_file}"
                file_path = Path(data_path)
                filename = file_path.name
                name, ext = filename.split('.')

                # 读取速度和温度
                with open(
                        f"{orgin_data_path}/{data_dirs}/Elaps.{ext}") as inputf:
                    input_params = inputf.read().replace(',', ' ')
                    
                with open(
                        f"{orgin_data_path}/{data_dirs}/Dat_uniXV.{ext}") as xf:
                    xv = xf.readlines()
                with open(
                        f"{orgin_data_path}/{data_dirs}/Dat_uniXV.{ext}") as zf:
                    zv = zf.readlines()
                    
                with open(
                        f"{orgin_data_path}/{data_dirs}/Dat_uniT.{ext}") as tf:
                    T = tf.readlines()

                input_param = []
                output_param = []

                for i in range(len(T)):
                    # 拼接输入
                    Temperature = T[i].replace('\n', '').split(',')
                    X_Velocity = xv[i].replace('\n', '').split(',')
                    Z_Velocity = zv[i].replace('\n', '').split(',')

                    input_param.append(f"{Temperature[0]} {Temperature[1]} {input_params}")
                    output_param.append(f"{Temperature[2]} {X_Velocity[2]} {Z_Velocity[2]}")

                with open(f"{final_dir}/input/{data_dirs}.{ext}", "wt", encoding="utf-8") as wf:
                    wf.write(','.join(input_param))

                with open(f"{final_dir}/output/{data_dirs}.{ext}", "wt", encoding="utf-8") as wf:
                    wf.write(','.join(output_param))

# 将数据集路径进行复制
def copy_dataset_lstm(tag: str, data: list[str]):
    # 创建文件夹
    final_dir = f"{target_dir}/lstm/{tag}"
    os.makedirs(f"{final_dir}/input", exist_ok=True)
    os.makedirs(f"{final_dir}/output", exist_ok=True)

    for data_dirs in data:
        data_files = os.listdir(f"{target_dir}/origin/{data_dirs}")
        for data_file in data_files:
            # 将输入输出分开
            data_path = f"{target_dir}/origin/{data_dirs}/{data_file}"
            file_path = Path(data_path)
            output_file = data_path.replace('Elaps', 'Dat_uniT')
            dir_name = os.path.basename(file_path.parent)
            filename = file_path.name
            name, ext = filename.split('.')
            copy_file(data_path, f"{final_dir}/input/{dir_name}.{ext}")
            copy_file(output_file, f"{final_dir}/output/{dir_name}.{ext}")


# 按照一定比例分割数据集
def split_dataset(data: list[str], ratio: float = 0.5):
    size1 = math.ceil(len(data) * ratio)  # 目标集大小

    data1 = random.sample(data, size1)
    data2 = list(set(data) - set(data1))  # 确保另一个数据集是从剩余元素中选取的，避免重复

    return data1, data2


def split_nn():
    data = os.listdir(f"{target_dir}/origin")
    final_data = []
    for dir_data in data:
        for file_data in os.listdir(f"{target_dir}/origin/{dir_data}"):
            if "Elaps" in file_data:
                final_data.append(f"{target_dir}/origin/{dir_data}/{file_data}")

    # 80%作为训练集
    train_data, test_valid_data = split_dataset(final_data, 0.8)

    # 剩余各10%为测试集和验证集
    test_data, valid_data = split_dataset(test_valid_data, 0.5)
    print(f"训练集：{len(train_data)}")
    print(f"测试集：{len(test_data)}")
    print(f"验证集：{len(valid_data)}")
    print(f"总计： {len(train_data) + len(test_data) + len(valid_data)}")
    copy_dataset("train", train_data)
    copy_dataset("test", test_data)
    copy_dataset("valid", valid_data)


def split_lstm():
    data = os.listdir(f"{target_dir}/origin")

    # 80%作为训练集
    train_data, test_valid_data = split_dataset(data, 0.8)

    # 剩余各10%为测试集和验证集
    test_data, valid_data = split_dataset(test_valid_data, 0.5)
    print(f"训练集：{len(train_data)}")
    print(f"测试集：{len(test_data)}")
    print(f"验证集：{len(valid_data)}")
    print(f"总计： {len(train_data) + len(test_data) + len(valid_data)}")
    copy_dataset_lstm("train", train_data)
    copy_dataset_lstm("test", test_data)
    copy_dataset_lstm("valid", valid_data)


def split_fnn():
    data = os.listdir(f"{target_dir}/origin")

    # 80%作为训练集
    train_data, test_valid_data = split_dataset(data, 0.02)         # 0.8

    # 剩余各10%为测试集和验证集
    test_data, valid_data = split_dataset(test_valid_data, 0.995)    # 0.5
    print(f"训练集：{len(train_data)}")
    print(f"测试集：{len(test_data)}")
    print(f"验证集：{len(valid_data)}")
    print(f"总计： {len(train_data) + len(test_data) + len(valid_data)}")
    copy_dataset_fnn("train", train_data)
    copy_dataset_fnn("test", test_data)
    copy_dataset_fnn("valid", valid_data)


def split_fnn2():
    data = os.listdir(f"{target_dir}/origin_new_7500/input")

    # 80%作为训练集
    train_data, test_valid_data = split_dataset(data, 0.7)         # 0.8

    # 剩余各10%为测试集和验证集
    test_data, valid_data = split_dataset(test_valid_data, 0.67)    # 0.5
    print(f"训练集：{len(train_data)}")
    print(f"测试集：{len(test_data)}")
    print(f"验证集：{len(valid_data)}")
    print(f"总计： {len(train_data) + len(test_data) + len(valid_data)}")
    copy_dataset2("train", train_data)
    copy_dataset2("test", test_data)
    copy_dataset2("valid", valid_data)


def split_fnn3():
    data = os.listdir(f"{target_dir}/origin_20260607")

    # 80%作为训练集
    copy_dataset_fnn2("valid_20260607", data)


def split_fnn4():
    data = os.listdir(f"{target_dir}/train_0.7_7500/input")

    # 80%作为训练集
    train_data, _ = split_dataset(data, 0.107143)         # 0.8

    print(f"训练集：{len(train_data)}")
    copy_dataset2("train", train_data)


def split_fnn5():
    tag = "20260721"
    data = os.listdir(f"{target_dir}/origin_{tag}")

    # 80%作为训练集
    copy_dataset_fnn2(f"test_{tag}", data, tag)


def generate_input():
    for i in range(6, 7):
        for j in range(2, 3):
            for k in range(4, 5):
                for l in range(1, 2):
                    # 速度设置
                    velname = f"Vel{l:02}"
                    vel = 3.17 * l
                    # vel = 3.17 * 15.0

                    # 倾角设置，dip等于k*15.0
                    dipname = f"{k * 15.0:.1f}"
                    dip = k * 15.0

                    # 年龄设置
                    agename = f"Age{j:02}"
                    age = 20.0 + 10.0 * (j - 1)
                    # age = 12.0

                    # 热流设置
                    heatname = f"Heat{i:02}"
                    heat = 10.0 * i
                    # heat = 6.0

                    target_data = f"Temp{heatname}{dipname}{agename}{velname}"
                    data_file_list = os.listdir(f"{data_dir}/{target_data}")

                    os.makedirs(f"{target_dir}/origin/{target_data}", exist_ok=True)

                    for data_file in data_file_list:
                        file_name = f"{data_dir}/{target_data}/{data_file}"
                        if "Elaps" in data_file:
                            # 处理输入参数，以时间（Ma）、热流、倾角、板块年龄、俯冲速度
                            with open(file_name, encoding='utf-8') as f:
                                content = f.read()
                                input_params = [
                                    content,
                                    str(heat),
                                    str(dip),
                                    str(age),
                                    str(vel)
                                ]
                                with open(f"{target_dir}/origin/{target_data}/{data_file}", "wt", encoding="utf-8") as wf:
                                    wf.write(",".join(input_params))
                        else:
                            # 处理温度场数据，剔除坐标参数
                            # 坐标范围：x[0, 660], z[600, 0], 均以6为间隔
                            with open(file_name, encoding='utf-8') as f:
                                content = f.readlines()
                                target_temp = []
                                for temp in content:
                                    templist = temp.split(" ")
                                    target_temp.append(templist[2][:-1])
                                with open(f"{target_dir}/origin/{target_data}/{data_file}", "wt", encoding="utf-8") as wf:
                                    wf.write(",".join(target_temp))


def generate_input_fnn():
    tag = "20260721"

    for i in range(6, 7):
        for j in range(2, 3):
            for k in range(3, 4):
                for l in range(2, 3):
                    # 速度设置
                    velname = f"Vel{l:02}"
                    vel = 3.17 * l
                    # vel = 3.17 * 15.0

                    # 倾角设置，dip等于k*15.0
                    dipname = f"{k * 15.0:.1f}"
                    dip = k * 15.0

                    # 年龄设置
                    agename = f"Age{j:02}"
                    age = 20.0 + 10.0 * (j - 1)
                    # age = 12.0

                    # 热流设置
                    heatname = f"Heat{i:02}"
                    heat = 10.0 * i
                    # heat = 6.0

                    target_data = f"Temp{heatname}{dipname}{agename}{velname}_{tag}"
                    data_file_list = os.listdir(f"{data_dir}/{target_data}")

                    os.makedirs(f"{target_dir}/origin_{tag}/{target_data}", exist_ok=True)

                    for data_file in data_file_list:
                        file_name = f"{data_dir}/{target_data}/{data_file}"
                        print(file_name)
                        if "Elaps" in data_file:
                            # 处理输入参数，以时间（Ma）、热流、倾角、板块年龄、俯冲速度
                            with open(file_name, encoding='utf-8') as f:
                                content = f.read()
                                input_params = [
                                    content,
                                    str(heat),
                                    str(dip),
                                    str(age),
                                    str(vel)
                                ]
                                with open(f"{target_dir}/origin_{tag}/{target_data}/{data_file}", "wt", encoding="utf-8") as wf:
                                    wf.write(",".join(input_params))
                        else:
                            # 处理温度场数据，剔除坐标参数
                            # 坐标范围：x[0, 660], z[600, 0], 均以6为间隔
                            with open(file_name, encoding='utf-8') as f:
                                content = f.readlines()
                                target_temp = []
                                for temp in content:
                                    templist = temp.replace(' ', ',')
                                    target_temp.append(templist)
                                with open(f"{target_dir}/origin_{tag}/{target_data}/{data_file}", "wt", encoding="utf-8") as wf:
                                    wf.write("".join(target_temp))


def generate_input_fnn2():
    for i in range(2, 9):
        for j in range(1, 4):
            for k in range(2, 3):
                for l in range(1, 4):
                    # 速度设置
                    velname = f"Vel{l:02}"
                    vel = 3.17 * 2.0 + 0.317 * l
                    # vel = 3.17 * 15.0

                    # 倾角设置
                    dipname = "45.0"
                    dip = 45.0

                    # 年龄设置
                    agename = f"Age{j:02}"
                    age = 40.0 + 1.0 * (j - 1)
                    # age = 12.0

                    # 热流设置
                    heatname = f"Heat{i:02}"
                    heat = 1.0 * i + 35.0
                    # heat = 6.0

                    target_data = f"Temp{heatname}{dipname}{agename}{velname}_20260607"
                    data_file_list = os.listdir(f"{data_dir}/{target_data}")

                    os.makedirs(f"{target_dir}/origin_20260607/{target_data}", exist_ok=True)

                    for data_file in data_file_list:
                        file_name = f"{data_dir}/{target_data}/{data_file}"
                        print(file_name)
                        if "Elaps" in data_file:
                            # 处理输入参数，以时间（Ma）、热流、倾角、板块年龄、俯冲速度
                            with open(file_name, encoding='utf-8') as f:
                                content = f.read()
                                input_params = [
                                    content,
                                    str(heat),
                                    str(dip),
                                    str(age),
                                    str(vel)
                                ]
                                with open(f"{target_dir}/origin_20260607/{target_data}/{data_file}", "wt", encoding="utf-8") as wf:
                                    wf.write(",".join(input_params))
                        else:
                            # 处理温度场数据，剔除坐标参数
                            # 坐标范围：x[0, 660], z[600, 0], 均以6为间隔
                            with open(file_name, encoding='utf-8') as f:
                                content = f.readlines()
                                target_temp = []
                                for temp in content:
                                    templist = temp.replace(' ', ',')
                                    target_temp.append(templist)
                                with open(f"{target_dir}/origin_20260607/{target_data}/{data_file}", "wt", encoding="utf-8") as wf:
                                    wf.write("".join(target_temp))


if __name__ == "__main__":
    # generate_input()
    # split_nn()
    # split_lstm()
    # generate_input_fnn()
    # generate_input_fnn2()
    # split_fnn()
    # split_fnn2()
    # split_fnn3()
    # split_fnn4()
    # split_fnn5()
    
    # shutil.rmtree("D:\\GeoScience\\俯冲带与PINN\\俯冲板片\\train\\data\\fnn\\train")
