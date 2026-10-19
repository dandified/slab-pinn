import argparse
import uuid
from tqdm import tqdm
from model import FNN
from dataset import FNNDataset, FNNTestDataset
from dataset.NNDataset import P_SCALE, VISCOSITY_0
import torch
import os
import numpy as np
from torch import nn
import torchvision
import traceback
from torch.utils.data import DataLoader
from generate_boundary import generate_BC_Tensor, gradient

# 全局参数定义
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
root_dir = "./data/fnn"
# 数据集标签（output 含 T/Vx/Vz/P 四个预测通道，第5列 Vi 为逐点粘度）
data_tag = "train_20260831_p"
workers = 2
nx = 111
nz = 101

# 时间间隔常量
time_constant = 1.0
# 损失权重默认值：alpha 质量守恒 / beta 能量守恒 / gamma 初始条件 / delta 边界条件 / epsilon Stokes 动量残差
# 训练时以命令行 --weights 传入为准，此处仅作默认值及手动调用时的回退
alpha = 1.0
beta = 1.0
gamma = 1.0
delta = 1.0
epsilon = 1.0


def run_name_of(alpha: float, beta: float, gamma: float, delta: float, epsilon: float) -> str:
    """该组权重对应的结果文件/目录名前缀"""
    return f"pifnn_{alpha}_{beta}_{gamma}_{delta}_{epsilon}"


# 计算初始条件，注意，热量需要乘0.001
def ic_loss(input_):
    h1 = 7000.0
    h2 = 30000.0
    wholedepth = 600000.0
    conduct = 2.5
    Q1 = 1.3e-6
    Q2 = 2.7e-7
    # lithdepth = 100000.0
    c2 = (Q1 - Q2) * h1 * h1 / 2.0 / conduct

    # x_tensor.requires_grad = True

    # print(input_[:, 3])

    a2 = (input_[:, 3] * 0.001-h1*(Q1-Q2))/conduct

    # print(input_[:, 1])

    i = 100 - input_[:, 1] / 6
    j = input_[:, 0] / 6

    X2 = i / 100

    # print(i)
    # print(j)
    # print(X2)

    z = (1.0 - X2)*600000./2./torch.sqrt(input_[:, 5] * 22932979.2)
    t = 1.0 / (1.0 + 0.5 * z)

    left_bc = torch.where(z < 1.0, 1. - t*torch.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
                            t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
                            t*(-0.82215223+t*0.17087277))))))))), 1.0)

    temperature2 = a2 * (h1 + h2) + c2 - Q2 / conduct / 2.0 * (h1 + h2) * (h1 + h2)
    qmoho = -Q2 * (h1 + h2) / conduct + a2

    right = torch.where(X2 > (1. - h1/wholedepth), input_[:, 3] * 0.001/conduct*(1.0 - X2)*wholedepth - Q1 /conduct /2.0*(1.0-X2)*wholedepth*(1.0-X2)*wholedepth, 
                torch.where(X2 > (1.0 -(h1 + h2)/wholedepth), a2*(1.0-X2)*wholedepth+c2-Q2/conduct/2.0*(1.0-X2)*wholedepth*(1.0-X2)*wholedepth, 
                torch.clamp(temperature2+qmoho*(1.0-X2-(h1+h2)/wholedepth)*wholedepth, max=1320.0)))
    
    # print(right)

    right_bc = right / 1320.0

    final = torch.where(i + j < 100, left_bc, torch.where(i + j > 100, right_bc, 0.5 * (left_bc + right_bc)))

    return torch.clamp(final, min=0.0, max=1.0)


# pde损失函数
# 目前质量守恒方程量级在能量守恒方程的10%
# 若beta偏大，会导致模型过拟合
def pde_loss(input_, output_, viscosity, alpha, beta, epsilon):
    # print(input_[:, 6])
    # print(output_[:, 1])

    # 全局热扩散率，W/m³
    k = 3.0
    # 地幔密度，kg/m³
    rho = 3400.0
    # 比热容，J/(kg·K)
    cp = 1250.0

    T = output_[:, 0] * 1320.0  # 单位：K
    U = (output_[:, 1] * 2.0 - 1.0) * input_[:, 6] / np.sqrt(2.0) * 1e-10   # 单位：m/s
    V = (output_[:, 2] * 2.0 - 1.0) * input_[:, 6] / np.sqrt(2.0) * 1e-10   # 单位：m/s
    # 压力反归一化，单位：Pa
    P = (output_[:, 3] - 0.5) * 2.0 * P_SCALE
    # 逐点动力粘度，单位：Pa·s（Dat_uniVi 为无量纲相对粘度，地表约为1）
    mu = viscosity * VISCOSITY_0

    # print(U)
    # 距离单位从km转换为m
    dU_dx = gradient(U, input_)[0][:, 0] / (660 * 1e3)
    dV_dz = gradient(V, input_)[0][:, 1] / (600 * 1e3)

    # 质量守恒方程
    mass_formula = dU_dx + dV_dz
    # print(mass_formula)
    # 能量守恒方程

    # print(gradient(T, input_))
    # 距离单位从km转换为m
    dT_dx = gradient(T, input_)[0][:, 0] / (1e3)
    dT_dz = gradient(T, input_)[0][:, 1] / (1e3)
    # 时间单位从Ma转换为s
    dT_dt = gradient(T, input_)[0][:, 2] / (3.1536 * 1e13)
    # print(dT_dt)
    convection = U * dT_dx + V * dT_dz  # 对流项
    # print(convection)
    # print(gradient(dT_dx, input_))
    # print(gradient(dT_dz, input_))
    
    # 距离单位从km转换为m
    diffusion = k * (gradient(dT_dx, input_)[0][:, 0] + gradient(dT_dz, input_)[0][:, 1]) / (1e3)
    # print(diffusion)
    # 能量方程残差
    # 按照benchmark，稳态方程没有dT_dt项
    energy_residual = rho * cp * (convection + dT_dt) - diffusion
    # print(energy_residual)
    # energy_formula = U * gradient(T, input_)[0][:, 0] - gradient(T, input_)[0][:, 2] - gradient(T_t, input_)[0][:, 2]

    # Stokes 动量方程残差（不可压缩、低雷诺数，局部常粘度近似：忽略粘度空间导数项）
    # Dat_uniP 为地表置零的动力压力（已扣除静岩压力），故不显式包含浮力项；坐标每求导一次除以1e3换算到m
    # 残差单位：Pa/m
    dP_dx = gradient(P, input_)[0][:, 0] / (1e3)
    dP_dz = gradient(P, input_)[0][:, 1] / (1e3)

    dU_dx_m = gradient(U, input_)[0][:, 0] / (1e3)
    dU_dz_m = gradient(U, input_)[0][:, 1] / (1e3)
    d2U_dx2 = gradient(dU_dx_m, input_)[0][:, 0] / (1e3)
    d2U_dz2 = gradient(dU_dz_m, input_)[0][:, 1] / (1e3)

    dV_dx_m = gradient(V, input_)[0][:, 0] / (1e3)
    dV_dz_m = gradient(V, input_)[0][:, 1] / (1e3)
    d2V_dx2 = gradient(dV_dx_m, input_)[0][:, 0] / (1e3)
    d2V_dz2 = gradient(dV_dz_m, input_)[0][:, 1] / (1e3)

    momentum_x = -dP_dx + mu * (d2U_dx2 + d2U_dz2)
    momentum_z = -dP_dz + mu * (d2V_dx2 + d2V_dz2)
    momentum_formula = torch.mean(momentum_x ** 2 + momentum_z ** 2)

    pde_l = alpha * torch.mean(mass_formula ** 2) + beta * torch.mean(energy_residual ** 2) \
        + epsilon * momentum_formula

    # print(f"PDE 损失：{pde_loss.item()}")

    # 返回均方误差（总体PDE损失，其中动量损失分量）
    return pde_l * time_constant, epsilon * momentum_formula * time_constant


# 训练函数入口（权重经参数传入，一个脚本即可顺序完成多组权重的训练与验证）
def train(alpha: float, beta: float, gamma: float, delta: float, epsilon: float):
    # 超参数
    run_name = run_name_of(alpha, beta, gamma, delta, epsilon)

    # 训练参数
    epoch = 5
    learning_rate = 5e-4
    minimum_validation_loss = 10000000000.0
    batch_size = 128

    # 加载数据
    # model_path = f"./result/conv_ae_{middle_channels}_{final_channels}.pth"
    model_path = f"./result/{run_name}.pth"

    # 数据集
    train_dataset = FNNDataset(root_dir, data_tag, output_index_end=4, with_viscosity=True)
    valid_dataset = FNNDataset(root_dir, data_tag, output_index_end=4, with_viscosity=True)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=workers)

    loss_fn = nn.MSELoss().to(device)

    # 定义FNN网络：7个输入（x,z,t,H,dip,age,vel），4个输出（T,Vx,Vz,P）
    model = FNN(7, 4).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)))
        # model.apply(weights_init)

    # 记录训练的总次数
    total_train_step = 0

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optim, step_size=5000, gamma=0.9)  # 设置学习率下降策略

    model.train()

    training_loss_list = []
    validation_loss_list = []

    try:
        # 训练循环
        for i in range(epoch):
            # 训练
            training_loss = 0.0
            step = 0
            for inputs, real_outputs, _, viscosity in train_dataloader:
                inputs = inputs.to(device)
                inputs.requires_grad = True
                # print(inputs.size())
                # print(inputs)
                real_outputs = real_outputs.to(device)
                viscosity = viscosity.to(device)
                # print(real_outputs)
                model.zero_grad()
                outputs = model(inputs)
                # print(outputs.size())
                # print(outputs)
                pde, momentum = pde_loss(inputs, outputs, viscosity, alpha, beta, epsilon)

                # 添加初始条件约束，令t=0
                ic_real = ic_loss(inputs)
                # print(ic_real)
                # 我需要复制一个input，将其第3列时间项设为0
                ic_inputs = inputs.clone()
                ic_inputs[:, 2] = 0.0
                # print(ic_inputs)
                ic_outputs = model(ic_inputs)[:, 0]
                # print(ic_outputs)
                icLoss = loss_fn(ic_outputs, ic_real) * gamma

                # print(icLoss)

                # 添加边界条件约束，令z=0和z=600的温度为0和1
                bc_inputs_top = inputs.clone()
                bc_inputs_top[:, 1] = 0.0
                bc_inputs_bottom = inputs.clone()
                bc_inputs_bottom[:, 1] = 600.0
                # print(ic_inputs)
                bc_outputs_top = model(bc_inputs_top)[:, 0]
                bc_outputs_bottom = model(bc_inputs_bottom)[:, 0]
                
                bc_real_top = torch.zeros(bc_outputs_top.size()).to(device)
                bc_real_bottom = torch.ones(bc_outputs_bottom.size()).to(device)
                
                # print(bc_outputs_top)
                # print(bc_outputs_bottom)
                bcLoss = (loss_fn(bc_outputs_top, bc_real_top) + loss_fn(bc_outputs_bottom, bc_real_bottom)) * delta

                # print(pde)
                # print(loss_fn(outputs, real_outputs))
                loss = loss_fn(outputs, real_outputs) + pde + icLoss + bcLoss

                loss.backward()
                optim.step()
                scheduler_lr.step()

                # debug 信息
                total_train_step = total_train_step + 1
                training_loss += loss.item()
                step += 1

                print(f"已完成{step}/{len(train_dataloader)}训练，损失：{loss.item()}，其中PDE损失：{pde.item()}，占比：{pde.item() / loss.item() * 100.0:.2f}%（动量残差：{momentum.item()}），其中初始条件损失：{icLoss.item()}，占比：{icLoss.item() / loss.item() * 100.0:.2f}%，其中边界条件损失：{bcLoss.item()}，占比：{bcLoss.item() / loss.item() * 100.0:.2f}%")
                training_loss_list.append(loss.item())

            # 验证
            with torch.no_grad():
                validation_loss = 0.0

                step = 0
                for valids, valid_outputs, _, _ in valid_dataloader:
                    valids = valids.to(device)
                    valid_outputs = valid_outputs.to(device)
                    validation = model(valids)

                    loss = loss_fn(validation, valid_outputs)
                    validation_loss += loss.item()
                    step += 1

                    print(f"已完成{step}/{len(valid_dataloader)}验证，损失：{loss.item()}")

                    validation_loss_list.append(loss.item())

                    # Update the statistics for the best model
                    if validation_loss <= minimum_validation_loss:
                        minimum_validation_loss = validation_loss

                        # 保存当前模型
                        torch.save(model.state_dict(), model_path)
                        print("Model is now saved!")
                        best_model_index = epoch

                torch.save(model.state_dict(), f"./result/{run_name}_{i}.pth")

        # Training finished, print the statistics for the best model
        print('Training Finished')
        torch.save(model.state_dict(), f"./result/{run_name}_final.pth")

        print("Best model has a validation loss of {} and it's in epoch {}".format(minimum_validation_loss,
                                                                                   best_model_index + 1))

        train_file = open(f'./result/{run_name}_trainingData.txt', "w")
        train_file.write(",".join([str(elem) for elem in training_loss_list]))
        train_file.close()

        valid_file = open(f'./result/{run_name}_validationData.txt', "w")
        valid_file.write(",".join([str(elem) for elem in validation_loss_list]))
        valid_file.close()

        print("Training Data saved!")

    except Exception as e:
        print(traceback.format_exc())
        print(f"训练出错：{e}")

        train_file = open(f'./result/{run_name}_trainingData.txt', "w")
        train_file.write(",".join([str(elem) for elem in training_loss_list]))
        train_file.close()

        valid_file = open(f'./result/{run_name}_validationData.txt', "w")
        valid_file.write(",".join([str(elem) for elem in validation_loss_list]))
        valid_file.close()

        print("Training Data saved!")


def bc_test():
    arr = []
    for x in range(0, 666, 6):
        for z in range(0, 606, 6):
            arr.append([x, 600 - z, 217.5, 60.0, 45.0, 40.0, 3.17 * 2.0])

    input_tensor = torch.from_numpy(np.array(arr, dtype=float)).float().to(device)

    final_bc = bc_loss(input_tensor)
    print(final_bc)

# 测试函数
def test(model_path: str, run_name: str = None):
    if run_name is None:
        run_name = run_name_of(alpha, beta, gamma, delta, epsilon)
    model_dict = torch.load(model_path)
    # model = ConvAE(middle_channels, final_channels, kernal_size, stride).to(device)
    model = FNN(7, 4).to(device)

    model.load_state_dict(model_dict)
    model.eval()

    # test_dataset = FNNDataset(root_dir, 'test')
    # test_dataloader = DataLoader(test_dataset, 1, shuffle=True, num_workers=workers)

    loss_fn = nn.MSELoss().to(device)

    test_loss = 0.0

    os.makedirs(f"./result/{run_name}_test", exist_ok=True)
    os.makedirs(f"./result/{run_name}_convert", exist_ok=True)

    with torch.no_grad():
        # 创建行和列的网格索引
        arr = []
        for x in range(0, 666, 6):
            for z in range(0, 606, 6):
                arr.append([x, 600 - z, 1.6, 32.5, 45.0, 67.5, 3.17 * 4.7])

        input_tensor = torch.from_numpy(np.array(arr, dtype=float)).float().to(device)

        fake_img = model(input_tensor)
        print(fake_img)
        fake = fake_img.cpu().numpy()[:, 0]
        # print(fake)
        np.savetxt(f"./result/{run_name}_test/fakeTempHeat0945.0Age09Vel98.300.txt", [fake], fmt="%f", delimiter=" ")

        '''
        for n, (real_img, real_output, file_name) in enumerate(test_dataloader, 0):
            real_img = real_img.to(device)
            real_output = real_output.to(device)
            # print(real_img.size())
            fake_img = model(real_img)

            loss = loss_fn(fake_img, real_output)
            test_loss += loss.item()

            # 保存对比文件
            input_data = real_img.cpu().numpy()
            real = real_output.cpu().numpy()
            fake = fake_img.cpu().numpy()
            np.savetxt(f"./result/fnn_test/input{file_name[0]}.txt", input_data, fmt="%f")
            np.savetxt(f"./result/fnn_test/real{file_name[0]}.txt", real, fmt="%f")
            np.savetxt(f"./result/fnn_test/fake{file_name[0]}.txt", fake, fmt="%f")

    print(f"Test loss: {test_loss / len(test_dataloader)}")
    '''

# 测试函数
def test_full(model_path: str, run_name: str = None):
    if run_name is None:
        run_name = run_name_of(alpha, beta, gamma, delta, epsilon)
    model_dict = torch.load(model_path)
    # model = ConvAE(middle_channels, final_channels, kernal_size, stride).to(device)
    model = FNN(7, 4).to(device)

    model.load_state_dict(model_dict)
    model.eval()

    test_dataset = FNNTestDataset(root_dir, 'test')
    test_dataloader = DataLoader(test_dataset, 1, shuffle=True, num_workers=workers)

    loss_fn = nn.MSELoss().to(device)

    test_loss = 0.0

    os.makedirs(f"./result/{run_name}_test", exist_ok=True)
    os.makedirs(f"./result/{run_name}_convert", exist_ok=True)

    with torch.no_grad():

        mse_arr = []

        for inputs, real_outputs, file_name in tqdm(test_dataloader):
            
            # print(file_name[0])
            
            input_tensor = inputs.to(device)

            fake_img = model(input_tensor)
            # print(fake_img)
            fake = fake_img.cpu().numpy()[:, 0]

            inputs = inputs.to(device)
            real_outputs = real_outputs.to(device)
            # print(real_img.size())
            fake_img = model(inputs)

            loss = loss_fn(fake_img[0], real_outputs[0])
            test_loss += loss.item()
            mse_arr.append([file_name[0], loss.item()])
            # print([file_name[0], loss.item()])

            # 保存对比文件
            input_data = inputs.cpu().numpy()[0]
            # print(input_data)
            real = real_outputs.cpu().numpy()[0]
            # print(real)
            fake = fake_img.cpu().numpy()[0]
            np.savetxt(f"./result/{run_name}_test/input{file_name[0]}.txt", input_data, fmt="%f")
            np.savetxt(f"./result/{run_name}_test/real{file_name[0]}.txt", real, fmt="%f")
            np.savetxt(f"./result/{run_name}_test/fake{file_name[0]}.txt", fake, fmt="%f")

    np.savetxt(f"./result/{run_name}_test/mse.txt", mse_arr, fmt="%s", delimiter=",")
    print(f"Test loss: {test_loss / len(test_dataloader)}")

# 测试函数
def test_full_20260607(model_path: str, run_name: str = None, tag: str = "20260607"):
    if run_name is None:
        run_name = run_name_of(alpha, beta, gamma, delta, epsilon)
    model_dict = torch.load(model_path)
    # model = ConvAE(middle_channels, final_channels, kernal_size, stride).to(device)
    model = FNN(7, 4).to(device)

    model.load_state_dict(model_dict)
    model.eval()

    test_dataset = FNNTestDataset(root_dir, f'test_{tag}')
    test_dataloader = DataLoader(test_dataset, 1, shuffle=True, num_workers=workers)

    loss_fn = nn.MSELoss().to(device)

    test_loss = 0.0

    os.makedirs(f"./result/{run_name}_{tag}_test", exist_ok=True)
    os.makedirs(f"./result/{run_name}_{tag}_convert", exist_ok=True)

    with torch.no_grad():

        mse_arr = []

        for inputs, real_outputs, file_name in tqdm(test_dataloader):
            
            # print(file_name[0])
            
            input_tensor = inputs.to(device)

            fake_img = model(input_tensor)
            # print(fake_img)
            fake = fake_img.cpu().numpy()[:, 0]

            inputs = inputs.to(device)
            real_outputs = real_outputs.to(device)
            # print(real_img.size())
            fake_img = model(inputs)

            loss = loss_fn(fake_img[0], real_outputs[0])
            test_loss += loss.item()
            mse_arr.append([file_name[0], loss.item()])
            # print([file_name[0], loss.item()])

            # 保存对比文件
            input_data = inputs.cpu().numpy()[0]
            # print(input_data)
            real = real_outputs.cpu().numpy()[0]
            # print(real)
            fake = fake_img.cpu().numpy()[0]
            np.savetxt(f"./result/{run_name}_{tag}_test/input{file_name[0]}.txt", input_data, fmt="%f")
            np.savetxt(f"./result/{run_name}_{tag}_test/real{file_name[0]}.txt", real, fmt="%f")
            np.savetxt(f"./result/{run_name}_{tag}_test/fake{file_name[0]}.txt", fake, fmt="%f")

    np.savetxt(f"./result/{run_name}_{tag}_test/mse.txt", mse_arr, fmt="%s", delimiter=",")
    print(f"Test loss: {test_loss / len(test_dataloader)}")


# 测试函数
def test_full_15000(model_path: str, run_name: str = None):
    if run_name is None:
        run_name = run_name_of(alpha, beta, gamma, delta, epsilon)
    model_dict = torch.load(model_path)
    # model = ConvAE(middle_channels, final_channels, kernal_size, stride).to(device)
    model = FNN(7, 4).to(device)

    model.load_state_dict(model_dict)
    model.eval()

    test_dataset = FNNTestDataset(root_dir, 'origin_new_15000')
    test_dataloader = DataLoader(test_dataset, 1, shuffle=True, num_workers=workers)

    loss_fn = nn.MSELoss().to(device)

    test_loss = 0.0

    os.makedirs(f"./result/{run_name}_15000_test", exist_ok=True)
    os.makedirs(f"./result/{run_name}_15000_convert", exist_ok=True)

    with torch.no_grad():

        mse_arr = []

        for inputs, real_outputs, file_name in tqdm(test_dataloader):
            
            # print(file_name[0])
            
            input_tensor = inputs.to(device)

            fake_img = model(input_tensor)
            # print(fake_img)
            fake = fake_img.cpu().numpy()[:, 0]

            inputs = inputs.to(device)
            real_outputs = real_outputs.to(device)
            # print(real_img.size())
            fake_img = model(inputs)

            loss = loss_fn(fake_img[0], real_outputs[0])
            test_loss += loss.item()
            mse_arr.append([file_name[0], loss.item()])
            # print([file_name[0], loss.item()])

            # 保存对比文件
            input_data = inputs.cpu().numpy()[0]
            # print(input_data)
            real = real_outputs.cpu().numpy()[0]
            # print(real)
            fake = fake_img.cpu().numpy()[0]
            np.savetxt(f"./result/{run_name}_15000_test/input{file_name[0]}.txt", input_data, fmt="%f")
            np.savetxt(f"./result/{run_name}_15000_test/real{file_name[0]}.txt", real, fmt="%f")
            np.savetxt(f"./result/{run_name}_15000_test/fake{file_name[0]}.txt", fake, fmt="%f")

    np.savetxt(f"./result/{run_name}_15000_test/mse.txt", mse_arr, fmt="%s", delimiter=",")
    print(f"Test loss: {test_loss / len(test_dataloader)}")


# 测试泛用性
def predict(model_path: str):
    model_dict = torch.load(model_path)
    # model = ConvAE(middle_channels, final_channels, kernal_size, stride).to(device)
    model = FNN(7, 4).to(device)

    model.load_state_dict(model_dict)
    model.eval()

    test_dataset = FNNDataset(root_dir, 'predict')
    test_dataloader = DataLoader(test_dataset, 1, shuffle=True, num_workers=workers)

    loss_fn = nn.MSELoss().to(device)

    test_loss = 0.0

    os.makedirs("./result/fnn_test", exist_ok=True)

    with torch.no_grad():
        for n, (real_img, real_output, file_name) in enumerate(test_dataloader, 0):
            real_img = real_img.to(device)
            real_output = real_output.to(device)
            # print(real_img.size())
            fake_img = model(real_img)

            loss = loss_fn(fake_img, real_output)
            test_loss += loss.item()

            # 保存对比文件
            input_data = real_img.cpu().numpy()
            real = real_output.cpu().numpy()
            fake = fake_img.cpu().numpy()
            np.savetxt(f"./result/fnn_test/input{file_name[0]}.txt", input_data, fmt="%f")
            np.savetxt(f"./result/fnn_test/real{file_name[0]}.txt", real, fmt="%f")
            np.savetxt(f"./result/fnn_test/fake{file_name[0]}.txt", fake, fmt="%f")

    print(f"Test loss: {test_loss / len(test_dataloader)}")


if __name__ == "__main__":
    # 一次传入多组权重，顺序完成各组的训练与验证，权重顺序：alpha,beta,gamma,delta,epsilon
    #   python train_pifnn.py --weights 1.0,1.0,1.0,1.0,1.0 1.0,1.0,1.0,1.0,1e-8
    # 加 --test 可在每组训练完成后自动执行 test_full 测试集评估（需存在 ./data/fnn/test）
    parser = argparse.ArgumentParser(description="PIFNN 多组权重训练")
    parser.add_argument("--weights", nargs="+", default=[f"{alpha},{beta},{gamma},{delta},{epsilon}"],
                        help="一组或多组权重，格式 alpha,beta,gamma,delta,epsilon，多组用空格分隔")
    parser.add_argument("--test", action="store_true",
                        help="每组训练完成后对该模型执行 test_full（需存在 ./data/fnn/test）")
    args = parser.parse_args()

    for w in args.weights:
        parts = w.split(",")
        if len(parts) != 5:
            raise ValueError(f"权重组格式错误：{w}，应为 alpha,beta,gamma,delta,epsilon 五个数")
        a, b, g, d, e = (float(v) for v in parts)
        run_name = run_name_of(a, b, g, d, e)
        print(f"===== 开始权重组 {run_name} =====")
        train(a, b, g, d, e)
        if args.test:
            if os.path.exists(f"{root_dir}/test"):
                test_full(f"./result/{run_name}.pth", run_name)
            else:
                print(f"跳过测试：{root_dir}/test 不存在")

    # 手动调用示例：
    # test(f"./result/{run_name}.pth", run_name)
    # test_full_20260607(f"./result/{run_name}.pth", run_name, "20260730")
    # test_full_15000(f"./result/{run_name}.pth", run_name)
    # predict("./result/nn.pth")
    # bc_test()
