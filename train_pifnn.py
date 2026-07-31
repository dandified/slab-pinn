import uuid
from tqdm import tqdm
from model import FNN
from dataset import FNNDataset, FNNTestDataset
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
workers = 2
nx = 111
nz = 101

# 时间间隔常量
time_constant = 1e13
# 质量守恒方程单独缩放因子（使质量损失量级与能量损失接近）
mass_scale = 1e24

# 自适应权重模式: 'gradnorm' | 'fixed' | 'dynamic'
adapt_mode = 'gradnorm'

# 初始权重（仅在非fixed模式下作为初始值）
# 注意：初始值不能为0，否则GradNorm会陷入零值陷阱
initial_alpha = 1.0
initial_beta = 1.0
initial_gamma = 1.0
initial_delta = 1.0

# GradNorm 参数
gradnorm_alpha = 0.1
gradnorm_update_freq = 100
gradnorm_min_weight = 1e-4
gradnorm_max_weight = 1e16


class AdaptiveWeights:
    def __init__(self, mode='gradnorm', initial_weights=None):
        self.mode = mode
        if initial_weights is None:
            initial_weights = [initial_alpha, initial_beta, initial_gamma, initial_delta]
        self.weights = torch.tensor(initial_weights, dtype=torch.float32, device=device)
        self.weights.requires_grad = False
        self.loss_history = []
        self.grad_history = []
        self.step_count = 0
    
    def get_weights(self):
        return self.weights.detach().cpu().numpy()
    
    def compute_grad_norm(self, loss, model, retain_graph=True):
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=retain_graph)
        grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads if g is not None))
        return grad_norm
    
    def update_gradnorm(self, loss_terms, model):
        if self.mode != 'gradnorm':
            return
        
        self.step_count += 1
        if self.step_count % gradnorm_update_freq != 0:
            return
        
        num_terms = len(loss_terms)
        grad_norms = []
        
        for i, loss_term in enumerate(loss_terms):
            if loss_term.item() > 0:
                grad_norm = self.compute_grad_norm(loss_term, model)
            else:
                grad_norm = torch.tensor(0.0, device=device)
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        avg_grad_norm = grad_norms.mean()
        
        grad_norm_threshold = 1e-20
        new_weights = []
        for i in range(num_terms):
            if grad_norms[i] > grad_norm_threshold:
                ratio = avg_grad_norm / grad_norms[i]
                new_w = self.weights[i] * (ratio ** gradnorm_alpha)
            else:
                new_w = self.weights[i] * 10.0
            new_w = torch.clamp(new_w, gradnorm_min_weight, gradnorm_max_weight)
            new_weights.append(new_w)
        
        self.weights = torch.tensor(new_weights, dtype=torch.float32, device=device)
        self.weights.requires_grad = False
    
    def update_dynamic(self, loss_terms):
        if self.mode != 'dynamic':
            return
        
        loss_values = torch.tensor([lt.item() for lt in loss_terms], device=device)
        loss_sum = loss_values.sum()
        
        if loss_sum > 0:
            self.weights = loss_sum / (len(loss_terms) * loss_values)
            self.weights = torch.clamp(self.weights, gradnorm_min_weight, gradnorm_max_weight)
    
    def update(self, loss_terms, model=None):
        if self.mode == 'gradnorm':
            self.update_gradnorm(loss_terms, model)
        elif self.mode == 'dynamic':
            self.update_dynamic(loss_terms)


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
def pde_loss(input_, output_):
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

    # pde_l = alpha * torch.mean(mass_formula ** 2) + beta * torch.mean(energy_residual ** 2)

    # print(f"PDE 损失：{pde_loss.item()}")

    # 返回均方误差（质量守恒使用单独的缩放因子）
    return torch.mean(mass_formula ** 2) * time_constant * mass_scale, torch.mean(energy_residual ** 2) * time_constant


# 训练函数入口
def train():
    # 超参数

    # 训练参数
    epoch = 5
    learning_rate = 5e-4
    minimum_validation_loss = 10000000000.0
    batch_size = 128

    # 初始化自适应权重
    adapt_weights = AdaptiveWeights(mode=adapt_mode)

    # 加载数据
    # model_path = f"./result/conv_ae_{middle_channels}_{final_channels}.pth"
    model_path = f"./result/pifnn_{adapt_mode}.pth"

    # 数据集
    train_dataset = FNNDataset(root_dir, 'train', output_index_end=3)
    valid_dataset = FNNDataset(root_dir, 'valid', output_index_end=3)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=workers)

    loss_fn = nn.MSELoss().to(device)

    # 定义FNN网络
    model = FNN(7, 3).to(device)

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
    weights_history = []

    try:
        # 训练循环
        for i in range(epoch):
            # 训练
            training_loss = 0.0
            step = 0
            for inputs, real_outputs, _ in train_dataloader:
                inputs = inputs.to(device)
                inputs.requires_grad = True
                real_outputs = real_outputs.to(device)
                
                model.zero_grad()
                outputs = model(inputs)

                # 添加初始条件约束，令t=0
                ic_real = ic_loss(inputs)
                ic_inputs = inputs.clone()
                ic_inputs[:, 2] = 0.0
                ic_outputs = model(ic_inputs)[:, 0]
                
                # 添加边界条件约束，令z=0和z=600的温度为0和1项设为0
                bc_inputs_top = inputs.clone()
                bc_inputs_top[:, 1] = 0.0
                bc_inputs_bottom = inputs.clone()
                bc_inputs_bottom[:, 1] = 600.0
                bc_outputs_top = model(bc_inputs_top)[:, 0]
                bc_outputs_bottom = model(bc_inputs_bottom)[:, 0]

                bc_real_top = torch.zeros(bc_outputs_top.size()).to(device)
                bc_real_bottom = torch.ones(bc_outputs_bottom.size()).to(device)

                # 计算各项损失（不带权重）
                m_base, e_base = pde_loss(inputs, outputs)
                print(f"  Mass Loss: {m_base.item()}")
                print(f"  Energy Loss: {e_base.item()}")
                icLoss_base = loss_fn(ic_outputs, ic_real)
                bcLoss_base = loss_fn(bc_outputs_top, bc_real_top) + loss_fn(bc_outputs_bottom, bc_real_bottom)
                print(f"  IC Loss: {icLoss_base.item()}")
                print(f"  BC Loss: {bcLoss_base.item()}")
                data_loss = loss_fn(outputs, real_outputs)

                # 获取当前权重
                w = adapt_weights.weights
                alpha_w, beta_w, gamma_w, delta_w = w[0], w[1], w[2], w[3]

                # 带权重的损失（用于计算梯度范数）
                m_weighted = m_base * alpha_w
                e_weighted = e_base * beta_w
                icLoss_weighted = icLoss_base * gamma_w
                bcLoss_weighted = bcLoss_base * delta_w

                # 更新自适应权重（在 backward 之前）
                if adapt_mode != 'fixed':
                    loss_terms = [m_base, e_base, icLoss_base, bcLoss_base]
                    adapt_weights.update(loss_terms, model)
                    w = adapt_weights.weights
                    alpha_w, beta_w, gamma_w, delta_w = w[0], w[1], w[2], w[3]
                    m_weighted = m_base * alpha_w
                    e_weighted = e_base * beta_w
                    icLoss_weighted = icLoss_base * gamma_w
                    bcLoss_weighted = bcLoss_base * delta_w

                # 总损失
                loss = data_loss + m_weighted + e_weighted + icLoss_weighted + bcLoss_weighted

                loss.backward()
                optim.step()
                scheduler_lr.step()

                # debug 信息
                total_train_step = total_train_step + 1
                training_loss += loss.item()
                step += 1

                print(f"已完成{step}/{len(train_dataloader)}训练，损失：{loss.item():.6f}")
                print(f"  Data Loss: {data_loss.item():.6f}")
                print(f"  Mass Loss: {m_weighted.item():.6f} (w={alpha_w.item():.6f})")
                print(f"  Energy Loss: {e_weighted.item():.6f} (w={beta_w.item():.6f})")
                print(f"  IC Loss: {icLoss_weighted.item():.6f} (w={gamma_w.item():.6f})")
                print(f"  BC Loss: {bcLoss_weighted.item():.6f} (w={delta_w.item():.6f})")
                training_loss_list.append(loss.item())
                weights_history.append(adapt_weights.get_weights().tolist())

            # 验证
            with torch.no_grad():
                validation_loss = 0.0

                step = 0
                for valids, valid_outputs, _ in valid_dataloader:
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

                torch.save(model.state_dict(), f"./result/pifnn_{adapt_mode}_{i}.pth")

        # Training finished, print the statistics for the best model
        print('Training Finished')
        torch.save(model.state_dict(), f"./result/pifnn_{adapt_mode}_final.pth")

        print("Best model has a validation loss of {} and it's in epoch {}".format(minimum_validation_loss,
                                                                                   best_model_index + 1))

        train_file = open(f'./result/pifnn_{adapt_mode}_trainingData.txt', "w")
        train_file.write(",".join([str(elem) for elem in training_loss_list]))
        train_file.close()

        valid_file = open(f'./result/pifnn_{adapt_mode}_validationData.txt', "w")
        valid_file.write(",".join([str(elem) for elem in validation_loss_list]))
        valid_file.close()

        if adapt_mode != 'fixed':
            weights_file = open(f'./result/pifnn_{adapt_mode}_weightsData.txt', "w")
            for w in weights_history:
                weights_file.write(",".join([str(elem) for elem in w]) + "\n")
            weights_file.close()
            print("Weights History saved!")

        print("Training Data saved!")

    except Exception as e:
        print(traceback.format_exc())
        print(f"训练出错：{e}")

        train_file = open(f'./result/pifnn_{adapt_mode}_trainingData.txt', "w")
        train_file.write(",".join([str(elem) for elem in training_loss_list]))
        train_file.close()

        valid_file = open(f'./result/pifnn_{adapt_mode}_validationData.txt', "w")
        valid_file.write(",".join([str(elem) for elem in validation_loss_list]))
        valid_file.close()

        if adapt_mode != 'fixed' and weights_history:
            weights_file = open(f'./result/pifnn_{adapt_mode}_weightsData.txt', "w")
            for w in weights_history:
                weights_file.write(",".join([str(elem) for elem in w]) + "\n")
            weights_file.close()

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
def test(model_path: str):
    model_dict = torch.load(model_path)
    # model = ConvAE(middle_channels, final_channels, kernal_size, stride).to(device)
    model = FNN(7, 3).to(device)

    model.load_state_dict(model_dict)
    model.eval()

    # test_dataset = FNNDataset(root_dir, 'test')
    # test_dataloader = DataLoader(test_dataset, 1, shuffle=True, num_workers=workers)

    loss_fn = nn.MSELoss().to(device)

    test_loss = 0.0

    os.makedirs(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_test", exist_ok=True)
    os.makedirs(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_convert", exist_ok=True)

    with torch.no_grad():
        # 创建行和列的网格索引
        arr = []
        for x in range(0, 666, 6):
            for z in range(0, 606, 6):
                arr.append([x, 600 - z, 217.5, 60.0, 45.0, 30.0, 3.17 * 1.0])

        input_tensor = torch.from_numpy(np.array(arr, dtype=float)).float().to(device)

        fake_img = model(input_tensor)
        print(fake_img)
        fake = fake_img.cpu().numpy()[:, 0]

        target_name = "TempHeat0645.0Age02Vel01.11350"

        # print(fake)
        np.savetxt(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_test/fake{target_name}.txt", [fake], fmt="%f", delimiter=" ")

        # 读取文件
        with open(f"{root_dir}/valid/output/{target_name}", 'r') as f:
            content = f.read().strip()

        # 替换逗号为换行
        processed_content = content.replace(',', '\n')

        # 写入临时文件
        with open('temp_data.txt', 'w') as f:
            f.write(processed_content)

        # 加载真实数据
        real_img = np.load(f"{root_dir}/valid/output/{target_name}")[:, 0]
        print(real_img)

        # 清理临时文件
        os.remove('temp_data.txt')


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
def test_full(model_path: str):
    model_dict = torch.load(model_path)
    # model = ConvAE(middle_channels, final_channels, kernal_size, stride).to(device)
    model = FNN(7, 3).to(device)

    model.load_state_dict(model_dict)
    model.eval()

    test_dataset = FNNTestDataset(root_dir, 'test')
    test_dataloader = DataLoader(test_dataset, 1, shuffle=True, num_workers=workers)

    loss_fn = nn.MSELoss().to(device)

    test_loss = 0.0

    os.makedirs(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_test", exist_ok=True)
    os.makedirs(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_convert", exist_ok=True)

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
            print([file_name[0], loss.item()])

            # 保存对比文件
            input_data = inputs.cpu().numpy()[0]
            real = real_outputs.cpu().numpy()[0]
            fake = fake_img.cpu().numpy()[0]
            np.savetxt(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_test/input{file_name[0]}.txt", input_data, fmt="%f")
            np.savetxt(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_test/real{file_name[0]}.txt", real, fmt="%f")
            np.savetxt(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_test/fake{file_name[0]}.txt", fake, fmt="%f")

    np.savetxt(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_test/mse.txt", mse_arr, fmt="%s", delimiter=",")
    print(f"Test loss: {test_loss / len(test_dataloader)}")


# 测试泛用性
def predict(model_path: str):
    model_dict = torch.load(model_path)
    # model = ConvAE(middle_channels, final_channels, kernal_size, stride).to(device)
    model = FNN(7, 3).to(device)

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
    train()
    # test(f"./result/pifnn_{alpha}_{beta}_{gamma}_{delta}_final.pth")
    # vpredict("./result/nn.pth")
    # bc_test()
