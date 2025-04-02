import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from matplotlib.font_manager import FontProperties

def calculate_mse_gpu(mu1, mu2):
    """
    计算 GPU 加速的 Mean Squared Error (MSE)
    :param mu1: 真实数据的均值 (1x2048) (torch.Tensor)
    :param mu2: 生成数据的均值 (1x2048) (torch.Tensor)
    :return: MSE 分数
    """
    device = mu1.device  # 获取当前设备（CPU / GPU）
    
    mse = torch.mean((mu1 - mu2) ** 2)
    
    # 返回tensor的数值
    return mse.cpu().item()

def calculate_fid_gpu(mu1, sigma1, mu2, sigma2):
    """
    计算 GPU 加速的 Fréchet Inception Distance (FID)
    :param mu1: 真实数据的均值 (1x2048) (torch.Tensor)
    :param sigma1: 真实数据的协方差矩阵 (2048x2048) (torch.Tensor)
    :param mu2: 生成数据的均值 (1x2048) (torch.Tensor)
    :param sigma2: 生成数据的协方差矩阵 (2048x2048) (torch.Tensor)
    :return: FID 分数
    """
    device = mu1.device  # 获取当前设备（CPU / GPU）
    
    diff = mu1 - mu2
    covmean = torch.from_numpy(scipy.linalg.sqrtm((sigma1 @ sigma2).cpu().numpy())).real.to(device)

    fid = torch.dot(diff, diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.cpu().item()  # 返回 Python 数值

def calculate_kid_gpu(X, Y, gamma=0.001):
    """
    计算 GPU 加速的 Kernel Inception Distance (KID)
    :param X: 真实数据特征 (1x2048) (torch.Tensor)
    :param Y: 生成数据特征 (1x2048) (torch.Tensor)
    :param gamma: RBF 核的 gamma 参数 (控制尺度)
    :return: KID 分数
    """
    device = X.device  # 获取当前设备
    
    K_XX = torch.exp(-gamma * torch.cdist(X, X) ** 2)
    K_YY = torch.exp(-gamma * torch.cdist(Y, Y) ** 2)
    K_XY = torch.exp(-gamma * torch.cdist(X, Y) ** 2)
    
    kid = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return kid.cpu().item()  # 返回 Python 数值

def calculate_kid_gpu2(X, Y, gamma=None, device="cuda"):
    """
    计算 Kernel Inception Distance (KID)
    :param X: 真实数据特征 (6000, 2048) (torch.Tensor)
    :param Y: 生成数据特征 (6000, 2048) (torch.Tensor)
    :param gamma: RBF 核的 gamma 参数，若 None 则自动计算
    :param device: 计算设备 ("cuda" 或 "cpu")
    :return: KID 分数 (float)
    """
    X, Y = X.to(device), Y.to(device)
    
    # 计算 gamma（如果未提供）
    if gamma is None:
        gamma = 1.0 / (2 * X.var())

    # 计算欧氏距离
    XX_dist = torch.cdist(X, X) ** 2
    YY_dist = torch.cdist(Y, Y) ** 2
    XY_dist = torch.cdist(X, Y) ** 2

    # 计算 RBF 核
    K_XX = torch.exp(-gamma * XX_dist)
    K_YY = torch.exp(-gamma * YY_dist)
    K_XY = torch.exp(-gamma * XY_dist)

    # 计算 KID
    kid = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

    return kid.cpu().item()  # 转换为 Python float

def polynomial_kernel(X, Y, degree=3, coef0=1):
    """
    计算多项式核矩阵
    :param X: 真实数据特征 (N, D)
    :param Y: 生成数据特征 (M, D)
    :param degree: 多项式的次数
    :param coef0: 核的偏置项
    :return: (N, M) 形状的核矩阵
    """
    return (X @ Y.T + coef0) ** degree

def compute_kid_polynomial(X, Y, degree=3, coef0=1, device="cuda"):
    """
    使用多项式核计算 Kernel Inception Distance (KID)
    :param X: 真实数据特征 (N, D) (torch.Tensor)
    :param Y: 生成数据特征 (M, D) (torch.Tensor)
    :param degree: 多项式核的次数
    :param coef0: 偏置项
    :param device: 计算设备 ("cuda" 或 "cpu")
    :return: KID 分数 (float)
    """
    X, Y = X.to(device), Y.to(device)

    # 计算 KID 相关的核矩阵
    K_XX = polynomial_kernel(X, X, degree, coef0)
    K_YY = polynomial_kernel(Y, Y, degree, coef0)
    K_XY = polynomial_kernel(X, Y, degree, coef0)

    # 计算 KID
    kid = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

    return kid.cpu().item()  # 转换为 float

def add_salt_and_pepper_noise_torch(data, noise_ratio=0.05, min_val=-1, max_val=1):
    """
    在 PyTorch 张量中添加椒盐噪声
    :param data: 原始数据 (PyTorch Tensor)
    :param noise_ratio: 噪声比例 (0~1)
    :param min_val: 椒盐噪声中的最小值
    :param max_val: 椒盐噪声中的最大值
    :return: 添加噪声后的数据 (Tensor)
    """
    noisy_data = data  # 直接修改原数据
    num_noisy_points = int(len(data) * noise_ratio)  # 计算需要加噪声的点数
    noise_indices = torch.randperm(len(data))[:num_noisy_points]  # 生成随机索引
    
    # 选一半替换为最小值，另一半替换为最大值
    half = len(noise_indices) // 2
    noisy_data[noise_indices[:half]] = min_val  # 椒（黑点）
    noisy_data[noise_indices[half:]] = max_val  # 盐（白点）
    
    return noisy_data


def add_noise_plot(noise_type, noise_rate):

    # 设置matplotlib正常显示中文和负号  调制为宋体五号
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    font_properties = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=35)  # 10号字体对应五号字体

    noise_rate = torch.tensor(noise_rate).cuda()

    sign_name = 'LFM'
    DataRoot = 'D://DeepLearning//Radar//RealData//'
    

    #读取npy文件
    npy_path = DataRoot + sign_name
    # fake_npy_path = './GenerateData/'+ fake_name  # diffusion_result GenerateData

    # 读取6000个real_sign和fake_sign拼接成两个矩阵
    real_sign_matrix = torch.zeros((6000, 2048)).cuda()
    fake_sign_matrix = torch.zeros((6000, 2048)).cuda()
    # 用于保存每一步加噪后的数据，使得噪声可以逐步累加
    fake_add_noise = torch.zeros((6000, 2048)).cuda()


    for i in range(6000):
        real_sign = np.load(os.path.normpath(os.path.join(npy_path,'real_sign_{}.npy'.format(i))))

        # fake_sign = np.load(os.path.normpath(os.path.join(fake_npy_path,'fake_sign_{}.npy'.format(i))))
        
        #将数据转换为tensor同时放到GPU上
        real_sign = torch.tensor(real_sign).cuda()
        real_sign_matrix[i] = real_sign.clone()
        fake_add_noise[i] = real_sign.clone()
        




    # 计算均值
    real_sign_matrix_mean = torch.mean(real_sign_matrix, axis=0).cuda()
    

    # 计算方差
    real_sign_matrix_var = torch.var(real_sign_matrix, axis=0).cuda()

    #=============================================FID=============================================
    # 计算real_sign和fake_sign的协方差矩阵
    real_sign_matrix_cov = torch.cov(real_sign_matrix.T).cuda()



    # noise_rate 的长度作为循环次数
    num = len(noise_rate)
    # 创建noise_sign 来保存num个加噪后的数据
    noise_sign = torch.zeros((num, 2048)).cuda()

    # 创建三个空torch数组
    mse = []
    fid = []
    kid = []

    for j in range(num):

        for i in range(6000):
            # 高斯噪声
            if noise_type == 0:

                # 加噪声
                noise = 10 * torch.normal(0, 0.1, (2048,)).cuda()
                # if i==0 : print('噪声', noise)
                fake_sign = fake_add_noise[i].clone() 
                fake_sign = fake_sign + noise_rate[j] * noise # 加噪声
                # fake_sign 归一化每一行的数据
                fake_sign = fake_sign / torch.max(torch.abs(fake_sign))

            # 椒盐噪声
            if noise_type == 1:

                # 加噪声
                fake_sign = add_salt_and_pepper_noise_torch(fake_add_noise[i], noise_ratio=noise_rate[j]*0.5)


            # 随机擦除
            if noise_type == 2:

                # 随及长度的信号归零 
                begin = 0 if noise_rate[j]==1 else torch.randint(0, 2048-int(noise_rate[j]*2048), (1,)).item()
                # 这里的让fake_sign 指向fake_add_noise[i]
                fake_sign = fake_add_noise[i]
                # 对fake_sign加噪声后fake_add_noise的数据也会同样改变，使得后续噪声可以在当前基础上累加
                fake_sign[begin:begin+int(noise_rate[j]*2048)] = 0



            # fake_sign_matrix保存当前加噪后的数据
            fake_sign_matrix[i] = fake_sign.clone()

            #画出第一张图像
            if i==0:
                noise_sign[j] = fake_sign.clone() # 保存用于画图
                plt.figure()
                # 横坐标
                plt.xlabel('时间(s)', fontproperties=font_properties)
                # 纵坐标
                plt.ylabel('幅值(V)', fontproperties=font_properties)
                plt.plot(fake_sign.cpu())
                plt.savefig('第{}张图像.png'.format(j), dpi=300)
                #清除图像
                plt.clf()
            
        
        # 计算均值
        fake_sign_matrix_mean = torch.mean(fake_sign_matrix, axis=0)
        # 计算方差
        fake_sign_matrix_var = torch.var(fake_sign_matrix, axis=0)
        # 计算协方差矩阵
        fake_sign_matrix_cov = torch.cov(fake_sign_matrix.T)

        # 计算MSE
        MSE = calculate_mse_gpu(real_sign_matrix_mean, fake_sign_matrix_mean)
        print('MSE:',MSE)
        mse.append(MSE)

        # 计算FID
        FID = calculate_fid_gpu(real_sign_matrix_mean, real_sign_matrix_cov, fake_sign_matrix_mean, fake_sign_matrix_cov)
        print('FRD:',FID)
        fid.append(FID)

        #计算KID
        # KID = calculate_kid_gpu(real_sign_matrix, fake_sign_matrix)
        KID = compute_kid_polynomial(real_sign_matrix, fake_sign_matrix) # 使用多项式核计算 KID
        print('KRD:',KID)
        kid.append(KID)
    
    # noise_rate搬回cpu
    noise_rate = torch.tensor(noise_rate).cpu().numpy()

    # 将mse, fid, kid搬回到cpu上
    mse = np.array(mse)
    fid = np.array(fid)
    kid = np.array(kid)

    # 画噪声图
    fig, axs = plt.subplots(1, num, figsize=(11*num, 10))# 创建一个图形窗口

    # 总标题
    # fig.suptitle('添加高斯噪声过程', fontproperties=font_properties)

    for i in range(num):
        axs[i].plot(noise_sign[i].cpu())
        axs[i].set_title('噪声比例：{:.1f}'.format(noise_rate[i]), fontproperties=font_properties)
        axs[i].set_xlabel('时间(s)', fontproperties=font_properties)
        axs[i].set_ylabel('幅值(V)', fontproperties=font_properties)
        axs[i].tick_params(axis='both', which='major', labelsize=15)
    plt.savefig('随机擦除过程.png', dpi=300)



    






    

    # 归一化三个数据
    # mse = [i/max(mse) for i in mse]
    # fid = [i/max(fid) for i in fid]
    # kid = [i/max(kid) for i in kid]

    # mse = [i*10000 for i in mse]
    # fid = [i/100 for i in fid]
    # kid = [i/10000 for i in kid]





    # 三个数据的折线图 画在同一行

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))# 创建一个图形窗口

    # 总标题
    fig.suptitle('随机擦除实验评分', fontproperties=font_properties)

    


    # MSE
    axs[0].plot(noise_rate, mse, 'r-', marker='s', label='MSE', linewidth=4, markersize=18)  # 设置线条宽度
    axs[0].set_xlabel('噪声比例', font_properties=font_properties, fontweight='bold')  # 设置字体加粗
    axs[0].set_ylabel('MSE', font_properties=font_properties, fontweight='bold')  # 设置字体加粗
    axs[0].tick_params(axis='both', which='major', labelsize=20)  # 设置x轴和y轴刻度字体大小
    axs[0].legend(loc='upper left', fontsize=20)  # 设置图例字体加粗

    # FID
    axs[1].plot(noise_rate, fid, 'g-', marker='*', label='FRD', linewidth=4, markersize=18)  # 设置线条宽度
    axs[1].set_xlabel('噪声比例', font_properties=font_properties, fontweight='bold')  # 设置字体加粗
    axs[1].set_ylabel('FRD', font_properties=font_properties, fontweight='bold')  # 设置字体加粗
    axs[1].tick_params(axis='both', which='major', labelsize=20)  # 设置x轴和y轴刻度字体大小
    axs[1].legend(loc='upper left', fontsize=20)  # 设置图例字体加粗

    # KID
    axs[2].plot(noise_rate, kid, 'b-', marker='^', label='KRD', linewidth=4, markersize=18)  # 设置线条宽度
    axs[2].set_xlabel('噪声比例', font_properties=font_properties, fontweight='bold')  # 设置字体加粗
    axs[2].set_ylabel('KRD', font_properties=font_properties, fontweight='bold')  # 设置字体加粗
    axs[2].tick_params(axis='both', which='major', labelsize=20)  # 设置x轴和y轴刻度字体大小
    axs[2].legend(loc='upper left', fontsize=20)  # 设置图例字体加粗

    plt.savefig('随机擦除实验结果.png', dpi=300)



    # 单独画KID的图
    # font_properties_2 = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=15)  # 10号字体对应五号字体
    # plt.figure()
    # plt.title('使用高斯核计算KRD', fontproperties=font_properties_2)  # 设置字体加粗
    # plt.plot(noise_rate, kid, 'b-', marker='^', label='KRD', linewidth=2, markersize=5)  # 设置线条宽度
    # plt.xlabel('噪声比例', font_properties=font_properties_2,)  # 设置字体加粗
    # plt.ylabel('KRD', font_properties=font_properties_2)  # 设置字体加粗
    # plt.tick_params(axis='both', which='major', labelsize=8)  # 设置x轴和y轴刻度字体大小
    # plt.legend(loc='upper left', fontsize=5)  # 设置图例字体加粗
    # plt.savefig('KID.png', dpi=300)

    font_properties_2 = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=15)  # 10号字体对应五号字体
    plt.figure()
    plt.title('使用多项式核计算KRD', fontproperties=font_properties_2)  # 设置字体加粗
    plt.plot(noise_rate, kid, 'b-', marker='^', label='KRD', linewidth=2, markersize=5)  # 设置线条宽度
    plt.xlabel('噪声比例', font_properties=font_properties_2,)  # 设置字体加粗
    plt.ylabel('KRD', font_properties=font_properties_2)  # 设置字体加粗
    plt.tick_params(axis='both', which='major', labelsize=8)  # 设置x轴和y轴刻度字体大小
    plt.legend(loc='upper left', fontsize=5)  # 设置图例字体加粗
    plt.savefig('KRD.png', dpi=300)



# 计算生成数据的MSE、FID、KID
def evalution(real_sign_root, fake_sign_root):
    # 读取npy文件
    real_sign_matrix = torch.zeros((6000, 2048)).cuda()
    fake_sign_matrix = torch.zeros((6000, 2048)).cuda()

    # 四种信号的名字
    sign_name = ['LFM', 'AM_noise', 'FM_noise', 'ISRJ']

    # 保存四种信号的MSE、FID、KID
    mse = []
    fid = []
    kid = []

    for i in range(4):
        # 读取real_sign
        real_sign_path = real_sign_root + sign_name[i]
        for j in range(6000):
            real_sign = np.load(os.path.normpath(os.path.join(real_sign_path, 'real_sign_{}.npy'.format(j))))
            real_sign = torch.tensor(real_sign).cuda()
            real_sign_matrix[j] = real_sign.clone()

        # 读取fake_sign
        fake_sign_path = fake_sign_root + sign_name[i]
        for j in range(6000):
            fake_sign = np.load(os.path.normpath(os.path.join(fake_sign_path, 'fake_sign_{}.npy'.format(j))))
            fake_sign = torch.tensor(fake_sign).cuda()
            fake_sign_matrix[j] = fake_sign.clone()

        # 计算均值
        real_sign_matrix_mean = torch.mean(real_sign_matrix, axis=0).cuda()
        fake_sign_matrix_mean = torch.mean(fake_sign_matrix, axis=0).cuda()

        # 计算协方差矩阵
        real_sign_matrix_cov = torch.cov(real_sign_matrix.T).cuda()
        fake_sign_matrix_cov = torch.cov(fake_sign_matrix.T).cuda()

        # 计算MSE
        MSE = calculate_mse_gpu(real_sign_matrix_mean, fake_sign_matrix_mean)
        mse.append(MSE)

        # 计算FID
        FID = calculate_fid_gpu(real_sign_matrix_mean, real_sign_matrix_cov, fake_sign_matrix_mean, fake_sign_matrix_cov)
        fid.append(FID)

        # 计算KID
        KID = compute_kid_polynomial(real_sign_matrix, fake_sign_matrix)
        kid.append(KID) 
    
    return mse, fid, kid

if __name__ == "__main__":

    # noise_rate = [0, 0.1, 0.2,0.3,0.5,0.7,0.9, 1]
    noise_rate = [0, 0.2,0.5,0.7, 1]
    noise_type = 0 # 0:高斯噪声 1:椒盐噪声 2:随机擦除
    add_noise_plot(noise_type, noise_rate)

    # real_sign_root = 'D://DeepLearning//Radar//RealData//'
    # fake_sign_root = 'D://DeepLearning//Radar//GenerateData//'
    # mse, fid, kid = evalution(real_sign_root, fake_sign_root)
    # print('MSE:', mse)
    # print('FID:', fid)
    # print('KID:', kid)





    