import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

from net import Discriminator_conv, Generator_conv, MatchFilter

import matplotlib.pyplot as plt

import os


# def load_model(model, model_dict_path):
#     """
#     多GPU训练时,加载单GPU训练的模型
#     :param model: 模型
#     :param model_dict_path: 模型参数路径
#     """
#     from collections import OrderedDict
#     loaded_dict = torch.load(model_dict_path) # 加载模型参数
#     new_state_dict = OrderedDict() # 新建一个空的字典
#     for k, v in loaded_dict.items():
#         name = "module." + k[:]  # 添加'module.'是为了适应多GPU训练时保存的模型参数的key值
#         new_state_dict[name] = v # 新字典的key值对应的value一一对应

#     model.load_state_dict(new_state_dict) # 加载模型参数

#     return model

#参数
classnum = 4 # 类别数
PATH = './net_save/50/netG_epoch_6000.pth' # 模型路径

# 加载模型
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")  # 创建 device 对象
loaded_dict = torch.load(PATH, map_location=device)  # 指定加载到 device
netG = Generator_conv(4,100,100,2048,use_normalize=1).to(device)  # 初始化模型并指定设备
netG = nn.DataParallel(netG,device_ids=[5])  # 往 model 里面添加 module
netG.load_state_dict(loaded_dict)  # 加载模型参数
# 一定要开启评估模式
netG.eval() # netG评估模式

#打印netG的dict
#print(netG.state_dict().keys())


a = torch.tensor([3,1]).long().to(device) # 标签信息
z = torch.randn(2,100).to(device) # 随机噪声
fake_sign = netG(z,a) # 生成fake_sign

result = fake_sign[0].cpu().detach().numpy() # 查看第一个结果

plt.plot(result)
plt.savefig('./result_M.png')
print('打印结果')
print(a)



# 匹配滤波
T=10e-6;                           #脉冲宽度 10us
B=10e6;                            #带宽
C=3e8;                                 # propagation speed
K=B/T;                                 #chirp slope
Fs=5*B
Ts=1/Fs;                         #sampling frequency and sampling spacing

N_fft = 2000;         # fft点数

# 生成LFM信号
t = np.arange(0, T,Ts)
s = np.exp(1j * np.pi * K * t**2)
# 画图 画s的实部
# plt.plot(t,s.real)
# plt.show()

# 生成LFM信号的频谱
s_fft = np.fft.fft(s,N_fft)
# 画图 画s_fft的实部
t1= np.arange(0, N_fft)
# plt.plot(t1,abs(s_fft))
# plt.show()

# s的共轭序列
s1 = np.conj(s[::-1])
# 生成匹配滤波器的频谱
s1_fft = np.fft.fft(s1,N_fft)
# 画图 画s1_fft的实部
# plt.plot(t1,abs(s1_fft))
# plt.show()

# 生成匹配滤波器
s2 = np.fft.ifft(s1_fft*s_fft)

r_fft = np.fft.fft(result,N_fft)
sr = np.fft.ifft(s1_fft*r_fft)
# 画图 画sr的实部
plt.clf()
plt.plot(t1,abs(sr.real))
plt.savefig('./match_filter.png')

plt.clf()
# 画出r的频谱
# 对r_fft进行fftshift
r_fft = np.fft.fftshift(r_fft)
plt.plot(t1,abs(r_fft))
plt.savefig('./频谱.png')