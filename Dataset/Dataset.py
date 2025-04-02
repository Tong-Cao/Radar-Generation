#pytorch制作.mat数据集
import scipy.io as sio
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import os
import matplotlib.pyplot as plt


# 创建数据集 文件格式为./data/class1/1.mat 将data下所有文件夹下的.mat文件读取
class MyDataset(Dataset):
    def __init__(self, root, part,transform=None, target_transform=None):
        '''
        root: 数据集路径 数据文件夹格式为root/class1/1.mat
        part: 数据部分,realsp,imagsp,complexsp  (实部,虚部,IQ复数)
        '''
        super(MyDataset, self).__init__()
        self.root = root
        self.part = part
        self.transform = transform
        self.target_transform = target_transform

        self.data_path = [] # 存储所有数据路径
        # 遍历root下所有子文件夹再将所有子文件夹下的.mat文件路径存储
        for file in os.listdir(root):
            path = os.path.join(root, file)
            if os.path.isdir(path):
                for file in os.listdir(path):
                    if file.endswith('.mat'):
                        self.data_path.append(os.path.join(path, file))

        
        self.class_idx = self.find_classes() # 获取类别名称和对应的索引

    def find_classes(self):
        '''
        将path下的文件夹名称作为类别名称
        '''
        path = self.root # 数据集路径
        classes = [d.name for d in os.scandir(path) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return class_to_idx # 返回类别名称和对应的索引{'class1': 0, 'class2': 1, 'class3': 2, 'class4': 3'} 

    def __getitem__(self, index):
        d_path = self.data_path[index]
        data = sio.loadmat(d_path)[self.part] # 读取mat文件中的realsp或者其他部分
        data = data.reshape(-1) # 将数据转化为一维
        data = torch.from_numpy(data) # 将numpy转化为tensor

        d_path = d_path.replace('\\', '/') # 将路径上的'\\'替换为'/' （windows中路径分隔符为'\'，linux中为'/'）

        label_name = d_path.split('/')[-2] # 获取d_path倒数第二个字符串作为label os.path.join()拼接时用的为"/"
        label = self.class_idx[label_name] # 获取label对应的索引

        return data, label

    def __len__(self):
        return len(self.data_path)



if __name__ == "__main__":
    root = './data'

    mydataset = MyDataset(root=root, part='realsp')  # 实例化dataset
    train_iter = DataLoader(dataset=mydataset, batch_size=10,
                            drop_last=True,shuffle=True
                            ) 
  
    X, y = next(iter(train_iter))  # next(iter())迭代器 取出一条batch

    #打印X的数据范围
    print('X的数据范围')
    print(X.max())
    print(X.min())
    

    print(y)
    # 画出第一个batch的第一个数据
    plt.plot(X[0])
    plt.savefig('ganrao.png')

    # 匹配滤波器
    def filter():
        T=10e-6;                           #脉冲宽度 10us
        B=10e6;                            #带宽
        C=3e8;                                 #propagation speed
        K=B/T;                                 #chirp slope
        Fs=5*B
        Ts=1/Fs;                         #sampling frequency and sampling spacing

        N_fft = 2048;         # fft点数 需要和生成器输出维度一致
        # 生成LFM信号
        t = np.arange(0, T,Ts)
        s = np.exp(1j * np.pi * K * t**2)

        # 生成LFM信号的频谱
        s_fft = np.fft.fft(s,N_fft)

        # s的共轭序列
        s1 = np.conj(s[::-1])

        # 生成匹配滤波器的频谱
        s1_fft = np.fft.fft(s1,N_fft)

        return s1_fft

    filter_fft = filter()

    # 匹配滤波
    def match_filter(signl,filter_fft):
        # signl的频谱
        r_fft = np.fft.fft(signl,2048) # 2048为fft点数
        # 匹配滤波
        sr = np.fft.ifft(filter_fft*r_fft)
        return sr
    
    



    X = X.cpu().detach().numpy()
    sr = match_filter(X,filter_fft)
    sr = torch.from_numpy(sr).to(torch.float32)
    print(sr.shape)
    plt.clf()
    plt.plot(sr[0])
    plt.savefig('result.png')


    # for i in range(10):
        
    #     plt.plot(X[i])
    #     plt.show()

    # embedding = torch.nn.Embedding(4, 100)
    # embd = embedding(y)
    # print('y',y)
    # print('embd后的数据',embd)

    # a = torch.cat((X,embd),dim = -1)
    # print('a',a.shape)

    #print(output.shape)







    # for i, (data, label) in enumerate(train_iter):
    #     if (i+1)%50==0:
    #         print(i)
    #         print(data)
    #         print(label)








