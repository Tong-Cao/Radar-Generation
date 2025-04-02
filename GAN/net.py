import torch
import torch.nn as nn
import numpy as np

# # 创建1D反卷积网络
# # L_out=(L_in−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
# class R_Net(torch.nn.Module):
#     def __init__(self):
#         super(R_Net, self).__init__()
#         self.conv1 = torch.nn.Sequential( # 输入1通道,输出16通道,卷积核大小为5,步长为1,填充为2
#             torch.nn.ConvTranspose1d(1, 16, kernel_size=16, stride=1,dilation=1, padding=2),
#             torch.nn.ReLU(),
#         )
    
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# # 创建1D卷积网络
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Sequential( # 输入1通道,输出16通道,卷积核大小为5,步长为1,填充为2
#             torch.nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=3),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool1d(kernel_size=2, stride=2) # 池化层
#         )
    
#     def forward(self, x):
#         x = self.conv1(x)
#         return x


# # wavegan的生成器 仅使用linear线性层
# class Generator_linear(torch.nn.Module):
#     def __init__(self,nclass,embd=100,nz=100,nout=2048,nc=1):
#         """
#         nclass: 标签总数
#         embd:emdeding后的维度
#         nz: 噪声维度
#         nout: 生成器最终输出维度 (采样点个数)
#         nc: 生成器输出通道数
#         """
#         super(Generator_linear, self).__init__()
#         # 将标签embeding后和噪声一起作为输入
#         self.embedding = nn.Embedding(nclass, embd)
#         self.model = torch.nn.Sequential(
#             # 输入是一个nz维度的噪声 [b,nz+self.embd] -> [b,nout//8]
#             nn.Linear(nz+embd, nout//8),
#             nn.BatchNorm1d(nout//8),
#             nn.ReLU(),
#             # [b,nout//8] -> [b,nout//4]
#             nn.Linear(nout//8, nout//4),
#             nn.BatchNorm1d(nout//4),
#             nn.ReLU(),
#             # [b,nout//4] -> [b,nout//2]
#             nn.Linear(nout//4, nout//2),
#             nn.BatchNorm1d(nout//2),
#             nn.ReLU(),
#             # [b,nout//2] -> [b,nout]
#             nn.Linear(nout//2, nout),
#             nn.BatchNorm1d(nout),
#             # 根据最终输出值的范围(-1,1)设置tanh激活函数
#             # 如果后续数据集的幅值范围不是(-1,1)则需要修改
#             nn.Tanh(),
#         )
    
#     def forward(self, input,label):
#         # 将噪声和labele按最后一个维度拼接
#         input = torch.cat((input,self.embedding(label)),dim = -1) # 拼接[batch,nz]->[batch,nz+self.embd]
#         return self.model(input)

# # wavegan的判别器 仅使用linear线性层
# class Discriminator_linear(torch.nn.Module):
#     def __init__(self,nclass,embd=100,nin=2048,nc=1):
#         """
#         nclass: 标签总数
#         embd: emdeding后的维度
#         nin: 判别器输入维度 (采样点个数)
#         nc: 判别器输入通道数
#         """
#         super(Discriminator_linear, self).__init__()
#         # G和 D使用独立的embedding
#         self.embedding = nn.Embedding(nclass, embd)
#         self.nin = nin + embd
#         self.model = torch.nn.Sequential(
#             # 输入是一个nin维度的数据 [b,nin] -> [b,nin//2]
#             nn.Linear(self.nin, self.nin//2),
#             nn.LeakyReLU(0.2),
#             # [b,nin//2] -> [b,nin//4]
#             nn.Linear(self.nin//2, self.nin//4),
#             nn.LeakyReLU(0.2),
#             # [b,nin//4] -> [b,nin//8]
#             nn.Linear(self.nin//4, self.nin//8),
#             nn.LeakyReLU(0.2),
#             # [b,nin//8] -> [b,1]
#             nn.Linear(self.nin//8, 1),
#         )
    
#     def forward(self, input, label):
#         # 将输入和labele按最后一个维度拼接
#         input = torch.cat((input,self.embedding(label)),dim = -1)
#         return self.model(input)




# 卷积生成器
class Generator_conv(torch.nn.Module):
    def __init__(self, nclass,embd=100,nz=100,nout=2048,nc=1, use_normalize = 0):
        """
        nclass: 标签总数
        embd:emdeding后的维度
        nz: 噪声维度
        nout: 生成器最终输出维度 (采样点个数)
        nc: 生成器输出通道数
        """
        super(Generator_conv, self).__init__()

        # 选择normalize的类型
        switcher = {
            0: lambda x : nn.Identity(x), # 不使用normalize nn.Identity()是一个空层
            1: lambda x : nn.BatchNorm1d(x),
            2: lambda x : nn.InstanceNorm1d(x),
        }
        self.normalize = switcher.get(use_normalize)

        self.nout = nout
         # 将标签embeding后和噪声一起作为输入
        self.embedding = nn.Embedding(nclass, embd)
        self.linear = torch.nn.Sequential(

            # 输入是一个nz维度的噪声 [b,nz+self.embd] -> [b,nout]
            nn.Linear(nz+embd, nout),
            self.normalize(nout),
            nn.ReLU(),
        )

        # # 输入为[b,nout//8,8] 
        # self.conv = torch.nn.Sequential(
        #     # [b,nout//8,8] -> [b,nout//16,16]
        #     nn.ConvTranspose1d(nout//8, nout//16, kernel_size=4, stride=2, padding=1),
        #     self.normalize(nout//16),
        #     nn.ReLU(),
        #     # [b,nout//16,16] -> [b,nout//32,32]
        #     nn.ConvTranspose1d(nout//16, nout//32, kernel_size=4, stride=2, padding=1),
        #     self.normalize(nout//32),
        #     nn.ReLU(),
        #     # [b,nout//32,32] -> [b,nout//64,64]
        #     nn.ConvTranspose1d(nout//32, nout//64, kernel_size=4, stride=2, padding=1),
        #     self.normalize(nout//64),
        #     nn.ReLU(),
        #     # [b,nout//64,64] -> [b,nout//128,128]
        #     nn.ConvTranspose1d(nout//64, nout//128, kernel_size=4, stride=2, padding=1),
        #     self.normalize(nout//128),
        #     nn.ReLU(),
        #     # [b,nout//128,128] -> [b,nout//256,256]
        #     nn.ConvTranspose1d(nout//128, nout//256, kernel_size=4, stride=2, padding=1),
        #     self.normalize(nout//256),
        #     nn.ReLU(),
        #     # [b,nout//256,256] -> [b,nout//1024,1024]
        #     nn.ConvTranspose1d(nout//256, nout//1024, kernel_size=4, stride=4, padding=0),
        #     self.normalize(nout//1024),
        #     nn.ReLU(),
        #     # [b,nout//1024,1024] -> [b,nout//2048,2048]
        #     nn.ConvTranspose1d(nout//1024, nout//2048, kernel_size=4, stride=2, padding=1),
        #     self.normalize(nout//2048),
        #     nn.Tanh(),
        #     # [b,nout//2048,2048] -> [b,nc,2048]
        #     nn.ConvTranspose1d(nout//2048, nc, kernel_size=1, stride=1, padding=0),
        #     # 根据最终输出值的范围(-1,1)设置tanh激活函数
        #     # 如果后续数据集的幅值范围不是(-1,1)则需要修改
        #     nn.Tanh(),
        # )
        
        # 输入为[b,nout//8,8] 
        self.conv = torch.nn.Sequential(
            # [b,nout//8,8] -> [b,nout//16,16]
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(nout//8, nout//16, kernel_size=5, stride=1, padding=2),
            self.normalize(nout//16),
            nn.ReLU(),
            # [b,nout//16,16] -> [b,nout//32,32] 
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(nout//16, nout//32, kernel_size=5, stride=1, padding=2),
            self.normalize(nout//32),
            nn.ReLU(),
            # [b,nout//32,32] -> [b,nout//64,64]
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(nout//32, nout//64, kernel_size=5, stride=1, padding=2),
            self.normalize(nout//64),
            nn.ReLU(),
            # [b,nout//64,64] -> [b,nout//128,128]
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(nout//64, nout//128, kernel_size=5, stride=1, padding=2),
            self.normalize(nout//128),
            nn.ReLU(),
            # [b,nout//128,128] -> [b,nout//256,256]
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(nout//128, nout//256, kernel_size=5, stride=1, padding=2),
            self.normalize(nout//256),
            nn.ReLU(),
            # [b,nout//256,256] -> [b,nout//1024,1024]
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv1d(nout//256, nout//1024, kernel_size=5, stride=1, padding=2),
            self.normalize(nout//1024),
            nn.ReLU(),
            # [b,nout//1024,1024] -> [b,nout//2048,2048]
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(nout//1024, nout//2048, kernel_size=5, stride=1, padding=2),
            self.normalize(nout//2048),
            nn.Tanh(),
            # [b,nout//2048,2048] -> [b,nc,2048]
            nn.Conv1d(nout//2048, nc, kernel_size=1, stride=1, padding=0),
            # 根据最终输出值的范围(-1,1)设置tanh激活函数
            # 如果后续数据集的幅值范围不是(-1,1)则需要修改
            nn.Tanh(),
        )
    
    def forward(self, input,label):
        # 将噪声和labele按最后一个维度拼接
        input = torch.cat((input,self.embedding(label)),dim = -1) # 拼接[batch,nz]->[batch,nz+self.embd]
        input = self.linear(input) # [batch,nz+self.embd] -> [batch,nout]
        input = input.view(-1,self.nout//8,8) # [batch,nout] -> [batch,nout//8,8]
        return self.conv(input).squeeze(1) # [batch,nout//8,8] -> [batch,1,nout] -> [batch,nout]



# # 卷积判别器
# class Discriminator_conv(torch.nn.Module):
#     def __init__(self, nclass,embd=128,sign=2048,nc=1):
#         """
#         nclass: 标签总数
#         embd: emdeding后的维度
#         sign: 信号长度(采样点个数)
#         nc: 判别器输入通道数
#         """
#         super(Discriminator_conv, self).__init__()
#         # G 和 D使用独立的embedding
#         self.embedding = nn.Embedding(nclass, embd)
#         self.nin = sign + embd
#         self.conv = torch.nn.Sequential(
#             # 输入是一个nin维度的数据 [b,1,nin] -> [b,4,nin//2]
#             nn.Conv1d(nc, 4, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             # [b,4,nin//2] -> [b,8,nin//4]
#             nn.Conv1d(4, 8, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             # [b,8,nin//4] -> [b,16,nin//8]
#             nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             # [b,16,nin//8] -> [b,32,nin//16]
#             nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             # [b,32,nin//16] -> [b,64,nin//32]
#             nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             # 展开 [b,64,nin//32] -> [b,64*nin//32]
#             nn.Flatten(),
#             # [b,64*nin//32] -> [b,1]
#             nn.Linear(64*(self.nin//32), 1), 

#         )
    
#     def forward(self, input, label):
#         # 将输入和labele按最后一个维度拼接
#         input = torch.cat((input,self.embedding(label)),dim = -1)
#         input = input.unsqueeze(1) # [batch,nin] -> [batch,1,nin]
#         return self.conv(input)





class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, ues_1x1conv=False,strides=1):
        """
        初始化残差块中的网络
        :param in_channels: 输入通道个数
        :param out_channels: 输出通道个数
        :param ues_1x1conv: 是否使用1x1卷积改变输入通道个数后再加到输出
        :param strides: 滑动步幅 设置为2时size减半
        """
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=strides, padding=2) # 残差块中第一个残差网络输入的size减半（将stride设置为2）
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)  # 第二层卷积size不变
        self.LeakyReLU = nn.LeakyReLU(0.2)

        if ues_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=strides)  # 使用1x1卷积改变输入的通道加到最后一层
        else:
            self.conv3 = None


    def forward(self, x):
        y = self.conv1(x)  # 第一层卷积
        y = self.LeakyReLU(y)
        y = self.conv2(y)  # 第二层卷积

        if self.conv3:
            x = self.conv3(x)
        y = y + x

        return self.LeakyReLU(y)





# 卷积判别器
class Discriminator_conv(torch.nn.Module):
    def __init__(self, nclass,embd=128,sign=2048,nc=1,loss=0):
        """
        nclass: 标签总数
        embd: emdeding后的维度 设置为128方便和sign长度加起来可以整除到128 (如果设置为100,2148只能整除到4)
        sign: 信号长度(采样点个数)
        nc: 判别器输入通道数
        loss: 判别器损失函数类型 (BCE_loss需要最后加上sigmoid)
        """
        super(Discriminator_conv, self).__init__()
        # G 和 D使用独立的embedding
        self.embedding = nn.Embedding(nclass, embd)
        self.nin = sign + embd 

        self.block0 = nn.Sequential(self.resnet_block(1, 16, 2))  # [b,1,nin] -> [b,16,nin//2]
        self.block1 = nn.Sequential(self.resnet_block(16, 32, 2)) # [b,16,nin//2] -> [b,32,nin//4]
        self.block2 = nn.Sequential(self.resnet_block(32, 64, 2)) # [b,32,nin//4] -> [b,64,nin//8]
        self.block3 = nn.Sequential(self.resnet_block(64, 128, 2)) # [b,64,nin//8] -> [b,128,nin//16]
        self.block4 = nn.Sequential(self.resnet_block(128, 256, 2)) # [b,128,nin//16] -> [b,256,nin//32]
        self.block5 = nn.Sequential(self.resnet_block(256, 512, 2)) # [b,256,nin//32] -> [b,512,nin//64]
        self.conv = torch.nn.Sequential(
            self.block0,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            nn.Flatten(),
            # [b,512*nin//64] -> [b,1]
            nn.Linear(512*(self.nin//64), 512), 
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )
        # 根据损失函数类型选择是否使用sigmoid
        if loss == 0:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Identity()

    # resnet块
    def resnet_block(self, in_channels, out_channels, num_residuals):
        """
        ResNet中使用的残差模块 每一块中包含两个残差网络
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param num_residuals: 模块中残差网络个数
        """
        blk = []
        for i in range(num_residuals):
            if i == 0 :
                # 第一个残差网络需要将输入的通道加倍，size减半，所以需要使用1x1卷积改变通道数
                blk.append(Residual(in_channels, out_channels, ues_1x1conv=True,strides=2))

            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)


    def forward(self, input, label):
        # 将输入和labele按最后一个维度拼接
        input = torch.cat((input,self.embedding(label)),dim = -1)
        input = input.unsqueeze(1) # [batch,nin] -> [batch,1,nin]
        input = self.conv(input)
        return self.sigmoid(input)

# 使用pytorch实现匹配滤波
class MatchFilter(nn.Module):
    def __init__(self,T=10e-6,B=10e6):
        super(MatchFilter, self).__init__()
        self.C=3e8;                              #propagation speed
        self.T=T;                           #脉冲宽度 10us
        self.B=B;                            #带宽
        self.K=self.B/self.T;                                 #chirp slope
        self.Fs=5*self.B
        self.Ts=1/self.Fs;                         #sampling frequency and sampling spacing
        
        self.N_fft = 2048;         # fft点数 需要和生成器输出维度一致

        # pytorch生成LFM信号
        self.t = torch.arange(0, self.T,self.Ts)
        self.pi = torch.tensor(np.pi)
        self.s = torch.exp(1j * self.pi * self.K * self.t**2)

        # s的共轭序列
        self.s1 = torch.conj(self.s.flip(0)) #flip(0) 翻转第0维  conj()共轭

        # pytorch生成匹配滤波器的频谱
        self.filter_fft = torch.fft.fft(self.s1,self.N_fft)

    def forward(self, x):
        # 将filter_fft移动到x所在的设备上
        filter_fft = self.filter_fft.to(x.device)
        # x的频谱
        r_fft = torch.fft.fft(x,2048) # 2048为fft点数
        # 匹配滤波
        sr = torch.fft.ifft(filter_fft*r_fft)
        return sr.real # 返回实部


if __name__ == "__main__":
    # 产生一个(10,100)的噪声
    noise = torch.randn(10,100)
    y = torch.zeros(10,1).long()
   

    # 将噪声输入到生成器中
    netG = Generator_conv(1,100,100,2048,use_normalize=2)
    output = netG(noise,y)
    print(output.shape)

    # 将生成器的输出输入到判别器中
    # net = Discriminator_linear()
    # output = net(output,y)
    # print(output[0])



