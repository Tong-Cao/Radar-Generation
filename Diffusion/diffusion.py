import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader
from Dataset import MyDataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
import einops
from torch import nn
from unet import ResidualBlock, UNet
import time
import os
import random

class DDPM():

    # n_steps 就是论文里的 T
    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device) # 生成一个从min_beta到max_beta的等差数列
        alphas = 1 - betas # α =  1 - β 
        alpha_bars = torch.empty_like(alphas) # 生成一个和alphas一样大小的tensor
        product = 1 # 用来计算alpha_bar
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product # 将alpha累乘的结果 ~α 存入alpha_bars

        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    # 计算前向过程的 x_t = sqrt(~α) * x_0 + sqrt(1 - ~α) * eps_t    其中eps_t ~ N(0,1)
    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) # 将alpha_bar转换为（batch_size, ）的形状
        if eps is None:
            eps = torch.randn_like(x)       
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x 
        return res
    
    # 生成随机噪声再不断调用sample_backward_step来还原图像
    def sample_backward(self, img_shape, label, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device) # 生成一个随机噪声
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, label, net, simple_var)
        return x # 返回还原的图像
    
    # 计算反向过程的 x_(t-1) = (x_t - (1 - α) / sqrt(1 - ~α) * eps_t) / sqrt(α) + noise
    def sample_backward_step(self, x_t, t, label, net, simple_var=True):
        n = x_t.shape[0] # 得到图像的batch_size
        t_tensor = torch.tensor([t] * n, 
                                dtype=torch.long).to(x_t.device).unsqueeze(1) # 生成一个batch_size大小的tensor 值都为t：{t,t,...,t,t,t}
        eps = net(x_t, t_tensor, label) # 使用神经网络预测eps_t

        if t == 0: # 如果是第零步，那么噪声就是0
            noise = 0
        else: 
            if simple_var:  # 选择方差，两种方差的选择都可以
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t) # noise ~ N(0,1)
            noise *= torch.sqrt(var) # noise ~ N(0,var)
        
        # 计算x_t分布的均值 mean = (x_t - (1 - α) / sqrt(1 - ~α) * eps_t) / sqrt(α)  
        # 这里让神经网络预测的eps_t等于正向过程的eps_t时可以完全得到x_t的分布 (神经网络的目标就是让eps_t尽可能接近正向过程的eps_t)
        mean = (x_t -
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise # x_(t-1) ~ N(mean, var)
        
        return x_t
    

# 生成图像
def sample_imgs_plot(ddpm,
                net,
                label,
                n_sample=10,
                device='cuda',
                simple_var=True):

    net = net.to(device)
    net = net.eval()


    with torch.no_grad():

        
        shape = (n_sample, 1, 1, 2048)
        label_tensor = torch.tensor([label] * n_sample, dtype=torch.long).to(device)
        Y = ddpm.sample_backward(shape,
                                label_tensor,
                                net,
                                device=device,
                                simple_var=simple_var).detach().cpu()
        
        for i in range(n_sample):
            plt.clf()
            plt.plot(Y[i].squeeze())
            plt.savefig('./diffusion_result/sample_'+str(i)+'_t.png')
            np.save('D://Radar_data//diffusion_A2//'+'AM_noise'+'//fake_sign_'+str(i)+'.npy',Y[i].squeeze())
            print('save sample_'+str(i)+'_t.png')

            
            
# 使用训练好的网络生成图像数据
def sample_imgs(ddpm,
                net,
                label,
                n_sample=1024,
                device='cuda',
                simple_var=True):

    net = net.to(device)
    net = net.eval()

    # 每次生成256个图像
    batch_s = 256

    with torch.no_grad():

        for i in range(n_sample//batch_s):
            shape = (batch_s, 1, 1, 2048)
            label_tensor = torch.tensor([label] * batch_s, dtype=torch.long).to(device)
            Y = ddpm.sample_backward(shape,
                                    label_tensor,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        
            for j in range(batch_s):
                y = Y[j].squeeze()
                # plt.clf()
                # plt.plot(y)
                # plt.savefig('./diffusion_result/sample_'+str(_)+'_t.png')
                
                # 保存为npy文件
                np.save('./diffusion_result/LFM/fake_sign_'+str(i*batch_s+j)+'.npy',y)
                print('save fake_sign_'+str(i*batch_s+j)+'.npy')


# 生成3000个npy
def sample_imgs_save(ddpm,
                net,
                n_sample=3000,
                device='cuda',
                simple_var=True):

    net = net.to(device)
    net = net.eval()

    # 每次生成256个图像
    batch_s = 100

    label_list = ['AM_noise','FM_noise','ISRJ','LFM']

    with torch.no_grad():

        for j in range(4):

            for i in range(n_sample//batch_s):
                shape = (batch_s, 1, 1, 2048)
                label_tensor = torch.tensor([j] * batch_s, dtype=torch.long).to(device)
                Y = ddpm.sample_backward(shape,
                                        label_tensor,
                                        net,
                                        device=device,
                                        simple_var=simple_var).detach().cpu()
        
                for k in range(batch_s):
                    y = Y[k].squeeze()
                    # 保存为npy文件
                    np.save('D://Radar_data//diffusion_A//'+label_list[j]+'//fake_sign_'+str(i*batch_s+k)+'.npy',y)
        
                    print('save fake_sign_'+str(i*batch_s+k)+'.npy')
    

# 生成3000个npy
def sample_imgs_save2(ddpm,
                net,
                n_sample=3000,
                device='cuda',
                simple_var=True):

    net = net.to(device)
    net = net.eval()

    # 每次生成256个图像
    batch_s = 100

    label_list = ['AM_noise','FM_noise','ISRJ','LFM']

    with torch.no_grad():

        for j in range(4):

            for i in range(n_sample//batch_s):
                shape = (batch_s, 1, 1, 2048)
                label_tensor = torch.tensor([j] * batch_s, dtype=torch.long).to(device)
                Y = ddpm.sample_backward(shape,
                                        label_tensor,
                                        net,
                                        device=device,
                                        simple_var=simple_var).detach().cpu()
        
                for k in range(batch_s):
                    y = Y[k].squeeze()
                    # 保存为npy文件
                    np.save('D://Radar_data//diffusion_A2//'+label_list[j]+'//fake_sign_'+str(i*batch_s+k)+'.npy',y)
        
                    print('save fake_sign_'+str(i*batch_s+k)+'.npy')

        


# 画出正向添加噪声的正向过程
def plot_img():
    root = './data'

    mydataset = MyDataset(root=root, part='realsp')  # 实例化dataset
    train_iter = DataLoader(dataset=mydataset, batch_size=10,
                            drop_last=True,shuffle=True
                            ) 
  
    X, y = next(iter(train_iter))  # next(iter())迭代器 取出一条batch
    # 将x从（batch_size,2048）-> (batch_size,1,1,2048)
    X = X.unsqueeze(1).unsqueeze(1)

    # 画出X的第一个数据
    plt.plot(X[0][0][0])
    plt.savefig('sample_0.png')

    
    # 取batch中的第一个数据
    x = X[0]
    print('输入',x.shape)
    # 将x复制10份
    X = torch.cat([x] * 10, dim=0)
    X = X.unsqueeze(1)
    print('输入',X.shape)
    

    ddpm = DDPM('cpu', 1000)
    
    # 画出不同t时的正向过程
    t = torch.tensor([[1],[10],[50],[100],[500],[600],[700],[800],[900],[999]])
    print(t.shape)

    X_t = ddpm.sample_forward(X, t)
    print(X_t.shape)

    for _ in range(10):
        X_y = X_t[_].squeeze()
        plt.clf()
        plt.plot(X_y)
        #去掉方框和坐标轴
        plt.axis('off')
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.savefig('sample_'+str(_)+'_t.png',dpi = 300)

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



# 训练函数
def train(root, n_epochs, batch_s, ddpm: DDPM, net, device, learning_rate, shutdown=False):
    # n_steps 就是公式里的 T
    # net 是某个继承自 torch.nn.Module 的神经网络
    n_steps = ddpm.n_steps # 在DDPM类初始化时定义

    mydataset = MyDataset(root=root, part='realsp')  # 实例化dataset
    train_iter = DataLoader(dataset=mydataset, batch_size=batch_s,
                            drop_last=True,shuffle=True
                            ) 
    
    net = net.to(device)
    loss_fn = nn.MSELoss() # 使用均方误差作为损失函数
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    

    train_start_time = time.time()

    from torch.utils.tensorboard import SummaryWriter
    # 删除之前的tensorboard文件
    os.system('rd /s/q tensorboard_save')
    writer = SummaryWriter('tensorboard_save') #建立一个保存数据用的东西，save是输出的文件名


    for e in range(n_epochs):
        loss_sum, n = 0.0 , 0
        start_time = time.time()

        for x, label in train_iter:
            label = label.to(device)
            x = x.to(torch.float32)
            # 将x从（batch_size,2048）-> (batch_size,1,1,2048)
            x = x.unsqueeze(1).unsqueeze(1)

            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device) # 随机生成一个[0, n_steps)的整数(每次随机选择一个t来训练而不是每次都要训练所有的过程)
            eps = torch.randn_like(x).to(device) # 随机噪声
            x_t = ddpm.sample_forward(x, t, eps) # 生成第t步的图像
            eps_theta = net(x_t, t.reshape(current_batch_size, 1),label) # 使用神经网络预测eps_t
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.cpu().item()
            n += current_batch_size
        
        # 打印每个epoch的损失
        print('epoch %d, loss %.4f, time %.4f' % (e + 1, loss_sum/n, time.time() - start_time))
        # 将每个epoch的损失写入tensorboard
        writer.add_scalar(tag="loss/train", scalar_value=loss_sum/n,
                        global_step=e + 1)
        
        # 每50轮保存一次模型
        if (e + 1) % 50 == 0:
            torch.save(net.state_dict(), './diffusion_net_save/model_{}.pth'.format(e+1))
    

    # 将训练时间转为小时写入time.txt
    train_time = time.time() - train_start_time
    with open('time.txt','w') as f:
        f.write('train time: '+str(train_time/3600)+' hours')
    
    # 关机
    if shutdown:
        # 训练完成关机
        print('训练完成, 将在60s后关机')
        os.system('shutdown -s -f -t 60')
    




if __name__ == '__main__':
    # 随机数种子
    seed_num = 24
    setup_seed(seed_num)

    n_steps = 1000
    n_epochs = 150
    batch_s = 32
    device = 'cuda'
    lr = 1e-4 # 学习率
    model_path = './diffusion_model/att_net2_128_250.pth' # ./diffusion_model/nonatt_320.pth  ./diffusion_net_save/model_50.pth

    dataset_root = './data'

    unet_res_cfg = {
    'channels': [4, 8, 16, 32, 64, 128],
    # 'channels': [10, 20, 40, 80, 160, 320],
    'n_classes': 4,
    'pe_dim': 128,
    'residual': True,
    'attention': True
    }
    net = UNet(n_steps, **unet_res_cfg)
    
    ddpm = DDPM(device, n_steps)

    # 使用训练好的网络继续训练
    # net.load_state_dict(torch.load(model_path))

    # train(dataset_root, n_epochs, batch_s, ddpm, net, device=device, learning_rate=lr, shutdown=True)

    net.load_state_dict(torch.load(model_path))

    # 保存生成的图像
    # sample_imgs_save(ddpm, net, n_sample=3000, device=device)


   # label: 
   # 0: AM_noise
   # 1: FM_noise
   # 2: ISRJ
   # 3: LFM

    sample_imgs_plot(ddpm, net, label=0, n_sample=10, device=device) 

    # sample_imgs(ddpm, net, label=3, n_sample=1024, device=device)

     # 训练完成关机
    # print('训练完成, 将在60s后关机')
    # os.system('shutdown -s -f -t 60')

   







