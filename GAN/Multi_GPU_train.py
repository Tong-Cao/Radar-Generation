import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import time
import random

from Dataset import MyDataset
# from net import Discriminator_linear, Generator_linear, Discriminator_conv, Generator_conv
from net import Discriminator_conv, Generator_conv, MatchFilter

#删除之前的tensorboard文件
if os.path.exists('tensorboard_save'):
    import shutil
    shutil.rmtree('tensorboard_save')
    
# 设置超参数
batch_size = 64 # batch设置为1时，batchnorm会报错，不使用batchnorm
lr = 0.001 # 学习率
nz = 100 # 噪声维度
nclass = 4 # 类别数
embd = 100 # embedding维度
nepoch = 6000 # 训练轮数
nout = 2048 # 生成器输出维度
normalize = 1 # batchnorm类别 0:不使用 1:BatchNorm1d, 2:InstanceNorm1d
loss = 1 # loss类别   0:BCE_loss     1:WGAN-GP  2: WGAN-GP+match
mactch_filter_lambda = 0.0 # 匹配滤波损失比例系数
seed_num = 24 # 随机数种子 20
continue_train = False # 是否继续训练


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 保证每次运行结果一样
setup_seed(seed_num)


from torch.utils.data.distributed import DistributedSampler
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")

# 2） 配置每个进程的gpu
local_rank = torch.distributed.get_rank()
print('local_rank',local_rank)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# 设置数据集
root = './data'
mydataset = MyDataset(root=root, part='realsp')  # 实例化dataset
# 3）使用DistributedSampler
train_iter = DataLoader(dataset=mydataset,
                        batch_size=batch_size,
                        drop_last=True,
                        shuffle=False,
                        sampler=DistributedSampler(mydataset))

# 实例化生成器和判别器
netG = Generator_conv(nclass,embd,nz,nout,use_normalize=normalize).to(device)
netD = Discriminator_conv(nclass,loss=loss).to(device)
netD_FM = Discriminator_conv(nclass,loss=loss).to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
    netG = torch.nn.parallel.DistributedDataParallel(netG,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    netD = torch.nn.parallel.DistributedDataParallel(netD,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    netD_FM = torch.nn.parallel.DistributedDataParallel(netD_FM,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
                        
# 匹配滤波器
match_filter = MatchFilter().to(device)

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
    
# 继续训练
if continue_train:
    netG.load_state_dict(torch.load('./net_dict/netG_epoch_4000.pth'))
    netD.load_state_dict(torch.load('./net_dict/netD_epoch_4000.pth'))
    netD_FM.load_state_dict(torch.load('./net_dict/netD_FM_epoch_4000.pth'))
    # netG = load_model(netG,'./net_save/22/netG_epoch_1000.pth')
    # netD = load_model(netD,'./net_save/22/netD_epoch_1000.pth')
    # netD_FM = load_model(netD_FM,'./net_save/22/netD_epoch_1000.pth')
    print('load netG and netD')


# 设置优化器
optimizerD = torch.optim.SGD(netD.parameters(), lr=lr)#优化器只有netD的参数 更新时不会影响netG
optimizerD_FM = torch.optim.SGD(netD_FM.parameters(), lr=lr)
optimizerG = torch.optim.SGD(netG.parameters(), lr=lr)

# 设置损失函数
BCe_loss = torch.nn.BCELoss().to(device) # 二分类交叉熵
lambda_gp = 10 # 比例系数

# 计算梯度惩罚项
def compute_gradient_penalty(D,real_samples,fake_samples,label):
    # 获取一个随机数，作为真假样本的采样比例
    eps = torch.FloatTensor(real_samples.size(0),1).uniform_(0,1).to(device)
    # 按照eps比例生成真假样本采样值X_inter
    X_inter = (eps * real_samples + ((1-eps)*fake_samples)).requires_grad_(True)
    d_interpolates = D(X_inter,label)
    # 创造一个全为1的张量，1*10
    fake = torch.full((real_samples.size(0),1),1,device=device) # 计算梯度输出的掩码，在本例中需要对所有梯度进行计算，故需要按照样本个数生成全为1的张量。

    # 求梯度
    gradients = torch.autograd.grad(outputs=d_interpolates, # 输出值outputs，传入计算过的张量结果
                              inputs=X_inter,# 待求梯度的输入值inputs，传入可导的张量，即requires_grad=True
                              grad_outputs=fake, # 传出梯度的掩码grad_outputs，使用1和0组成的掩码，在计算梯度之后，会将求导结果与该掩码进行相乘得到最终结果。
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True
                              )[0]
    gradients = gradients.view(gradients.size(0),-1)
    gradient_penaltys = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penaltys



from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard_save') #建立一个保存数据用的东西，save是输出的文件名

# 训练
for epoch in range(nepoch):

    g_loss_sum, d_loss_sum, n = 0.0 , 0.0 , 0
    start_time = time.time()

    for i, (data, label) in enumerate(train_iter):

        real_sign = data.to(torch.float32).to(device)
        label = label.to(device)

        # 判别器每训练5次生成器训练1次
        for ii in range(3):
            # 判别器训练
            netD.zero_grad() # 梯度清零
            netD_FM.zero_grad()

            # 对real_sign进行判别
            real_out = netD(real_sign,label)
            # 生成随机值 
            z = torch.randn(label.size(0),100).to(device)
            fake_sign = netG(z,label) # 生成fake_sign
            fake_out = netD(fake_sign,label) # 对fake_sign进行判别
            
            # 选择损失函数
            if loss == 0:
                d_loss = BCe_loss(real_out,torch.ones_like(real_out).to(device))+BCe_loss(fake_out,torch.zeros_like(fake_out).to(device))
            elif loss == 1:
                # 计算梯度惩罚项
                gradient_penalty = compute_gradient_penalty(netD,real_sign.data,fake_sign.data,label)
            
                d_loss = -torch.mean(real_out)+torch.mean(fake_out) + gradient_penalty
                d_loss.backward()
                optimizerD.step()
                d_loss_sum += d_loss.cpu().item()

            elif loss == 2:
                # 计算梯度惩罚项
                gradient_penalty = compute_gradient_penalty(netD,real_sign.data,fake_sign.data,label)
                d_loss_1 = (1-mactch_filter_lambda) * (-torch.mean(real_out)+torch.mean(fake_out) + gradient_penalty)

                # 匹配滤波
                real_sp = match_filter(real_sign.detach()) # 对real_sign进行匹配滤波 使用detach()阻止梯度传递到netG节省计算资源
                fake_sp = match_filter(fake_sign.detach()) 
                # 对匹配滤波后的信号进行判别
                real_sp_out = netD_FM(real_sp,label)
                fake_sp_out = netD_FM(fake_sp,label)
                # 计算匹配滤波损失
                gradient_penalty_sp = compute_gradient_penalty(netD_FM,real_sp.data,fake_sp.data,label)
                d_loss_2 = mactch_filter_lambda * (-torch.mean(real_sp_out)+torch.mean(fake_sp_out) + gradient_penalty_sp)
                
                d_loss_1.backward()
                optimizerD.step()
                d_loss_2.backward()
                optimizerD_FM.step()

                d_loss_sum += d_loss_1.cpu().item() + d_loss_2.cpu().item()
            
        
        # 生成器训练
        netG.zero_grad()
        z = torch.randn(label.size(0),100).to(device)
        fake_sign = netG(z,label)
        fake_out = netD(fake_sign,label)
        # 选择损失函数
        if loss == 0:
            g_loss = BCe_loss(fake_out,torch.ones_like(fake_out).to(device))
        elif loss == 1:
            g_loss = -torch.mean(fake_out)
        elif loss == 2:
            # 生成信号损失
            g_loss = (1-mactch_filter_lambda) * (-torch.mean(fake_out))

            # 匹配滤波
            fake_sp = match_filter(fake_sign) # 对fake_sign进行匹配滤波
            fake_sp_out = netD_FM(fake_sp,label)
            # 加上匹配滤波损失
            g_loss += mactch_filter_lambda * (-torch.mean(fake_sp_out))

        g_loss.backward()
        optimizerG.step()
        g_loss_sum += g_loss.cpu().item()
        n += label.size(0)
    
    
    # 打印loss
    print('epoch %d, g_loss %.4f, d_loss %.4f, time %.4f' % (epoch + 1, g_loss_sum/n, d_loss_sum/n, time.time() - start_time))
    # tensorboard记录loss
    writer.add_scalar(tag="g_loss/train", scalar_value=g_loss_sum/n,
                        global_step=epoch + 1)

    writer.add_scalar(tag="d_loss/train", scalar_value=d_loss_sum/n, # D每个batch训练5次
                        global_step=epoch + 1)
    # 保存结果
    if (epoch+1)%50==0:
        # torch.save(netG.state_dict(),'./net_dict/netG_epoch_{}.pth'.format(epoch+1))
        # torch.save(netD.state_dict(),'./net_dict/netD_epoch_{}.pth'.format(epoch+1))
        # print('save netG and netD')
        # 保存fake_sign
        fake_sign = torch.cat((fake_sign,label.unsqueeze(1)),dim=-1) # 加上标签信息
        fake_sign = fake_sign.cpu().detach().numpy()
        fake_sign = fake_sign.reshape(-1,2048+1)
        np.save('./result/fake_sign_epoch_{}.npy'.format(epoch+1),fake_sign)

    # 保存最后一轮网络
    if epoch == nepoch-1:
        torch.save(netG.state_dict(),'./net_dict/netG_epoch_{}.pth'.format(epoch+1))
        torch.save(netD.state_dict(),'./net_dict/netD_epoch_{}.pth'.format(epoch+1))
        torch.save(netD_FM.state_dict(),'./net_dict/netD_FM_epoch_{}.pth'.format(epoch+1))
        print('save netG and netD')
 


