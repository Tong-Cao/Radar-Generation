import torch
import torch.nn as nn
import torch.nn.functional as F

# 位置编码 将时间步t编码到一个d_model维的向量中
class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):

        # max_seq_len: 序列的最大长度 也就是训练的T最大值
        # d_model: 位置编码的维度, （一般将t先embedded到d_model维度然后再通过线性层将其转化为和图片channel相同的维度通过广播机制直接加在图片上）
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0
        
        # 使用sin和cos函数来编码位置信息
        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe # 将pe作为embedding的权重
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)
    


# 残差块
class ResidualBlock(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.actvation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.actvation2 = nn.ReLU()
        # 如果输入输出通道数不一样，需要对shortcut进行处理
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                          nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.actvation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(input)
        x = self.actvation2(x)
        return x
    

# UNet块
class UnetBlock(nn.Module):

    def __init__(self, shape, in_c, out_c, residual=False, attention=False):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.LeakyReLU()
        self.residual = residual
        self.attention = attention
        # 是否加入残差连接
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)
        # 是否加入注意力机制
        if attention:
            self.pe = PositionalEncoding(2048, 32) # 位置编码 ，映射到32维
            self.position_linear = nn.Linear(32, out_c) # 位置编码映射到和图片通道数相同
            self.attention = nn.MultiheadAttention(out_c, 1, batch_first=True) # embed_dim, num_heads, batch_first
            

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)

        if self.attention:
            out = out.permute(2,0,3,1).squeeze(0).contiguous() # (b,c,h,w)->(h,b,w,c)->(h*b,w,c)
            # 位置编码
            position = torch.arange(out.shape[1]).to(out.device) # (w,)  
            position = self.pe(position) # (w,)->(w,32)
            position = self.position_linear(position) # (w,32)->(w,c)
            out = out + position # (h*b,w,c) + (w,c) = (h*b,w,c)                                                         
            out = self.attention(out, out, out)[0]
            out = out.unsqueeze(0).permute(1,3,0,2).contiguous()

        
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
            
        return out


class UNet(nn.Module):

    def __init__(self,
                 n_steps,
                 n_classes, # 类别数
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 residual=False,
                 attention=False) -> None:
        super().__init__()
        # C, H, W = get_img_shape()
        C, H, W = 1, 1, 2048
        layers = len(channels)
        # 记录每一层的长度
        Ws = [W]  
        cW = W
        # 计算每一层的大小
        for _ in range(layers - 1):
            cW //= 2
            Ws.append(cW)

        self.pe = PositionalEncoding(n_steps, pe_dim)
        self.label_em = nn.Embedding(n_classes, pe_dim) # 类别编码

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.label_linears_en = nn.ModuleList() # 类别编码的线性层encoder部分
        self.label_linears_de = nn.ModuleList() # 类别编码的线性层decoder部分
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_channel = C
        # 编码器 这里先将label进行concat然后卷积提取特征再下采样（b,pre_c,h,w)->(b,pre_c+1,h,w)->(b,c,h,w)->(b,c,h/2,w/2)
        for channel, cW in zip(channels[0:-1], Ws[0:-1]):
            # 位置编码直接加在图片上
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))
            
            prev_channel = prev_channel + 1 # 将类别编码concat到图片上 此时通道数+1
            # 类别编码 将类别编码映射到和图片size相同再直接concat到图片上
            self.label_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, 1*cW), nn.ReLU(),
                                nn.Linear(1*cW, 1*cW), 
                                nn.Unflatten(1, (1, 1, cW)))) # （b,pe_dim）-> (b,1,1,cW)
            
            self.encoders.append(
                nn.Sequential(
                    UnetBlock((prev_channel, 1, cW),
                              prev_channel,
                              channel,
                              residual=residual,
                              attention=attention),
                    UnetBlock((channel, 1, cW),
                              channel,
                              channel,
                              residual=residual,
                              attention=attention)))
            self.downs.append(nn.Conv2d(channel, channel, (1,2), (1,2)))
            prev_channel = channel

        # 中间层
        self.pe_mid = nn.Linear(pe_dim, prev_channel) # 将位置编码的维度映射到和图片通道数相同
        prev_channel = prev_channel + 1 # 将类别编码concat到图片上 此时通道数+1
        self.label_em_mid = nn.Sequential(nn.Linear(pe_dim, 1*Ws[-1]), 
                                        nn.ReLU(),nn.Unflatten(1, (1, 1, Ws[-1])))
        channel = channels[-1]
        self.mid = nn.Sequential(
            UnetBlock((prev_channel, 1, Ws[-1]),
                      prev_channel,
                      channel,
                      residual=residual,
                      attention=attention),
            UnetBlock((channel, 1, Ws[-1]),
                      channel,
                      channel,
                      residual=residual,
                      attention=attention),
        )
        prev_channel = channel
        
        # 解码器 
        # 先通过反卷积上采样到当前ch和cw （b,pre_c,ch/2,cw/2）->(b,c,h,w)
        # 再将encoder的输出和label一起concat到图像上 （b,c,h,w）->(b,2*c+1,h,w)
        # 再通过unetblock解码 （b,2*c+1,h,w）->(b,c,h,w)

        for channel, cW in zip(channels[-2::-1], Ws[-2::-1]):
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            
            # 类别编码 将类别编码映射到和图片size相同再直接concat到图片上
            self.label_linears_de.append(
                nn.Sequential(nn.Linear(pe_dim, 1*cW), nn.ReLU(),
                                nn.Linear(1*cW, 1*cW),
                                nn.Unflatten(1, (1, 1, cW))))
            
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, (1,2), (1,2))) # 反卷积
            self.decoders.append(
                nn.Sequential(
                    UnetBlock((channel * 2 + 1, 1, cW),
                              channel * 2 + 1,
                              channel,
                              residual=residual,
                              attention=attention),
                    UnetBlock((channel, 1, cW),
                              channel,
                              channel,
                              residual=residual,
                              attention=attention)))

            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t, label):
        n = t.shape[0]
        t = self.pe(t)
        label = self.label_em(label) #（b,1）-> (b,pe_dim)

        encoder_outs = []
        for pe_linear, label_linear, encoder, down in zip(self.pe_linears_en, self.label_linears_en,
                                                self.encoders,
                                                self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)# (b,pe_dim)->(b,c,1,1)
            label_em = label_linear(label) # (b,pre_dim)->(b,1,h,w)
            x = x + pe # 将位置编码加到输入上
            x = torch.cat((x, label_em), dim=1) # 将label和图片concat (b,c,h,w)->(b,c+1,h,w)
            x = encoder(x) # 编码 (b,c+1,h,w)->(b,c*2,h/2,w/2)
            encoder_outs.append(x)
            x = down(x) # 下采样 (b,c*2,h/2,w/2)
        
        # 中间层    
        pe = self.pe_mid(t).reshape(n, -1, 1, 1) # (b,pe_dim)->(b,c,1,1)
        label_em = self.label_em_mid(label) # (b,pre_dim)->(b,1,h,w)
        x = x + pe # 将位置编码加到输入上 (b,c,h,w) = (b,c,h,w) + (b,c,1,1)
        x = torch.cat((x, label_em), dim=1) # 将label和图片concat (b,c,h,w)->(b,c+1,h,w)
        x = self.mid(x)

        for pe_linear, label_linear, decoder, up, encoder_out in zip(self.pe_linears_de, self.label_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1, 1) # (b,pe_dim)->(b,pre_c,1,1)
            label_em = label_linear(label) # (b,pre_dim)->(b,1,h,w)
            x = up(x) # 上采样 (b,c,h,w)->(b,c*2,h*2,w*2)
            
            # 当channel序列不是按照2的倍数递增时，需要对x进行padding
            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1) # (b,c,h,w)->(b,2*c,h,w) = (b,pre_c,h,w)
            x = x + pe
            x = torch.cat((x, label_em), dim=1) #  (b,2*c+1,h,w)
            x = decoder(x) # 解码 (b,2*c+1,h,w)->(b,c,h,w)

        x = self.conv_out(x) # (b,c,h,w)->(b,1,h,w)
        return x
