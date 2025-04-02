clc;clear;close all
data_num=1;                     %生成样本数
T=5e-6;                            %脉冲宽度 10us
B=20e6;                            %带宽
C=3e8;                             %传播速度
K=B/T;                             %调频斜率
Fs=5*B;Ts=1/Fs;                    %采样频率以及采样时间间隔
Nwid=ceil(T/Ts);                   %LFM信号采样点数  ceil向上取整
t=linspace(0,T,Nwid);              % LFM信号序列 600个点
f=linspace(-Fs/2,Fs/2,Nwid);

lfm = exp(1j*pi*K*t.^2);           % LFM信号

samp_num=2048;                     %距离窗点数即信号长度 设置成2048方便后续深度学习网络的运算
t1=linspace(0,samp_num*Ts,samp_num);  %整个信号的时间长度

realsp=zeros(1,samp_num);%存储矩阵 信号的实部
imagsp=zeros(1,samp_num);%存储矩阵 信号的虚部

%=========================================================
% 噪声调频干扰
% J(t)=[U0]exp(ωjt+2pi*K*（un的积分）+ϕ)
fc = 10e6;%载频
K1 = 10e6; % 调频斜率 将噪声从0-1的范围调制到fc
Un = randn(1,samp_num); % 产生均值为0，方差为0.1的高斯白噪声
fa = 10;%滤波的范围（0-fa）
filter1 = fir1(40,fa/(Fs/2)); % 产生带限为fa的低通滤波器  40：点数越高越陡峭
Un = filter(filter1,1,Un); % 通过滤波器

yn = zeros(1,samp_num); % yn作为un的积分结果
for i=1:samp_num-1
    yn(i+1)=Un(i)+yn(i);
end

phi = 2*pi*rand(1,1); % 载波相位  rand：产生均值为0.5、幅度在0~1之间的伪随机数
J=exp(1i*(2*pi*fc.*t1 + 2*pi*K1.*yn*Ts + phi)); % 按照公式给出噪声  这里注意yn要乘以Ts
% 按照公式求导S的频率应该为 fc+un 
% 由于t的每一格时间长为Ts  所以fc.*t结果为[0,Ts*fc,2*Ts*fc,,]此时求导也就是做差 得到的值为Ts*fc而非fc
% 同理由于矩阵的单位时长并非为1 如果直接加 2*pi*K.*yn 对其作差因当为t = 0，1，2时的结果而非t= 0，Ts，2Ts，，的结果
% 所以这里必须要改成 2*pi*K.*yn*Ts




%=========================================================
% 画图
figure;
subplot(211);
plot(t,lfm);
title('LFM信号时域');
subplot(212);
fft_y=fftshift(fft(lfm));
plot(f,abs(fft_y));
title('LFM信号频谱');

%脉冲压缩
N_fft = 2048;         % fft点数

%% 参考信号
Sig_ref = exp(1i*pi*K*(t).^2);
F_Sig_ref = fft(Sig_ref,N_fft);
%% 雷达信号的脉冲压缩
PC_Sig_rec = fftshift( ifft(fft(lfm,N_fft).*(conj(F_Sig_ref))) );

figure;
plot(t1,PC_Sig_rec);
title('lfm压缩结果');

figure;
subplot(211);
plot(t1,J);
title('噪声信号时域');
subplot(212);
fft_J=fftshift(fft(J));
f1=linspace(-Fs/2,Fs/2,2048);
plot(f1,abs(fft_J));
title('噪声信号频谱');


range=1+round(rand(1,1)*1400);%LFM信号起始点  LFM信号一共600个点 信号起始点范围应该在(1-1400)
sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
sp=sp/std(sp);

sp(range:length(lfm)+range-1)=sp(range:length(lfm)+range-1)+lfm;  %噪声+目标回波 目标在距离窗内range点处
% sp(range:length(lfm)+range-1)=lfm;
sp=sp/max(max(sp)); %归一化
figure;
subplot(211);
plot(t1,sp);
title('归一化信号时域');
subplot(212);
fft_sp=fftshift(fft(sp));

plot(f1,abs(fft_sp));
title('归一化信号频域');


%脉冲压缩
N_fft = 2048;         % fft点数

%% 参考信号
Sig_ref = exp(1i*pi*K*(t).^2);
F_Sig_ref = fft(Sig_ref,N_fft);
%% 雷达信号的脉冲压缩
PC_Sig_rec = fftshift( ifft(fft(sp,N_fft).*(conj(F_Sig_ref))) );

figure;
plot(t1,PC_Sig_rec);
title('未加干扰脉冲压缩结果');

%加干扰
sp = sp + 1*J; %
sp=sp/max(max(sp)); %归一化
figure;
subplot(211);
plot(t1,sp);
title('加入干扰后信号时域');
subplot(212);
fft_sp=fftshift(fft(sp));
plot(f1,abs(fft_sp));
title('加入干扰后信号频域');

%脉冲压缩
N_fft = 2048;         % fft点数

%% 参考信号
Sig_ref = exp(1i*pi*K*(t).^2);
F_Sig_ref = fft(Sig_ref,N_fft);
%% 雷达信号的脉冲压缩
PC_Sig_rec = fftshift( ifft(fft(sp,N_fft).*(conj(F_Sig_ref))) );

figure;
plot(t1,PC_Sig_rec);
title('脉冲压缩结果');


%=========================================================
%生成信号
% for m=1:data_num
%     sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
%     sp=sp/std(sp);
%     range=1+round(rand(1,1)*1400);%LFM信号起始点  LFM信号一共600个点 信号起始点范围应该在(1-1400)
% 
%     sp(range:length(lfm)+range-1)=sp(range:length(lfm)+range-1)+1*lfm;  %噪声+目标回波 目标在距离窗内range点处
%     %sp(range:length(lfm)+range-1)=10*lfm;% 这里直接将噪声的一段变成纯LFm信号（不和噪声叠加） Lfm的幅值设置为噪声的10倍
%     sp=sp/max(max(sp)); %归一化
% 
%     fc = 10e6;%载频
%     K1 = 10e6; % 调频斜率 将噪声从0-1的范围调制到fc
%     Un = randn(1,samp_num); % 产生均值为0，方差为0.1的高斯白噪声
%     fa = 10;%滤波的范围（0-fa）
%     filter1 = fir1(40,fa/(Fs/2)); % 产生带限为fa的低通滤波器  40：点数越高越陡峭
%     Un = filter(filter1,1,Un); % 通过滤波器
%     
%     yn = zeros(1,samp_num); % yn作为un的积分结果
%     for i=1:samp_num-1
%         yn(i+1)=Un(i)+yn(i);
%     end
%     
%     phi = 2*pi*rand(1,1); % 载波相位  rand：产生均值为0.5、幅度在0~1之间的伪随机数
%     J=exp(1i*(2*pi*fc.*t1 + 2*pi*K1.*yn*Ts + phi));
% 
%     sp = sp + 1*J; %加上调幅噪声干扰
%     sp=sp/max(max(sp)); %归一化
% 
% 
%     % 将sp分为实部、虚部保存
%     realsp(1,1:2048)=real(sp);
%     imagsp(1,1:2048)=imag(sp);
%     save(['D:\雷达信号处理\RadarGAN\data\FM_noise\',num2str(m),'.mat'],'realsp',"imagsp")
% 
% end

