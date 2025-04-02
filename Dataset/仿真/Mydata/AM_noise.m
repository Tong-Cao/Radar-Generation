clc;clear;close all
data_num=6000;                     %生成样本数
T=5e-6;                            %脉冲宽度 10us
B=20e6;                            %带宽
C=3e8;                             %传播速度
K=B/T;                             %调频斜率
Fs=5*B;Ts=1/Fs;                    %采样频率以及采样时间间隔
Nwid=ceil(T/Ts);                   %LFM信号采样点数  ceil向上取整
t=linspace(0,T,Nwid);              % LFM信号序列 600个点

lfm = exp(1j*pi*K*t.^2);           % LFM信号

samp_num=2048;                     %距离窗点数即信号长度 设置成2048方便后续深度学习网络的运算
t1=linspace(0,samp_num*Ts,samp_num);  %整个信号的时间长度

realsp=zeros(1,samp_num);%存储矩阵 信号的实部
imagsp=zeros(1,samp_num);%存储矩阵 信号的虚部

%=========================================================
% 噪声调幅干扰
% Un(t)为噪声信号 ωj为载波频率 ϕ为载波相位
% J(t)=[U0+ Un(t)]exp(ωjt+ϕ)
fc = 10e6; %噪声载频
%产生高斯带限白噪声
Un = randn(1,samp_num); % 产生均值为0，方差为0.1的高斯白噪声
fa = 5e6;%滤波的范围（0-fa）
filter1 = fir1(40,fa/(Fs/2)); % 产生带限为fa的低通滤波器  40：点数越高越陡峭
Un = filter(filter1,1,Un); % 通过滤波器
%Un = ifft(fft(Un).*filter);
U0 = 1; % 载波幅度
omega = 2*pi*fc; % 载波频率
phi = 2*pi*rand(1,1); % 载波相位  rand：产生均值为0.5、幅度在0~1之间的伪随机数
J = (U0+Un).*exp(1j*(omega*t1+phi)) ; % 噪声调幅干扰  分布在整个信号时长上

%=========================================================
% 画图
figure;
subplot(211);
plot(t,lfm);
title('LFM信号时域');
subplot(212);
fft_y=fftshift(fft(lfm));
f=linspace(-Fs/2,Fs/2,Nwid);
plot(f,abs(fft_y));
title('LFM信号频谱');

figure;
subplot(211);
plot(t1,J);
title('Noise amplitude modulation interference time domain');
subplot(212);
fft_J=fftshift(fft(J));
f1=linspace(-Fs/2,Fs/2,2048);
plot(f1,abs(fft_J));
title('Noise amplitude modulation interference frequency domain');
saveas(gcf, 'Noise amplitude modulation interference', 'png');

range=1+round(rand(1,1)*1400);%LFM信号起始点  LFM信号一共600个点 信号起始点范围应该在(1-1400)
sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
sp=sp/std(sp);

sp(range:length(lfm)+range-1)=sp(range:length(lfm)+range-1)+ lfm;  %噪声+目标回波 目标在距离窗内range点处
% sp(range:length(lfm)+range-1)=lfm;% 这里直接将噪声的一段变成纯LFm信号（不和噪声叠加） Lfm的幅值设置为噪声的10倍
figure;
subplot(211);
plot(t1,sp);
title('未加入干扰信号时域');
subplot(212);
fft_sp=fftshift(fft(real(sp)));
plot(f1,abs(fft_sp));
title('未加入干扰信号频域');


sp = sp + 1*J; %加上调幅噪声干扰
sp=sp/max(max(sp)); %归一化


figure;
subplot(211);
plot(t1,sp);
title('加入干扰后信号时域');
subplot(212);
fft_sp=fftshift(fft(real(sp)));
plot(f1,abs(fft_sp));
title('加入干扰后信号频域');

%脉冲压缩
N_fft = 2048;         % fft点数

% 参考信号
Sig_ref = exp(1i*pi*K*(t).^2);
F_Sig_ref = fft(Sig_ref,N_fft);
% 雷达信号的脉冲压缩
PC_Sig_rec = fftshift( ifft(fft(sp,N_fft).*(conj(F_Sig_ref))) );

figure;
plot(t1,abs(PC_Sig_rec));
title('脉冲压缩结果');

for m=1:data_num
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
    sp=sp/std(sp);
    range=1+round(rand(1,1)*1400);%LFM信号起始点  LFM信号一共600个点 信号起始点范围应该在(1-1400)

    sp(range:length(lfm)+range-1)=sp(range:length(lfm)+range-1)+1*lfm;  %噪声+目标回波 目标在距离窗内range点处
    %sp(range:length(lfm)+range-1)=lfm;% 这里直接将噪声的一段变成纯LFm信号（不和噪声叠加） Lfm的幅值设置为噪声的10倍
    
    %产生高斯带限白噪声
    Un = randn(1,samp_num); % 产生均值为0，方差为0.1的高斯白噪声
    fa = 5e6;%滤波的范围（0-fa）
    filter1 = fir1(40,fa/(Fs/2)); % 产生带限为fa的低通滤波器  40：点数越高越陡峭
    Un = filter(filter1,1,Un); % 通过滤波器
    %Un = ifft(fft(Un).*filter);
    U0 = 1; % 载波幅度
    omega = 2*pi*fc; % 载波频率
    phi = 2*pi*rand(1,1); % 载波相位  rand：产生均值为0.5、幅度在0~1之间的伪随机数
    J = (U0+Un).*exp(1j*(omega*t1+phi)) ; % 噪声调幅干扰  分布在整个信号时长上

    sp = sp + 1*J; %加上调幅噪声干扰
    sp=sp/max(max(sp)); %归一化

    % 将sp分为实部、虚部保存
    realsp(1,1:2048)=real(sp);
    imagsp(1,1:2048)=imag(sp);
    save(['D:\雷达信号处理\RadarGAN\data\AM_noise\',num2str(m),'.mat'],'realsp',"imagsp")

end