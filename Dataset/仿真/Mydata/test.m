clc;clear;close all
i=111;
filename = strcat('D:\雷达信号处理\RadarGAN\data\FM_noise\',num2str(i),'.mat'); % 读取文件名 num2str(i)将i转换为字符串 
load(filename);




T=5e-6;                            %脉冲宽度 10us
B=20e6;                            %带宽
C=3e8;                             %传播速度
K=B/T;                             %调频斜率
Fs=5*B;Ts=1/Fs;                    %采样频率以及采样时间间隔
Nwid=ceil(T/Ts);                   %LFM信号采样点数  ceil向上取整

t=linspace(0,T,Nwid);              % LFM信号序列 600个点

samp_num=2048; 
t1=linspace(0,samp_num*Ts,samp_num);
f1=linspace(-Fs/2,Fs/2,2048);

figure;
subplot(211);
plot(t1,realsp);
title('LFM信号时域');
subplot(212);
fft_y=fftshift(fft(realsp));
plot(f1,abs(fft_y));
title('LFM信号频谱');

% %脉冲压缩
N_fft = 2048;         % fft点数

% 参考信号
Sig_ref = exp(1i*pi*K*(t).^2);
F_Sig_ref = fft(Sig_ref,N_fft);
% 雷达信号的脉冲压缩
PC_Sig_rec = fftshift( ifft(fft(realsp,N_fft).*(conj(F_Sig_ref))) );

figure;
plot(t1,PC_Sig_rec);
title('脉冲压缩结果');