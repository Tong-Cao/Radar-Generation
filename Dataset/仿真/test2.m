T=5e-6;                            %脉冲宽度 10us
B=20e6;                            %带宽
C=3e8;                             %传播速度
K=B/T;                             %调频斜率
Fs=5*B;Ts=1/Fs;                    %采样频率以及采样时间间隔
Nwid=ceil(T/Ts);                   %LFM信号采样点数  ceil向上取整
t=linspace(0,T,Nwid);              % LFM信号序列 600个点

lfm = exp(1j*pi*K*t.^2);           % LFM信号

samp_num=2048;                     %距离窗点数即信号长度 设置成2048方便后续深度学习网络的运算
data_num=6000;                     %样本数



sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
sp=sp/std(sp);
range=1+round(rand(1,1)*1400);%LFM信号起始点  LFM信号一共600个点 信号起始点范围应该在(1-1400)

%sp(range:length(lfm)+range-1)=sp(range:length(lfm)+range-1)+10*lfm;  %噪声+目标回波 目标在距离窗内range点处
sp(range:length(lfm)+range-1)=10*lfm;% 这里直接将噪声的一段变成纯LFm信号（不和噪声叠加） Lfm的幅值设置为噪声的10倍
sp=sp/max(max(sp)); %归一化

figure;
t1 = linspace(1,2048,2048);
plot(t1,sp);

%脉冲压缩
N_fft = 2048;         % fft点数
    %% 参考信号
    Sig_ref = exp(1i*pi*K*(t).^2);
    F_Sig_ref = fft(Sig_ref,N_fft);
    %% 雷达信号的脉冲压缩
    PC_Sig_rec = fftshift( ifft(fft(sp,N_fft).*(conj(F_Sig_ref))) );

    figure;
    plot(t1,PC_Sig_rec)




    
N_fft = 2048;         % fft点数

%% 参考信号
Sig_ref = exp(1i*pi*K*(t).^2);
F_Sig_ref = fft(Sig_ref,N_fft);
%% 雷达信号的脉冲压缩
PC_Sig_rec = fftshift( ifft(fft(lfm,N_fft).*(conj(F_Sig_ref))) );

figure;
plot(t1,PC_Sig_rec);
title('lfm压缩结果');