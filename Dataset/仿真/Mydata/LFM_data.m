T=5e-6;                            %脉冲宽度 5us
B=40e6;                            %带宽 20M
C=3e8;                             %传播速度
K=B/T;                             %调频斜率
Fs=5*B;Ts=1/Fs;                    %采样频率以及采样时间间隔
Nwid=ceil(T/Ts);                   %LFM信号采样点数  ceil向上取整
t=linspace(-T/2,T/2,Nwid);              % LFM信号序列 600个点

lfm = exp(1j*pi*K*t.^2);           % LFM信号

samp_num = 2048;                     %距离窗点数即信号长度 设置成2048方便后续深度学习网络的运算
data_num = 6000;                     %样本数

realsp=zeros(1,samp_num);%存储矩阵 信号的实部
imagsp=zeros(1,samp_num);%存储矩阵 信号的虚部

%画图 lfm信号时频图
figure;
subplot(211);
plot(t,lfm);
title('LFM信号时域');
xlabel('时间(s)');ylabel('幅值(V)');
subplot(212);
f=linspace(-Fs/2,Fs/2,Nwid);
fft_y=fftshift(fft(lfm));
plot(f,abs(fft_y));
title('LFM信号频域');
xlabel('频率(Hz)');ylabel('幅值(dB)');
saveas(gcf, 'LFM信号时频图', 'png');

%LFM脉冲压缩
N_fft = Nwid;         % fft点数
%% 参考信号
Sig_ref = exp(1j*pi*K*(t).^2);
F_Sig_ref = fft(Sig_ref,N_fft);
%% 雷达信号的脉冲压缩
PC_Sig_rec = fftshift( ifft(fft(lfm,N_fft).*(conj(F_Sig_ref))) );

figure;
plot(t,abs(PC_Sig_rec))
title('LFM信号脉冲压缩结果');
xlabel('时间(s)');ylabel('幅值(dB)');
saveas(gcf, 'LFM信号脉冲压缩结果', 'png');





% %生成数据集
% for m=1:data_num
%     sp=randn([1,samp_num]) + 1j*randn([1,samp_num]);%噪声基底
%     sp=sp/std(sp);
%     range=1+round(rand(1,1)*1400);%LFM信号起始点  LFM信号一共600个点 信号起始点范围应该在(1-1400)
% 
%     %sp(range:length(lfm)+range-1)=sp(range:length(lfm)+range-1) + 10*lfm;  %噪声+目标回波 目标在距离窗内range点处
%     sp(range:length(lfm)+range-1)=10*lfm;% 这里直接将噪声的一段变成纯LFm信号（不和噪声叠加） Lfm的幅值设置为噪声的10倍
%     sp=sp/max(max(sp)); %归一化
% 
%     % 将sp分为实部、虚部保存
%     realsp(1,1:2048)=real(sp);
%     imagsp(1,1:2048)=imag(sp);
%     save(['D:\雷达信号处理\RadarGAN\data\LFM\',num2str(m),'.mat'],'realsp',"imagsp")
% 
% end




% 数据集脉冲压缩(2048长度)
% i=1;
% filename = strcat('D:\雷达信号处理\RadarGAN\data\LFM\',num2str(i),'.mat'); % 读取文件名 num2str(i)将i转换为字符串 
% load(filename);
% 
% figure;
% t1 = linspace(1,2048,2048);
% plot(t1,realsp);
% 
% %脉冲压缩
% N_fft = 2048;         % fft点数
% %% 参考信号
% Sig_ref = exp(1j*pi*K*(t).^2);
% F_Sig_ref = fft(Sig_ref,N_fft);
% %% 雷达信号的脉冲压缩
% PC_Sig_rec = fftshift( ifft(fft(realsp,N_fft).*(conj(F_Sig_ref))) );
% 
% figure;
% plot(t1,abs(PC_Sig_rec))
% title('Pulse compression results for LFM signals');
% saveas(gcf, 'Pulse compression results for LFM signals', 'png');
