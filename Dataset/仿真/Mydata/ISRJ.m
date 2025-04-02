clc;clear;close all
data_num=1;                     %生成样本数
T=5e-6;                            %脉冲宽度 5us
B=20e6;                            %带宽
C=3e8;                             %传播速度
K=B/T;                             %调频斜率
Fs=5*B;Ts=1/Fs;                    %采样频率以及采样时间间隔
Nwid=ceil(T/Ts);                   %LFM信号采样点数  ceil向上取整
t=linspace(0,T,Nwid);              % LFM信号序列 500个点
f=linspace(-Fs/2,Fs/2,Nwid);

lfm = exp(1j*pi*K*t.^2);           % LFM信号

samp_num=2048;                     %距离窗点数即信号长度 设置成2048方便后续深度学习网络的运算
t1=linspace(0,samp_num*Ts,samp_num);  %整个信号的时间长度

realsp=zeros(1,samp_num);%存储矩阵 信号的实部
imagsp=zeros(1,samp_num);%存储矩阵 信号的虚部


% figure;
% plot(t,lfm);
% title('LFM信号时域');
% 
% figure;
% plot(t1,sp);
% title('完整信号时域');

%设置干扰的转发次数
repetion_times=[4,3,2,1];%重复次数
period=[5e-6/4,5e-6/2]; %脉冲抽样周期 分别对应4个脉冲 2个脉冲
duty=[20,25,33.33,50]; %功率 （占空比）

for m=1:data_num

    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
    sp=sp/std(sp);
    range=1+round(rand(1,1)*1000);%LFM信号起始点  LFM信号一共500个点 信号起始点范围应该在(1-1400)
    
    sp(range:length(lfm)+range-1)=sp(range:length(lfm)+range-1)+10*lfm;  %噪声+目标回波 目标在距离窗内range点处

    index1=1+round(rand(1,1)); % 1-2
    index2=1+round(rand(1,1)*3); % 1-4
    period1=period(index1); % 随机索引period里的值
    duty1=duty(index2); %设置占空比
    repetion_times1=repetion_times(index2);%重复次数
    
    squa=(square((1/period1)*2*pi*t, duty1)+1)/2;%生成周期为period1占空比为duty1的脉冲信号
    squa(501)=0; % 将最后一个上升沿去掉 
%     figure;
%     plot(t,squa);
%     title('脉冲信号');
    
    sign = lfm .* squa; %采样后的信号
%     figure;
%     plot(t,sign);
%     title('采样信号');
    
    delay_time=period1*duty1*0.01; %延时 即高电平持续时间
    delay_num=ceil(delay_time*Fs); % 延时点数

    for i=1:repetion_times1 %多次转发 将采样后的lfm信号多次转发每次延时delay_time
        
        sp(range+i*delay_num:range+i*delay_num+500)=sp(range+i*delay_num:range+i*delay_num+500)+100*sign;
      
    end
    sp=sp/max(max(sp));

    % 将sp分为实部、虚部保存
    realsp(1,1:2048)=real(sp);
    imagsp(1,1:2048)=imag(sp);
    save(['D:\雷达信号处理\RadarGAN\data\ISRJ\',num2str(m),'.mat'],'realsp',"imagsp")

end

figure;
plot(t1,sp);
title('加入干扰信号');

%脉冲压缩
% N_fft = 2048;         % fft点数

%% 参考信号
% Sig_ref = exp(1i*pi*K*(t).^2);
% F_Sig_ref = fft(Sig_ref,N_fft);
%% 雷达信号的脉冲压缩
% PC_Sig_rec = fftshift( ifft(fft(sp,N_fft).*(conj(F_Sig_ref))) );
% figure;
% plot(t1,abs(PC_Sig_rec),'b');
% hold on
% PC_Sig_rec1 = fftshift( ifft(fft(lfm,N_fft).*(conj(F_Sig_ref))) );
% plot(t1,abs(PC_Sig_rec1),'r');
% hold off
% title('脉冲压缩结果');
