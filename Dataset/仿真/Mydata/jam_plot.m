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

font_size = 10; %字体大小

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
figure('Position', [100, 100, 1200, 600]);
subplot(231);
plot(t1,J);
title('Time Domain of Noise AM jamming','FontSize', font_size);
xlabel('Time(s)');ylabel('Amplitude(V)');
subplot(234);
fft_J=fftshift(fft(J));
f1=linspace(-Fs/2,Fs/2,2048);
plot(f1,abs(fft_J));
title('Frequency Domain of Noise AM jamming','FontSize', font_size);
xlabel('Frequency(Hz)');ylabel('Amplitude(dB)');


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

subplot(232);
plot(t1,J);
title('Time Domain of Noise FM jamming','FontSize', font_size);
xlabel('Time(s)');ylabel('Amplitude(V)');
subplot(235);
fft_J=fftshift(fft(J));
f1=linspace(-Fs/2,Fs/2,2048);
plot(f1,abs(fft_J));
title('Frequency Domain of Noise FM jamming','FontSize', font_size);
xlabel('Frequency(Hz)');ylabel('Amplitude(dB)');

SNR=0; %信噪比dB

repetion_times=[4,3,2,1];%重复次数
period=[5e-6/4,5e-6/2]; %脉冲抽样周期 分别对应4个脉冲 2个脉冲
duty=[20,33.33,50,80]; %功率 （占空比）
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
    sp=sp/std(sp);
    range=1+round(rand(1,1)*1000);%LFM信号起始点  LFM信号一共500个点 信号起始点范围应该在(1-1400)
    
    sp(range:length(lfm)+range-1)=sp(range:length(lfm)+range-1);  %噪声+目标回波 目标在距离窗内range点处

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
        
        sp(range+i*delay_num:range+i*delay_num+500)=sp(range+i*delay_num:range+i*delay_num+500)+10*sign;
      
    end
    sp=sp/max(max(sp));
    

    subplot(233);
    plot(t1,sp);
    title('Time Domain of ISRJ','FontSize', font_size);
xlabel('Time(s)');ylabel('Amplitude(V)');
    subplot(236);   
fft_J=fftshift(fft(sp));
f1=linspace(-Fs/2,Fs/2,2048);
plot(f1,abs(fft_J));
title('Frequency Domain of ISRJ','FontSize', font_size);
xlabel('Frequency(Hz)');ylabel('Amplitude(dB)');


saveas(gcf, '3jam', 'png');
    

