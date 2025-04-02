%=========================================================
%噪声调频干扰
T=5e-6;                            %脉冲宽度 10us
B=20e6;                            %带宽
C=3e8;                             %传播速度
K=B/T;                             %调频斜率
Fs=5*B;Ts=1/Fs;                    %采样频率以及采样时间间隔
Nwid=ceil(T/Ts);                   %LFM信号采样点数  ceil向上取整
t=linspace(0,T,Nwid);              % LFM信号序列 600个点
f=linspace(-Fs/2,Fs/2,Nwid);

lfm = exp(1j*pi*K*t.^2);           % LFM信号

fc = 10e6;%载频
K = 11e6; % 调频斜率 将噪声从0-1的范围调制到fc
Un = randn(1,Nwid); % 产生均值为0，方差为0.1的高斯白噪声
fa = 10;%滤波的范围（0-fa）
filter1 = fir1(40,fa/(Fs/2)); % 产生带限为fa的低通滤波器  40：点数越高越陡峭
Un = filter(filter1,1,Un); % 通过滤波器

yn = zeros(1,Nwid); % yn作为un的积分结果
for i=1:Nwid-1
    yn(i+1)=Un(i)+yn(i);
end

phi = 2*pi*rand(1,1); % 载波相位  rand：产生均值为0.5、幅度在0~1之间的伪随机数
S=exp(1i*(2*pi*fc.*t + 2*pi*K.*yn*Ts + phi)); % 按照公式给出噪声  这里注意yn要乘以Ts
% 按照公式求导S的频率应该为 fc+un 
% 由于t的每一格时间长为Ts  所以fc.*t结果为[0,Ts*fc,2*Ts*fc,,]此时求导也就是做差 得到的值为Ts*fc而非fc
% 同理由于矩阵的单位时长并非为1 如果直接加 2*pi*K.*yn 对其作差因当为t = 0，1，2时的结果而非t= 0，Ts，2Ts，，的结果
% 所以这里必须要改成 2*pi*K.*yn*Ts
fft_s=fftshift(fft(S));

figure;
subplot(211);
plot(t,S);
subplot(212);
plot(f,abs(fft_s));