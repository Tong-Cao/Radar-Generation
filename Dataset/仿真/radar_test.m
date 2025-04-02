T=10e-6;                                  % 脉冲宽度 10us
Prt=100e-6;                               % 脉冲周期 500us
B=30e6;                                   % 频带 30MHz
Rmin=10000;Rmax=15000;                    % 目标物距离范围
R=[10500,11000,12000,12008,13000,13002];  % 目标起始距离
Target_speed = [300,300,300,300,300,300]; % 目标朝雷达移动速度 m/s
RCS=[1 1 1 1 1 1];                        %目标反射RCS


%=========================================================
%%参数
C=3e8;                                 % 电磁波速度
fc=0;                                  % 中心频率
K=B/T;                                 % 调频率
Rwid=Rmax-Rmin;                        % 目标最大距离范围长度
Twid=2*Rwid/C + Prt;                   % 接收到回波的时间范围长度 
Fs=5*B;Ts=1/Fs;                        % 采样频率
Nwid=ceil(Twid/Ts);                    % 接收范围内采样点  ceil向上取整

%==================================================================
%%回波     
t=linspace(2*Rmin/C,2*Rmax/C+Prt,Nwid);       % 接收框                           
M=length(R);                            %number of targets 目标数量                                        
td1=ones(M,1)*t-2*R'./(Target_speed+C)'*ones(1,Nwid);     % 时间延迟矩阵 创造各个目标的延迟时间矩阵
% ones(M,1)*t- 为M x Nwid的矩阵，每一行都是从开始接收时间到结束接受时间的序列
% 2*R'./(Target_speed+C)为接收每一个目标的回波的时间 再*ones(1,Nwid)结果为M*Nwid矩阵每一行为不同目标的延迟时间
% 第一行为第一个目标的延迟时间，数值从0到(2*Rmax/C-2*R1/C) 例如 0，1，2，3，4
% 第二行为第二个目标的延迟时间，数值从-(2*R2/C-2*R1/C)到(2*Rmax/C-2*R2/C) 例如 -2，-1，0，1，2 

%第二个脉冲回波延迟序列
td2 = ones(M,1)*t-(2*(R-Prt*Target_speed)'./(Target_speed+C)'*ones(1,Nwid)+Prt); 
%R-Prt*Target_speed为第二个脉冲发出时目标的距离

td = [td1;td2]; % 将td1和td2按列拼接

Srt=[RCS,RCS]*(exp(1j*pi*K*td.^2).*(abs(td)<T/2));    %  雷达回波信号矩阵 
% （abs(td)<T/2）信号有脉冲宽度，超过T/2的信号为0
%  td为两个矩阵拼接 2M x Nwid 需要将RCS同样复制一倍大小(按行拼接)为1 x 2M

%=========================================================
%%Digtal processing of pulse compression radar using FFT and IFFT
Nchirp=ceil(T/Ts);                          %pulse duration in number 脉冲宽度内采样点数
Nfft=2^nextpow2(Nwid+Nwid-1);             %number needed to compute linear
% x = nextpow2(p)  2^(x-1) <= p <= 2^x
                                         %convolution using FFT algorithm
Srw=fft(Srt,Nfft);                           %fft of radar echo
t0=linspace(-T/2,T/2,Nchirp); 
St=exp(1j*pi*K*t0.^2);                       %chirp signal                
Sw=fft(St,Nfft);                             %fft of chirp signal
Sot=fftshift(ifft(Srw.*conj(Sw)));              %signal after pulse compression  conj复共轭 fftshift将零点移到中间

%=========================================================
N0=Nfft/2-Nchirp/2;     % 相当于将Sot的零点移到中间即Nfft/2，St翻转开始卷积从-Nchirp/2开始产生结果，所以从Nfft/2-Nchirp/2开始为有效数据
Z=abs(Sot(N0:N0+Nwid-1));
Z=Z/max(Z);
Z=20*log10(Z+1e-6);
%figure
subplot(211)
plot(t*1e6,real(Srt));axis tight;
xlabel('Time in u sec');ylabel('Amplitude')
title('Radar echo without compression');
subplot(212)
plot(t*C/2,Z)
axis([10000,15000,-60,0]);
xlabel('Range in meters');ylabel('Amplitude in dB')
title('Radar echo after compression');
%=========================================================

%=========================================================
%噪声调幅干扰
% Un(t)为噪声信号 ωj为载波频率 ϕ为载波相位
% J(t)=[U0+ Un(t)]cos(ωjt+ϕ)
Un = 0.1*randn(1,Nwid); % 产生均值为0，方差为0.1的高斯白噪声
U0 = 1; % 载波幅度
omega = 2*pi*fc; % 载波频率
%产生（0，2pi）均匀分布的载波相位
phi = 2*pi*rand(1,Nwid); % 载波相位  rand：产生均值为0.5、幅度在0~1之间的伪随机数
J = (U0+Un).*cos(omega*t+phi); % 噪声调幅干扰



