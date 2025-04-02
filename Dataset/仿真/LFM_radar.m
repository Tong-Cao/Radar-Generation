function LFM_radar(T,B,Rmin,Rmax,R,RCS)
if nargin==0
    T=10e-6;                           %pulse duration 10us
    B=30e6;                            %chirp frequency modulation bandwidth 30MHz
    Rmin=10000;Rmax=15000;            %range bin
    R=[10500,11000,12000,12008,13000,13002]; %position of ideal point targets
    %R = [10500];
    RCS=[1 1 1 1 1 1]; 
    %radar cross section
    %RCS=[10];
end
%=========================================================
%%Parameter
C=3e8;                                 %propagation speed
K=B/T;                                 %chirp slope
Rwid=Rmax-Rmin;                        %目标最大距离范围长度
Twid=2*Rwid/C;                          %接收到回波的时间范围长度 
Fs=5*B;Ts=1/Fs;                         %sampling frequency and sampling spacing
Nwid=ceil(Twid/Ts);                       %receive window in number  ceil向上取整

%==================================================================
%%Gnerate the echo      
t=linspace(2*Rmin/C,2*Rmax/C,Nwid);       %receive window
                                       %open window when t=2*Rmin/C
                                       %close window when t=2*Rmax/C                            
M=length(R);                            %number of targets 目标数量                                        
td=ones(M,1)*t-2*R'/C*ones(1,Nwid);     % 时间延迟矩阵 创造各个目标的延迟时间矩阵
% 第一行为第一个目标的延迟时间，数值从0到(2*Rmax/C-2*R1/C) 例如 0，1，2，3，4
% 第二行为第二个目标的延迟时间，数值从-(2*R2/C-2*R1/C)到(2*Rmax/C-2*R2/C) 例如 -2，-1，0，1，2 
Srt=RCS*(exp(1j*pi*K*td.^2).*(abs(td)<T/2));    %radar echo from point targets  雷达回波信号矩阵 （abs(td)<T/2）信号有脉冲宽度，超过T/2的信号为0

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