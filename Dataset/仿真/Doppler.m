T=10e-6;                                  % 脉冲宽度 10us
Prt=300e-6;                               % 脉冲周期 50us
B=30e6;                                   % 频带 30MHz
Rmin=10000;Rmax=15000;                    % 目标物距离范围

% R=[10500,11000,12000,12008,13000,13002];  % 目标起始距离
% Target_speed = [0,0,0,0,0,0]; % 目标朝雷达移动速度 m/s
% RCS=[1 1 1 1 1 1];                        % 目标反射RCS

R= 10500;
Target_speed = 0;
RCS = 1;

%=========================================================
%%参数
C=3e8;                                 % 电磁波速度
fc=0;                                  % 中心频率
K=B/T;                                 % 调频率
plus_num = 4;                         % 脉冲数量
Rwid=Rmax-Rmin;                        % 目标最大距离范围长度
Twid=2*Rwid/C + (plus_num-1)*Prt;      % 接收到回波的时间范围长度 
Fs=5*B;Ts=1/Fs;                        % 采样频率
Nwid=ceil(Twid/Ts);                    % 接收范围内采样点  ceil向上取整

%==================================================================
%%回波     
t=linspace(2*Rmin/C,2*Rmax/C+(plus_num-1)*Prt,Nwid);       % 接收框                           
M=length(R);                                  %number of targets 目标数量
td = ones(M,Nwid,plus_num);                   % 创建延时矩阵 M x Nwid x plus_num
Srt = ones(1,Nwid,plus_num);                  % 回波矩阵 1 x Nwid x plus_num
s_p = ones(plus_num,Nwid);

for i = 1:plus_num                            %分别计算第1到第plus_num个脉冲的回波数据
    td(:,:,i) = (((C+Target_speed)./(C-Target_speed))'*ones(1,Nwid)).*(ones(M,1)*t-2*R'./(Target_speed+C)'*ones(1,Nwid))-((i-1)*Prt)*ones(M,Nwid);
    Srt(:,:,i)=RCS*(exp(1j*pi*K*td(:,:,i).^2).*(abs(td(:,:,i))<T/2));    %  雷达回波信号矩阵 
    
    %figure;
    %=========================================================
    %%Digtal processing of pulse compression radar using FFT and IFFT
    Nchirp=ceil(T/Ts);                          %pulse duration in number 脉冲宽度内采样点数
    Nfft=2^nextpow2(Nwid+Nwid-1);             %number needed to compute linear
    % x = nextpow2(p)  2^(x-1) <= p <= 2^x
                                             %convolution using FFT algorithm
    Srw=fft(Srt(:,:,i),Nfft);                           %fft of radar echo
    t0=linspace(-T/2,T/2,Nchirp); 
    St=exp(1j*pi*K*t0.^2);                       %chirp signal                
    Sw=fft(St,Nfft);                             %fft of chirp signal
    Sot=fftshift(ifft(Srw.*conj(Sw)));              %signal after pulse compression  conj复共轭 fftshift将零点移到中间
    
    %=========================================================
    N0=Nfft/2-Nchirp/2;     % 相当于将Sot的零点移到中间即Nfft/2，St翻转开始卷积从-Nchirp/2开始产生结果，所以从Nfft/2-Nchirp/2开始为有效数据
    Z=abs(Sot(N0:N0+Nwid-1));
    Z=Z/max(Z);
    Z=20*log10(Z+1e-6);
    s_p(i,:) = Z;
    figure
    subplot(211)
    plot(t*1e6,real(Srt(:,:,i)));axis tight;
    xlabel('Time in u sec');ylabel('Amplitude')
    title('Radar echo without compression');
    subplot(212)
    plot((t-(i-1)*Prt)*C/2,s_p(i,:))
    axis([10000,15000,-60,0]);
    xlabel('Range in meters');ylabel('Amplitude in dB')
    title('Radar echo after compression');
    %=========================================================
    
end



