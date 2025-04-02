% fs=100;%采样率
% 
% T=5;%信号时间宽度
% 
% B=10;%信号带宽
% 
% k=B/T;%调频斜率
% 
% n=round(T*fs);%采样点个数
% 
% t=linspace(0, T,n);
% fc = 0;%载频
% 
% y = exp(1j*pi*(k*t.^2));%LFM信号
% 
% 
% figure;
% 
% plot(t,y);
% 
% title('LFM信号时域');
% 
% xlabel('t/s');
% 
% ylabel('幅度');
% 
% 
% fft_y=fftshift(fft(y));
% 
% f=linspace(-fs/2,fs/2,n);
% 
% figure;
% 
% plot(f,abs(fft_y));
% 
% title('LFM信号频谱');
% 
% xlabel('f/Hz');
% 
% ylabel('幅度');


%=========================================================
% 噪声调幅干扰
% Un(t)为噪声信号 ωj为载波频率 ϕ为载波相位
% J(t)=[U0+ Un(t)]cos(ωjt+ϕ)
% fc = 20;
%产生高斯带限白噪声
Un = 0.1*randn(1,n); % 产生均值为0，方差为0.1的高斯白噪声
fa = 10;%滤波的范围（0-fa）
filter1 = fir1(40,fa/(fs/2)); % 产生带限为fa的低通滤波器  40：点数越高越陡峭
Un = filter(filter1,1,Un); % 通过滤波器
%Un = ifft(fft(Un).*filter);
U0 = 0.1; % 载波幅度
omega = 2*pi*fc; % 载波频率
phi = 2*pi*rand(1,n); % 载波相位  rand：产生均值为0.5、幅度在0~1之间的伪随机数
J = (U0+Un).*exp(1j*omega*t+0) +0.1*y; % 噪声调幅干扰


figure;
subplot(211);
plot(t,J);

fft_J=fftshift(fft(J));

f=linspace(-fs/2,fs/2,n);
subplot(212);

plot(f,abs(fft_J)); % 因为J还有调制的过程相位的随机 J的频谱范围和Un不一样

figure;
%y1 = y + J;
y1 = J;
plot(t,y1);

fft_y=fftshift(fft(y1));


figure;
plot(f,abs(fft_y))




% 脉冲压缩
% Nchirp=ceil(T*fs);
% Srw=fft(y1);                           %fft of radar echo
% 
% t0=linspace(-T/2,T/2,Nchirp); 
% St= exp(1j*pi*(k*t.^2));                       %chirp signal                
% Sw=fft(St);                             %fft of chirp signal
% Sot=fftshift(ifft(Srw.*conj(Sw)));              %signal after pulse compression  conj复共轭 fftshift将零点移到中间
% figure;
% plot(t,Sot)
