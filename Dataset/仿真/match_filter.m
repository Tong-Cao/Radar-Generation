T=10e-6;                            %脉冲宽度 10us
B=30e6;                             %信号带宽 30MHz
K=B/T;                              %调频斜率
Fs=10*B;Ts=1/Fs;                    %采样频率FS
N=T/Ts;                             %采样点数
t=linspace(-T/2,T/2,N);             % 时间序列
St=exp(j*pi*K*t.^2);                   %LFM信号
Ht=exp(-j*pi*K*t.^2);                  %匹配滤波冲激响应
Sot=conv(St,Ht);                       %卷积运算 得到匹配滤波结果
subplot(211)
L=2*N-1;
t1=linspace(-T,T,L);
Z=abs(Sot);Z=Z/max(Z);                %归一化
Z=20*log10(Z+1e-6);
Z1=abs(sinc(B.*t1));                   %sinc函数
Z1=20*log10(Z1+1e-6);
t1=t1*B;                                         
plot(t1,Z,t1,Z1,'r.');
axis([-15,15,-50,inf]);grid on;
legend('emulational','sinc');
xlabel('Time in sec \times\itB');
ylabel('Amplitude,dB');
title('Chirp signal after matched filter');
subplot(212)                          %zoom
N0=3*Fs/B;
t2=-N0*Ts:Ts:N0*Ts;
t2=B*t2;
plot(t2,Z(N-N0:N+N0),t2,Z1(N-N0:N+N0),'r.');
axis([-inf,inf,-50,inf]);grid on;
set(gca,'Ytick',[-13.4,-4,0],'Xtick',[-3,-2,-1,-0.5,0,0.5,1,2,3]);
xlabel('Time in sec \times\itB');
ylabel('Amplitude,dB');
title('Chirp signal after matched filter (Zoom)');