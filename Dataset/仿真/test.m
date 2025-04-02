fs=100;

T=5;

B=10;

k=B/T;%调频斜率

n=round(T*fs);%采样点个数

t=linspace(0, T,n);

%y=exp(1j*pi*k*t.^2);%LFM信号
y = exp(1j*pi*2*t.*t);

figure;
 
 plot(t,y);
 
 title('LFM信号时域');
 
 xlabel('t/s');
 
 ylabel('幅度');
 
 
 fft_y=fftshift(fft(y));
 
 f=linspace(-fs/2,fs/2,n);
 
 figure;
 
 plot(f,abs(fft_y));
 
 title('LFM信号频谱');
 
 xlabel('f/Hz');
 
 ylabel('幅度');


R = 100000000;
V = 5000;
C = 3e8;
t1 = (t-2*R/(V+C))*(V+C)/(C-V);
s_1 =  exp(1j*pi*2*t1.*t1);
figure;
plot(t,s_1);


 fft_s_1=fftshift(fft(s_1));
 
 f=linspace(-fs/2,fs/2,n);
 
 figure;
 
 plot(f,abs(fft_s_1));

