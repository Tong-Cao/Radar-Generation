%% 
%--------------------噪声调幅干扰------------------
%信号长度
N=10000;
n=1:1:N;
%系统采样率
f_s=100e6;
T_s=1/f_s;
B_n=2e6;%射频信号要产生的带宽
f_0=10e6;
m_A=1;%有效调制系数
% 调制噪声参数
delta_F=B_n/2;%射频信号要产生的带宽B_n时，噪声要产生的带宽
N_n=round(N*(B_n/(2*f_s)));%频谱上采样点数
S_n=random('Normal',0,1,1,N)+1i*random('Normal',0,1,1,N);%高斯白噪声频谱,中值0，标准差1，维度(1，N)
for i=N_n:1:N-N_n-1
    S_n(i)=0;
end
s_n=ifft(S_n);
s_n=s_n/std(s_n,0,2);%归一化

%产生需要调制的射频噪声干扰s_0
s_0=exp(1j*2*pi*f_0*n*T_s);
A=std(s_n,0,2)/m_A;
s=(A+real(s_n)).*cos(2*pi*f_0*n*T_s);

figure(1);
subplot(3,1,1);
plot(n/f_s,s_0);
xlabel('us');title('原始信号');
subplot(3,1,2);
plot(n/f_s,real(s));
xlabel('us');title('噪声调幅干扰时域波形');
%幅度谱
subplot(3,1,3);
k = -(N-1)/2:(N-1)/2;
f = k/N*f_s;
plot(f,abs(fftshift(fft(s)))/N);
xlabel('hz');title('噪声调幅干扰功率谱');

figure;
plot(n/f_s,s_0);