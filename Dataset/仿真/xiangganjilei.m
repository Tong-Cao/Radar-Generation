%信号带宽B：1MHz，中心频率fo：1MHz，采样频率fs：4MHz
clear all 
close all
clc 
c=3e8;                              %光速
Te=100e-6;                          %发射脉冲宽度
Be=1e6;                             %带宽
K=Be/Te;                            %调频斜率
fs=(4*Be);
Ts=1/fs;                            % fs采样频率 Ts采样周期
Ro=50e3;                            % 起始距离
fo=1e6;                             % 中心频率
Vr=688;                             %径向速度
Nw=Te/Ts;
t=linspace(-Te/2,Te/2,400);         % Nw = 400
t1=linspace(-Te/2,Te/2,100);
W=exp(j*pi*K*t1.^2);                %匹配信号
Wf=fft(W,1024);
% figure;plot(1:1:1024,abs(Wf));
nnn=fix(2*(Ro-30e3)/75);            %采样的起始位置,从30km开始采样
R=0:75/2:15e3-75/2;                 %在30km和45km之间采样，采样间隔75/2m，400个点
for i=1:400
    for k=1 :64                     % 64个回波
        Ri(k,i)=R(i)-Vr*Ts*(k-1);   % 相当于64个脉冲 脉冲间隔设置为Ts
    end
end
taoi=2*Ri/c-Te/2;
for i=1:64
    s(:,i)=cos(2*pi*fo*taoi(i,:)+pi*K*taoi(i,:).^2);
end
S=fftshift(fft(s(:,1)));
figure;
subplot(2,1,1);plot(t,s(:,1));
xlabel('t(us)');ylabel('幅度');title('LFM信号的实部');
subplot(2,1,2);plot((0:fs/400:fs/2-fs/400),abs(S(1:400/2)));
xlabel('f(Hz)');ylabel('幅度');title('LFM频谱');
%%%%%%%%%%%%%%%%%%%%%%%%正交解调%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ns=400;M=64
n=0:1:Ns-1;
De_i=cos(n*fo/fs*2*pi);               %I路本振信号
De_q=sin(n*fo/fs*2*pi);               %Q路本振信号
De_i=repmat(De_i,1,M);
De_q=repmat(De_q,1,M);
De_i=reshape(De_i,Ns,M);
De_q=reshape(De_q,Ns,M);
si=s.*De_i;                           %I解调后信号
sq=s.*De_q;                           %Q解调后信号
window=chebwin(12,50);                %窗函数
[b a]=fir1(11,2*Be/fs,window);        %得到滤波器的参数
si_de=[];sq_de=[];
for i=1:1:M
    temp1=filter(b,a,si(:,i));        %解调I路信号
    temp2=filter(b,a,sq(:,i));        %解调Q路信号
    si_de=[si_de 2*temp1];
    sq_de=[sq_de 2*temp2];
end
s_de=si_de+j*sq_de;                   %解调后幅信号
S_de=fftshift(fft(s_de));             %幅信号的频谱
temp=1:1:Ns/4;                        
s_chouqu=s_de(4*temp,:);              %做1/4抽取
S_chouqu=fftshift(fft(s_chouqu));     %抽取的频谱带宽变低为10MHz
%正交解调输出
figure;
subplot(2,1,1),plot(t,si_de(:,1));
xlabel('t(us)');ylabel('幅度');title('解调后I路信号');
axis([-5e-5 5e-5 -1.2 1.2]);
subplot(2,1,2),plot(t,sq_de(:,1));
xlabel('t(us)');ylabel('幅度');title('解调后Q路信号');
axis([-5e-5 5e-5 -1.2 1.2])
figure;
plot((-fs/2:fs/Ns:fs/2-fs/Ns),abs(S_de(:,1)));
xlabel('f(Hz)');ylabel('幅度');title('解调后信号的频谱');
figure;
plot((-fo/2:4*fo/Ns:fo/2-4*fo/Ns),abs(S_chouqu(:,1)));
xlabel('f(Hz)');ylabel('幅度');title('1/4抽取后信号的频谱');

noise=10^0.225*0.707*(randn(64,1024)+j*randn(64,1024));
s_noise=noise;
for i=1:64                              %回波信号
  s_noise(i,nnn:nnn+99)=noise(i,nnn:nnn+99)+...
      s_chouqu(:,i)';
end
for i=1:64                              %脉冲压缩
     sp2(i,:)=ifft(fft(s_noise(i,:),1024).*conj(Wf),1024);
end
figure;plot(1:1:1024,abs(sp2(1,:)));
axis([1 1024 0 120])
 for k=1:1024                           % 相干积累
     sct(:,k)=abs(fftshift(fft(sp2(:,k),256)));
 end
 sct=sct./max(max(sct));                %归一化 
 sp=sp2./max(max(sp2));                 %归一化
 %积累前后信噪比输出
 figure
 plot(20*log10(abs(sp')))
 ylabel('-db')
 title('相干积累前')
 axis([1  1024 -30 0])
 figure
 plot(20*log10(sct'))
 ylabel(' - db')
 title('相干积累输出')
 axis([1  1024 -30 0])
%积累前后结果输出
r=((1:1024)*75/2+30e3)./1e3;
dp=(-128:127)*(Be/128)/1e3;
figure
mesh(r,dp,sct)
xlabel('距离 km')
ylabel('Doppler -  kHz')
title('相干积累输出结果')
figure
contour(r,dp,sct)
axis([30 100 -200 200])
xlabel('距离 km')
ylabel('Doppler -  kHz')
title('R-fd 等高线')
grid on
dp=(-32:31)*(Be/32)/1e3;
figure
mesh(r,dp,abs(s_noise)/max(max(abs(s_noise))))
xlabel('距离 km')
ylabel('Doppler -  kHz')
title('相干积累前的结果')