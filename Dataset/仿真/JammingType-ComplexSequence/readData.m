clc;
clear;
B = 10e6; %带宽
T = 20e-6; %脉冲宽度
f0 = 0; %中心频率 
N = round(T/(1/fs)); %采样点数
t = linspace(-0.5*T, 0.5*T, N); %时域采样点
k = B/T; %调频率
h = exp(-1i*pi*k*t.^2); %匹配滤波器
H = fft(h,NN); %匹配滤波器频域
tf = -fs/2:fs/N:fs/2-fs/N;  %频域采样点  从-fs/2到fs/2-fs/N，间隔为fs/N
for i = 1:500
    filename = strcat('./Dense_False_Target_Jam/',num2str(i),'.mat'); % 读取文件名 num2str(i)将i转换为字符串 
    load(filename);
    X = fft(complexsp, NN);
    y = ifft(H.*X);
    save(['./Jaming/',num2str(i),'.mat'],'','');
end
    
   
    
    