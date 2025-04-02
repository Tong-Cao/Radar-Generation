clc;clear;close all

font_size = 14; %字体大小
t1 = linspace(1,2048,2048); % 数据集数据长度为2048

% 查看数据集

% LFM
i=4;
filename = strcat('D:\雷达信号处理\RadarGAN\data\LFM\',num2str(i),'.mat'); % 读取文件名 num2str(i)将i转换为字符串 
load(filename);

figure('Position', [100, 100, 2000, 350]);
subplot(141);
plot(t1,realsp);
title('Time domain of LFM signal','FontSize', font_size);
xlabel('Time(s)','FontSize', font_size);ylabel('Amplitude(V)','FontSize', font_size);

% AM_noise
filename = strcat('D:\雷达信号处理\RadarGAN\data\AM_noise\',num2str(i),'.mat'); % 读取文件名 num2str(i)将i转换为字符串 
load(filename);

subplot(142);
plot(t1,realsp);
title('Time Domain of Noise AM jamming','FontSize', font_size);
xlabel('Time(s)','FontSize', font_size);ylabel('Amplitude(V)','FontSize', font_size);

% FM_noise
filename = strcat('D:\雷达信号处理\RadarGAN\data\FM_noise\',num2str(i),'.mat'); % 读取文件名 num2str(i)将i转换为字符串 
load(filename);

subplot(143);
plot(t1,realsp);
title('Time Domain of Noise FM jamming','FontSize', font_size);
xlabel('Time(s)','FontSize', font_size);ylabel('Amplitude(V)','FontSize', font_size);

% ISRJ
filename = strcat('D:\雷达信号处理\RadarGAN\data\ISRJ\',num2str(i),'.mat'); % 读取文件名 num2str(i)将i转换为字符串 
load(filename);

subplot(144);
plot(t1,realsp);
title('Time Domain of ISRJ','FontSize', font_size);
xlabel('Time(s)','FontSize', font_size);ylabel('Amplitude(V)','FontSize', font_size);

% 保存图片
saveas(gcf, 'data_set', 'png');
    

