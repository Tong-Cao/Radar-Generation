clc;clear;close all
% 间歇采样直接转发干扰 20230505
Tp = 50e-6;                 %雷达信号脉宽50us
B = 50e6;                   %雷达信号带宽50MHz
mu = B/Tp;                  %调频斜率
%Ts_jam =10e-6;
%Ts_jam =20e-6;              %间歇采样干扰的采样周期
%Ts_jam =0.1e-6;
Ts_jam =25e-6;
%tao_jam = 5e-6;             %间歇采样干扰的采样脉宽5us
tao_jam = 12.5e-6;
fs = 20*B;
Ts = 1/fs;
T_sample =Tp;               %采样时间
N_sample = ceil(T_sample/Ts); %采样点数
N_sample_tao_jam = ceil(tao_jam/Ts); %间歇采样干扰的点数
N_sample_Ts_jam = ceil(Ts_jam/Ts); %采样周期的点数
%% 坐标轴变量
t = linspace(0,T_sample,N_sample);

%% 回波信号
Sig_rec = exp(1i*pi*mu*(t).^2);

%% 间歇采样干扰
    %% 采样脉冲
    Sig_pulse = zeros(1,N_sample);
    N_jam = Tp/Ts_jam;      %干扰采样次数，即采样脉冲个数
    for ii = 1:N_jam
    Sig_pulse( (1+ (ii-1)*N_sample_Ts_jam) :N_sample_tao_jam + (ii-1)*N_sample_Ts_jam) = 1; %两个脉冲
    end
    %% 干扰信号
    Sig_jam = Sig_rec.*Sig_pulse; %回波信号乘以脉冲
    Sig_jam = [Sig_jam(N_sample-N_sample_tao_jam+1:N_sample),Sig_jam(1:N_sample-N_sample_tao_jam)]; % 转发 将Sig_jam往后延时


%% 脉冲压缩
    N_fft = 2*N_sample;         % fft点数
    %% 参考信号
    Sig_ref = exp(1i*pi*mu*(t).^2);
    F_Sig_ref = fft(Sig_ref,N_fft);
    %% 雷达信号的脉冲压缩
    PC_Sig_rec = fftshift( ifft(fft(Sig_rec,N_fft).*(conj(F_Sig_ref))) );
    %1 PC_Sig_rec = PC_Sig_rec(1:N_sample);
    %% 干扰信号的脉冲压缩
    PC_Sig_jam = fftshift( ifft(fft(Sig_jam,N_fft).*(conj(F_Sig_ref))) );
    %2 PC_Sig_jam = PC_Sig_jam(1:N_sample);
%% 画图

testpara_t = linspace(0,2*Tp,2*N_sample);
para_norm = max( abs( real( PC_Sig_rec ) ) );
figure;
plot(testpara_t*1e6,( abs( real( PC_Sig_rec ) )/para_norm ));
%plot(testpara_t*1e6,( abs( real( PC_Sig_rec ) )/para_norm )/max(( abs( real( PC_Sig_rec ) )/para_norm )));
xlabel('时间/us');ylabel('幅度');
%title('雷达回波信号脉冲压缩');
%xlim([96 104]);
%plot(testpara_t*1e6,20*log10( abs( real( PC_Sig_rec ) )/para_norm ));
grid on;
hold on;
plot(testpara_t*1e6,( abs( real( PC_Sig_jam ) )/para_norm ),'r');
%plot(testpara_t*1e6,( abs( real( PC_Sig_jam ) )/para_norm )/max(( abs( real( PC_Sig_jam ) )/para_norm )));
xlabel('时间（us）');ylabel('幅度');
%title('干扰信号脉冲压缩');
%xlim([48 65]);% 占空比0.5
%xlim([48 57]);% 占空比0.2
%plot(testpara_t*1e6,20*log10( abs( real( PC_Sig_jam ) )/para_norm ));
grid on;legend('回波信号','干扰信号');

% figure%画出来的不对
% N = fs*Tp;
% freq = linspace(-B/2, B/2, N);
% plot(freq/1e6, fftshift(abs(fft(Sig_rec))));
% xlabel('频率/MHz');ylabel('幅度');
% grid on; hold on;
% plot(freq/1e6, fftshift(abs(fft(Sig_jam))),'r');
% xlim([-1 4]);
% legend('回波信号','干扰信号');

figure;
plot(t*1e6,real(Sig_rec));
xlabel('时间/us');ylabel('幅度');
title('雷达信号');

figure;
plot(t*1e6,Sig_pulse);
xlabel('时间/us');ylabel('幅度');
title('采样脉冲串');

figure;
plot(t*1e6,real(Sig_jam));
xlabel('时间/us');ylabel('幅度');
title('间歇采样直接转发干扰');

testpara_t = linspace(0,2*Tp,2*N_sample);
para_norm = max( abs( real( PC_Sig_rec ) ) );
figure;
plot(testpara_t*1e6,( abs( real( PC_Sig_rec ) )/para_norm ));
xlabel('时间/us');ylabel('幅度');
title('雷达回波信号脉冲压缩');
%xlim([]);
%plot(testpara_t*1e6,20*log10( abs( real( PC_Sig_rec ) )/para_norm ));
grid on;

figure;
plot(testpara_t*1e6,( abs( real( PC_Sig_jam ) )/para_norm ),'r');
xlabel('时间/us');ylabel('幅度');
title('干扰信号脉冲压缩');
%xlim([]);
%plot(testpara_t*1e6,20*log10( abs( real( PC_Sig_jam ) )/para_norm ));
grid on;

figure;
plot(testpara_t*1e6,( abs( real( PC_Sig_rec ) )/para_norm ),'Linewidth',0.7);
xlabel('时间（us）');ylabel('幅度');
%xlim([95 110]);
hold on;
plot(testpara_t*1e6,( abs( real( PC_Sig_jam ) )/para_norm ),'r','Linewidth',0.7);
grid on;
