filename = strcat('./ISRJ/','10','.mat'); % 读取文件名 num2str(i)将i转换为字符串 
load(filename);
% t = linspace(1,2000,2000);
% figure;
% plot(t,realsp)
% 
% 信号频谱
% t1 = linspace(1,2000,2000);
% sp = (fft(realsp,2000));
% figure;
% plot(t1,sp);
% 
%     N_fft = 2000;         % fft点数
%     % 参考信号
%     Sig_ref = exp(1i*pi*B/taup*(t).^2);
%     F_Sig_ref = fft(Sig_ref,N_fft);
%     % 雷达信号的脉冲压缩
%     PC_Sig_rec = fftshift( ifft(fft(sp,N_fft).*(conj(F_Sig_ref))) );
% 
%     figure;
%     plot(t1,PC_Sig_rec)

