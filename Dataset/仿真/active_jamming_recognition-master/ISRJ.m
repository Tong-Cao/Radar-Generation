%2020.11.16
%产生间歇采样转发干扰作为分类数据，干噪比30-60dB之间随机；
close all;clear;clc
j=sqrt(-1);
data_num=1;   %干扰样本数
samp_num=2000;%距离窗点数
fs = 20e6; %采样频率
B = 10e6;  %信号带宽
taup = 20e-6; %信号脉宽
t = linspace(taup/2,taup/2*3,taup*fs);          %时间序列
lfm = exp(1j*pi*B/taup*t.^2);          %LFM信号

SNR=0; %信噪比dB
echo=zeros(data_num,samp_num,3);     %矩阵大小（500,2000,2）
echo_stft=zeros(data_num,100,247,3);  %矩阵大小（500,200,1000,2）
num_label = 2;
label=zeros(1,data_num)+num_label;                         %标签数据,此干扰标签为0

repetion_times=[4,3,2,1];%重复次数
period=[5e-6,10e-6]; %周期
duty=[20,25,33.33,50]; %功率

for m=1:data_num
    
    JNR=30+round(rand(1,1)*30); %干噪比30-60dB
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
    sp=sp/std(sp);
    As=10^(SNR/20);%目标回波幅度
    Aj=10^(JNR/20);%干扰回波幅度
%     Aj = 10;
    range_tar=1+round(rand(1,1)*1400); %信号起始点

    figure(1);
    plot(t,lfm);

    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+10*lfm;  %噪声+目标回波 目标在距离窗内200点处
    
    figure(2);
    plot(linspace(0,100,2000),sp);
    title('信号');

    
    index1=1+round(rand(1,1)); % 1-2
    index2=1+round(rand(1,1)*3); % 1-4
    period1=period(index1); % 随机索引period里的值
    duty1=duty(index2); %索引
%     repetion_times1=repetion_times(index2);%索引 重复次数
    repetion_times1 = 3;

    squa=(square((1/period1)*2*pi*t, duty1)+1)/2;%生成周期为period1占空比为duty1的脉冲信号
    squa(400)=0; % 将最后一个上升沿去掉 
    squa1=lfm.*squa; %回波信号乘以脉冲
    figure(3);
    plot(t,squa1);
    title('采样后');
    delay_time=period1*duty1*0.01; %延时 即高电平持续时间
    delay_num=ceil(delay_time*fs); % 延时点数
    for i=1:repetion_times1 %多次转发
        
        sp(range_tar+1+i*delay_num:range_tar+1+i*delay_num+399)=sp(range_tar+1+i*delay_num:range_tar+1+i*delay_num+399)+Aj*squa1;
      
    end
    

    sp=sp/max(max(sp));
    sp_abs=abs(sp);
    figure(4)
    plot(linspace(0,100,2000),sp);
    set(gca,'FontName','Times New Roman');
    xlabel('Time/μs','FontSize',15);ylabel('Normalized amplitude','FontSize',15)
    
    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)-10*lfm;
    figure(5)
    plot(linspace(0,100,2000),sp);
    title('去掉lfm');


    echo(m,1:2000,1)=real(sp); 
    echo(m,1:2000,2)=imag(sp);
    echo(m,1:2000,3)=sp_abs; 
%     echo(m,1:2000,4)=angle(sp); %信号实部、虚部分开存入三维张量中
    [S,~,~,~]=spectrogram(sp,32,32-8,100,20e6);
    
    S=S/max(max(S));
    S_abs=abs(S);
%     figure(5)
%     imagesc(linspace(0,100,size(S,1)),linspace(-10,10,size(S,2)),abs(S));
%     set(gca,'FontName','Times New Roman');
%     xlabel('Time/μs','FontSize',15);ylabel('Frequency/MHz','FontSize',15)


    
    echo_stft(m,1:size(S,1),1:size(S,2),1)=real(S);
    echo_stft(m,1:size(S,1),1:size(S,2),2)=imag(S);
    echo_stft(m,1:size(S,1),1:size(S,2),3)=S_abs;
%     echo_stft(m,1:size(S,1),1:size(S,2),4)=angle(S);
 
    
end

% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_2\echo.mat' ,'echo')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_2\echo_stft.mat' ,'echo_stft')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_2\label.mat' ,'label')

% t_data=load('D:\CodeSpace\active_jamming_recognition\data\t_data.mat').t_data;
% tf_data=load('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat').tf_data;
% gt_label=load('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat').gt_label;
% % 
% t_data(1+500*(num_label):500*(num_label+1),:,:)=echo; 
% tf_data(1+500*(num_label):500*(num_label+1),:,:,:)=echo_stft; 
% gt_label(1,1+500*(num_label):500*(num_label+1))=label;
% % 
% save('D:\CodeSpace\active_jamming_recognition\data\t_data.mat','t_data')
% save('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat','tf_data')
% save('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat','gt_label')


% figure(1)
% plot(1:400,squa)
% 
% figure(2)
% plot(1:400,squa1)
% figure(3)
% plot(1:2000,sp)
% figure(4)
% [S,~,~,~]=spectrogram(sp,32*2,32*2-1,200,20e6);
% 
% S=S/max(max(S));
% imagesc(1:size(S,1),1:size(S,2),abs(S))




