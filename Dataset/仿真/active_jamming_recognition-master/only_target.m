%2020.11.16
%��������Ŀ���������Ϊ��������,Ŀ��λ��0-1500�㴦��
close all;clear;clc
j=sqrt(-1);
data_num=500;   %����������
samp_num=2000;%���봰����
fs = 20e6; %����Ƶ��
B = 10e6;  %�źŴ���
taup = 20e-6; %�ź�����
t = linspace(taup/2,taup/2*3,taup*fs);          %ʱ������
lfm = exp(1j*pi*B/taup*t.^2);          %LFM�ź�

SNR=0; %�����dB
echo=zeros(data_num,samp_num,3);     %�����С��500,2000,2��
echo_stft=zeros(data_num,100,247,3);  %�����С��500,200,1000,2��
num_label = 0;
label=zeros(1,data_num)+num_label;                         %��ǩ����,�˸��ű�ǩΪ0

for m=1:data_num
    
    range=1+round(rand(1,1)*1500);
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%��������
    sp=sp/std(sp);
    As=10^(SNR/20);%Ŀ��ز�����
    
    sp(range:length(lfm)+range-1)=sp(range:length(lfm)+range-1)+As*lfm;  %����+Ŀ��ز� Ŀ���ھ��봰��range�㴦


    sp=sp/max(max(sp));
    sp_abs=abs(sp);
    figure(3)
    plot(linspace(0,100,2000),sp);
    set(gca,'FontName','Times New Roman');
    xlabel('Time/��s','FontSize',15);ylabel('Normalized amplitude','FontSize',15)
    
    
    echo(m,1:2000,1)=real(sp); 
    echo(m,1:2000,2)=imag(sp);
    echo(m,1:2000,3)=sp_abs;
%     echo(m,1:2000,4)=angle(sp); %�ź�ʵ�����鲿�ֿ�������ά������
    [S,~,~,~]=spectrogram(sp,32,32-8,100,20e6);
    
    S=S/max(max(S));
    S_abs=abs(S);
    figure(4)
    imagesc(linspace(0,100,size(S,1)),linspace(-10,10,size(S,2)),abs(S));
    set(gca,'FontName','Times New Roman');
    xlabel('Time/��s','FontSize',15);ylabel('Frequency/MHz','FontSize',15)
% ['\fontname{����}����\fontname{Times New Roman} (mm)']
    
    echo_stft(m,1:size(S,1),1:size(S,2),1)=real(S);
    echo_stft(m,1:size(S,1),1:size(S,2),2)=imag(S);
    echo_stft(m,1:size(S,1),1:size(S,2),3)=S_abs; 
%      echo_stft(m,1:size(S,1),1:size(S,2),4)=angle(S);
    
end

% % save('F:\���ѧϰ����ŷ�������_2020.11.16\jamming_data\only_target_0\echo.mat' ,'echo')
% % save('F:\���ѧϰ����ŷ�������_2020.11.16\jamming_data\only_target_0\echo_stft.mat' ,'echo_stft')
% % save('F:\���ѧϰ����ŷ�������_2020.11.16\jamming_data\only_target_0\label.mat' ,'label')
%--------------------------------------------------------------------------------------------%
t_data=load('D:\CodeSpace\active_jamming_recognition\data\t_data.mat').t_data;
tf_data=load('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat').tf_data;
gt_label=load('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat').gt_label;
% 
t_data(1+500*(0):500*(0+1),:,:)=echo; 
tf_data(1+500*(0):500*(0+1),:,:,:)=echo_stft; 
gt_label(1,1+500*(0):500*(0+1))=label;
% 
save('D:\CodeSpace\active_jamming_recognition\data\t_data.mat','t_data')
save('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat','tf_data')
save('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat','gt_label')

