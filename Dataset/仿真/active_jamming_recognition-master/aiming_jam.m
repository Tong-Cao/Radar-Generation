%2020.11.17
%������׼������Ϊ�������ݣ�20-40Mhz����
close all;clear;clc
j=sqrt(-1);
data_num=500;   %����������
samp_num=2000;%���봰����
fs = 20e6; %����Ƶ��
B = 10e6;  %�źŴ���
taup = 20e-6; %�ź�����
t = linspace(0,taup,taup*fs);          %ʱ������ 400����
lfm = exp(1j*pi*B/taup*t.^2);          %LFM�ź�

SNR=0; %�����dB
echo=zeros(data_num,samp_num,3);     %�����С��500,2000,3��
echo_stft=zeros(data_num,100,247,3);  %�����С��500,200,1000,3��
num_label = 5;
label=zeros(1,data_num)+num_label;                         %��ǩ����,�˸��ű�ǩΪ0

for m=1:data_num
    
    JNR=30+round(rand(1,1)*30); %�����30-60dB
    range=round(rand(1,1)*1500);%��Ƶ����������ʼ��
    Bj=20+round(rand(1,1)*20);%��Ƶ����20-40Mhz
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%��������
    sp=sp/std(sp);
    sp1=sp;
    As=10^(SNR/20);%Ŀ��ز�����
    Aj=10^(JNR/20);  %���Żز�����
    
    sp1_fft=fftshift(fft(sp1));
    sp1_fft(1:round(2000*((1-Bj/80)/2)))=0; % round����ȡ��  ��1-round(2000*((1-Bj/80)/2))����Ϊ0 ������֮ǰ����Ϊ0
    sp1_fft(round(2000*((1-Bj/80)/2))+round(Bj/80*2000):2000)=0; % ������֮������Ϊ0
    sp1=ifftshift(ifft(sp1_fft)); % �����˲�������
    range_tar=1+round(rand(1,1)*1400); % lfm��Χ���

    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+As*lfm;  %����+Ŀ��ز� Ŀ���ھ��봰��range�㴦
    sp=sp+Aj*sp1;
% 
    sp=sp/max(max(sp));
    sp_abs=abs(sp);
    figure(3)
    plot(linspace(0,100,2000),sp);xlabel('ʱ��/��s','FontSize',20);ylabel('��һ������','FontSize',20)
    
    echo(m,1:2000,1)=real(sp); 
    echo(m,1:2000,2)=imag(sp);
    echo(m,1:2000,3)=sp_abs;
%     echo(m,1:2000,4)=angle(sp); %�ź�ʵ�����鲿�ֿ�������ά������
     [S,~,~,~]=spectrogram(sp,32,32-8,100,20e6);
    
    S=S/max(max(S));
    S_abs=abs(S);
     figure(4)
    imagesc(linspace(0,100,size(S,1)),linspace(-40,40,size(S,2)),abs(S));
    xlabel('ʱ��/��s','FontSize',20);ylabel('Ƶ��/MHz','FontSize',20)


    echo_stft(m,1:size(S,1),1:size(S,2),1)=real(S);
    echo_stft(m,1:size(S,1),1:size(S,2),2)=imag(S);
    echo_stft(m,1:size(S,1),1:size(S,2),3)=S_abs;
%     echo_stft(m,1:size(S,1),1:size(S,2),4)=angle(S);

 
    
end

% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\aiming_jam_5\echo.mat' ,'echo')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\aiming_jam_5\echo_stft.mat' ,'echo_stft')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\aiming_jam_5\label.mat' ,'label')

t_data=load('D:\CodeSpace\active_jamming_recognition\data\t_data.mat').t_data;
tf_data=load('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat').tf_data;
gt_label=load('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat').gt_label;
% 
t_data(1+500*(num_label):500*(num_label+1),:,:)=echo; 
tf_data(1+500*(num_label):500*(num_label+1),:,:,:)=echo_stft; 
gt_label(1,1+500*(num_label):500*(num_label+1))=label;
% 
save('D:\CodeSpace\active_jamming_recognition\data\t_data.mat','t_data')
save('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat','tf_data')
save('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat','gt_label')


% 
% figure(3)
% plot(1:2000,sp)
% figure(4)
% [S,F,T,~]=spectrogram(sp,64,64-7,200,20e6);
% 
% S=S/max(max(S));
% imagesc(0:2000,-40e6:40e6,abs(S))