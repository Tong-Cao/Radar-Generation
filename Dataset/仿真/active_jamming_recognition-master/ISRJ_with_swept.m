%2020.11.16
%������Ъ����ת��������Ϊ�������ݣ������30-60dB֮�������
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
num_label = 11;
label=zeros(1,data_num)+num_label;                         %��ǩ����,�˸��ű�ǩΪ0

repetion_times=[4,3,2,1];
period=[5e-6,10e-6];
duty=[20,25,33.33,50];

for m=1:data_num
    
    JNR=30+round(rand(1,1)*30); %�����30-60dB
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%��������
    sp=sp/std(sp);
    As=10^(SNR/20);%Ŀ��ز�����
    Aj=10^(JNR/20);%���Żز�����
    range_tar=1+round(rand(1,1)*1400);
    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+As*lfm;  %����+Ŀ��ز� Ŀ���ھ��봰��200�㴦
    index1=1+round(rand(1,1));
    index2=1+round(rand(1,1)*3);
    period1=period(index1);
    duty1=duty(index2);
    repetion_times1=repetion_times(index2);
    squa=(square((1/period1)*2*pi*t, duty1)+1)/2;
    squa(400)=0;
    squa1=lfm.*squa;
    delay_time=period1*duty1*0.01;
    delay_num=ceil(delay_time*fs);
    for i=1:repetion_times1 %���ת��
        
        sp(range_tar+1+i*delay_num:range_tar+1+i*delay_num+399)=sp(range_tar+1+i*delay_num:range_tar+1+i*delay_num+399)+Aj/2*squa1;
      
    end
    
    T=round(45+(rand(1,1)*40))*1e-6;%ɨƵ����45-80us
    t1 = linspace(0,T,T*fs);          %ʱ������
    B1=2*B;

    swept = exp(1j*pi*B1/T*t1.^2);          %swept�ź�
    jam=swept;
     spp=randn([1,length(jam)])+1j*randn([1,length(jam)]);
    sppfft=fft(spp);
    sppfft(1:length(sppfft)/2-80)=0;sppfft(length(sppfft)/2+80:length(sppfft))=0;
    spp=ifft(sppfft);
     jam=swept.*spp;
    
    if length(jam)<1001
        sp(1:length(jam))=sp(1:length(jam))+Aj/2*jam;
        sp(length(jam)+1:2*length(jam))=sp(length(jam)+1:2*length(jam))+Aj/2*jam;
    else
        sp(1:length(jam))=sp(1:length(jam))+Aj/2*jam;
    end
    

    sp=sp/max(max(sp));
    sp_abs=abs(sp);
    figure(3)
    plot(linspace(0,100,2000),sp);xlabel('ʱ��/��s','FontSize',20);ylabel('��һ������','FontSize',20)
%     
    echo(m,1:2000,1)=real(sp); 
    echo(m,1:2000,2)=imag(sp);
    echo(m,1:2000,3)=sp_abs;
%     echo(m,1:2000,4)=angle(sp); %�ź�ʵ�����鲿�ֿ�������ά������
%     [S,~,~,~]=spectrogram(sp,16,16-8,50,20e6);
     [S,~,~,~]=spectrogram(sp,32,32-8,100,20e6);
    S=S/max(max(S));
    S_abs=abs(S);
    figure(4)
    imagesc(linspace(0,100,size(S,1)),linspace(-10,10,size(S,2)),abs(S));
    xlabel('ʱ��/��s','FontSize',20);ylabel('Ƶ��/MHz','FontSize',20)
    
    
    echo_stft(m,1:size(S,1),1:size(S,2),1)=real(S);
    echo_stft(m,1:size(S,1),1:size(S,2),2)=imag(S);
    echo_stft(m,1:size(S,1),1:size(S,2),3)=S_abs;
%     echo_stft(m,1:size(S,1),1:size(S,2),4)=angle(S);
 
    
end

% % save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_with_swept_11\echo.mat' ,'echo')
% % save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_with_swept_11\echo_stft.mat' ,'echo_stft')
% % save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_with_swept_11\label.mat' ,'label')

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






