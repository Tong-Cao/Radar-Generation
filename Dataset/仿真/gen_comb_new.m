function jamming_comb = gen_comb_new(fs,f0,fj,tj,cpi)
%% ʹ��ʾ��
%     fj_1 = [-50e6 -40e6 -30e6 -20e6 -10e6 0 10e6 20e6 30e6 40e6 50e6];%���ø�������Ƶ��(Hz)
%     tj_1 = 2e-6; % ����פ��ʱ��(us)
%     jamm_comb = gen_comb_new(480e6,120e6,fj_1,tj_1,10e-6);%������״�׸���
%% ����˵��
% ������״�׸���(��Ƶ�ź���)
% Input:
%   fs      --  ����Ƶ��(Hz)
%   fj      --  ��������Ƶ����(Hz)
%   tj      --  פ��ʱ��(us)
%   cpi     --  ���ų���ʱ��(us)
% Output:
%   jamming   -- ��״�׸���
%% ��״���ź�
N1 = round(tj*fs);          % ���ų���ʱ���ڵĲ�������
%���������״�׵ĸ���Ƶ��
N =ceil(cpi/tj);
M = length(fj);
num_choose =randi([1,M],1,N);
fj_rand =zeros(1,N);
for i=1:1:N
    fj_rand(1,i) =fj(1,num_choose(1,i));
end
%��״���ź�
tt = (1:N1)*(1/fs);         % ʱ������
jamming = [];
for j=1:1:N
    jamming_rand =exp(1*sqrt(-1)*2*pi*(fj_rand(1,j)+f0)*tt);
    jamming =[jamming_rand jamming];
end
jamming_comb =jamming(1,1:round(cpi*fs));
end