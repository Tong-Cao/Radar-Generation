function jamming_comb = gen_comb_new(fs,f0,fj,tj,cpi)
%% 使用示例
%     fj_1 = [-50e6 -40e6 -30e6 -20e6 -10e6 0 10e6 20e6 30e6 40e6 50e6];%设置干扰中心频率(Hz)
%     tj_1 = 2e-6; % 设置驻留时间(us)
%     jamm_comb = gen_comb_new(480e6,120e6,fj_1,tj_1,10e-6);%生成梳状谱干扰
%% 参数说明
% 产生梳状谱干扰(单频信号类)
% Input:
%   fs      --  采样频率(Hz)
%   fj      --  干扰中心频率组(Hz)
%   tj      --  驻留时间(us)
%   cpi     --  干扰持续时间(us)
% Output:
%   jamming   -- 梳状谱干扰
%% 梳状谱信号
N1 = round(tj*fs);          % 干扰持续时间内的采样点数
%随机产生梳状谱的干扰频点
N =ceil(cpi/tj);
M = length(fj);
num_choose =randi([1,M],1,N);
fj_rand =zeros(1,N);
for i=1:1:N
    fj_rand(1,i) =fj(1,num_choose(1,i));
end
%梳状谱信号
tt = (1:N1)*(1/fs);         % 时间向量
jamming = [];
for j=1:1:N
    jamming_rand =exp(1*sqrt(-1)*2*pi*(fj_rand(1,j)+f0)*tt);
    jamming =[jamming_rand jamming];
end
jamming_comb =jamming(1,1:round(cpi*fs));
end