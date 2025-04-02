#python load data example, 数据加载示例
import scipy.io as sio

#路径
root='.\RData\JammingType-ComplexSequence'
path=root+'\aiming_jam\1.mat';
file=sio.loadmat(path) 


#数据读取
realpart=file['realsp'] # real component，实部
imagpart=file['imagsp']  # imag component， 虚部
iqpart=file['complexsp']  # IQ sequence， IQ 复数
