#线性调频信号仿真
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def LFM(f0,f1,T,fs):
    t = np.arange(-T/2,T/2,1/fs)
    K = (f1-f0)/T
    phi = 2*np.pi*(f0*t+K/2*t*t)
    s = np.exp(1j*phi)
    return s,t

# 匹配滤波信号
def match_filter(f0,f1,T,fs):
    t = np.arange(-T/2,T/2,1/fs)
    K = (f1-f0)/T
    phi = 2*np.pi*(f0*t+K/2*t*t)
    s = np.exp(-1j*phi)
    return s,t



f0 = 0      #起始频率
f1 = 30e6   #终止频率 30MHZ
T = 10e-6   #时间范围 1us
fs = 70e6   #采样率 需要大于最高频率的两倍
s1,t1 = LFM(f0,f1,T,fs)
s2,t2 = match_filter(f0,f1,T,fs)

s3 = np.conv(s1,s2)



plt.figure()
plt.plot(t1,s3)


plt.show()




