import matplotlib.pyplot as plt
import numpy as np
import pywt
from functools import reduce
from argparse import Namespace
from matplotlib.font_manager import FontProperties
def fun1(opt):
    t = np.arange(-1,1, 1./opt.N)
    y = np.sin(np.sin(np.pi/3*t)*25*np.pi)
    return t,y
def make_phased_waves(opt):
    t = np.arange(0, 1, 1./opt.N)
    if opt.A is None:
        yt = reduce(lambda a, b: a + b,
                    [np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, phi in zip(opt.K, opt.PHI)])
    else:
        yt = reduce(lambda a, b: a + b,
                    [Ai * np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, Ai, phi in zip(opt.K, opt.A, opt.PHI)])
    return t,yt

chinese_font = FontProperties(fname='C:\Windows\Fonts\STXINGKA.TTF')
opt = Namespace()
opt.N = 1024
opt.K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
opt.A = [1 for _ in opt.K]
# opt.A = [0.1 * (a + 1) for a in range(len(opt.K))]
opt.PHI = [np.random.rand() for _ in opt.K]
sampling_rate = 1024
# t = np.arange(0, 1.0, 1.0 / sampling_rate)
f1 = 100
f2 = 200
f3 = 300
t,data = make_phased_waves(opt)
# data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
#                     [lambda t: np.sin(2 * np.pi * f1 * t), lambda t: np.sin(2 * np.pi * f2 * t),
#                      lambda t: np.sin(2 * np.pi * f3 * t)])
wavename = 'cgau8'
totalscal = 256
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(t, data)
plt.xlabel(u"时间(秒)", fontproperties=chinese_font)
plt.title(u"时频谱", fontproperties=chinese_font, fontsize=20)
plt.subplot(212)
plt.ylim(0,90)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"频率(Hz)", fontproperties=chinese_font)
plt.xlabel(u"时间(秒)", fontproperties=chinese_font)
plt.subplots_adjust(hspace=0.4)
plt.show()