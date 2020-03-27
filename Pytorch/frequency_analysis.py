import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from argparse import Namespace
from functools import reduce

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
def linear(opt,w,b):
    t = np.arange(-4, 4, 1. / opt.N)
    yt =  np.abs(t+1)-2*np.abs(t+3)+np.abs(t)+np.abs(t-2)+3*np.abs(t-3)-2*np.abs(t-2.5)
    return t,yt

def fft(opt, yt):
    n = len(yt) # length of the signal
    k = np.arange(n)
    T = n/opt.N
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range
    # -------------
    FFTYT = np.fft.fft(yt)/n # fft computing and normalization
    FFTYT = FFTYT[range(n//2)]
    fftyt = abs(FFTYT)
    return frq, fftyt

def plot_wave_and_spectrum(opt, x, yox):
    # Btw, "yox" --> "y of x"
    # Compute fft
    k, yok = fft(opt, yox)
    # Plot
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4))
    ax0.set_title("Function")
    ax0.plot(x, yox)
    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")
    ax1.set_title("FT of Function")
    ax1.plot(k, yok)
    ax1.set_xlabel("k")
    ax1.set_ylabel("f(k)")
    plt.show()

opt = Namespace()
opt.N = 200
opt.K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# opt.A = [1 for _ in opt.K]
# opt.A = [0.1 * (a + 1) for a in range(len(opt.K))]
opt.PHI = [np.random.rand() for _ in opt.K]
# Model parameters
opt.INP_DIM = 1
opt.OUT_DIM = 1
opt.WIDTH = 256
opt.DEPTH = 6
# Training
opt.CUDA = True
opt.NUM_ITER = 1000
opt.REC_FRQ = 100
opt.LR = 0.0003
## visualization
x, y = linear(opt,0.01,0.8)
plot_wave_and_spectrum(opt, x, y)