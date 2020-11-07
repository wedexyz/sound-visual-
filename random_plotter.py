from pylive import live_plotter
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


dt = 40e-9
t = np.arange(0, 1000e-6, dt)
fscale = t/max(t)
y = np.cos(2 * np.pi * 1e6 * t*fscale) + (np.cos(2 * np.pi * 2e6 *t*fscale) * np.cos(2 * np.pi * 2e6 * t*fscale))
y *= np.hanning(len(y))
yy = np.concatenate((y, ([0] * 10 * len(y))))

# FFT of this
Fs = 1 / dt  # sampling rate, Fs = 500MHz = 1/2ns
n = len(yy)  # length of the signal
k = np.arange(n)
T = n / Fs
frq = k / T  # two sides frequency range
frq = frq[range(n // 2)]  # one side frequency range
Y = fftpack.fft(yy) / n  # fft computing and normalization
Y = Y[range(n // 2)] / max(Y[range(n // 2)])
# plotting the specgram
plt.subplot(3, 1, 3)
Pxx, freqs, bins, im = plt.specgram(y, NFFT=512, Fs=Fs, noverlap=10)
plt.show()

size = 100
x_vec = np.linspace(0,1,size+1)[0:-1]
y_vec = np.random.randn(len(x_vec))
line1 = []



while True:
    rand_val = np.random.randn(1)
    y_vec[-1] = rand_val
    line1 = live_plotter(x_vec,y_vec,line1)
    y_vec = np.append(y_vec[1:],0.0)
