import pyaudio
import struct
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
from scipy import signal
from scipy.fftpack import fft
import time
plt.style.use('dark_background')


# constants
p = pyaudio.PyAudio()

CHUNK = 1024 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
# create matplotlib figure and axes
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, figsize=(5, 10))

CHUNK = int(RATE/20)
# stream object to get data from microphone
stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,output=True,frames_per_buffer=CHUNK)
# variable for plotting
x = np.arange(0, 2 * CHUNK, 2)       # samples (waveform)
xf = np.linspace(0, RATE, CHUNK)     # frequencies (spectrum)
# create a line object with random data
line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=1)

line1, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=1)
line2, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=1)
# create semilogx line for spectrum
line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK), '-', lw=1)

line1_fft, = ax2.plot(x, np.random.rand(CHUNK), '-', lw=1)
line2_fft, = ax2.plot(x, np.random.rand(CHUNK), '-', lw=1)

# format waveform axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(0, 255)
plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])
ax1.set_xlim(0, 2 * CHUNK)
#ax1.plot(x, .7*s, color='C4', linestyle='--')
# format spectrum axes
ax2.set_xlim(20, RATE / 2)


print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()



while True:
    # binary data
    data = stream.read(CHUNK)  
    # convert data to integers, make np array, then offset it by 127
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)

    # create np array and offset by 128
    data_np = np.array(data_int, dtype='b')[::2] + 128
    line.set_ydata(data_np)

    line1.set_ydata(data_np*0.5)
    line2.set_ydata(data_np*0.7)

    # compute FFT and update line
    yf = fft(data_int)
    line_fft.set_ydata(np.abs(yf[0:CHUNK])  / (128 * CHUNK))

    line1_fft.set_ydata(np.abs(yf[0:CHUNK])  / (10 * CHUNK))
    line2_fft.set_ydata(np.abs(yf[0:CHUNK])  / (1000 * CHUNK))

    #databaru = (np.max(data_np))
    #print(databaru)

    #data = np.array(data_int, dtype='b')
    #f, t, Sxx = signal.spectrogram(yf, RATE, nperseg=64)
    f, t, Sxx = signal.spectrogram(data_np, RATE, nperseg=64, nfft=256, noverlap=60)
    #f, t, Sxx = signal.spectrogram(yf, RATE, noverlap=250)
    #f, t, Sxx = signal.spectrogram(data_np, fs=CHUNK)
    dBS = 10 * np.log10(Sxx) #convert db
    #plt.pcolormesh(t, f, dBS)
    plt.pcolormesh(t, f, dBS, cmap='nipy_spectral')
    #plt.pcolormesh(t, f, dBS)
    #plt.show()
    #Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, 
    #OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, 
    # PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r,
    #  Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, 
    #afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, 
    # cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, 
    # gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, 
    # magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, 
    # summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
    widths = np.arange(1, 50)
    cwtmatr = signal.cwt(data_np, signal.ricker, widths)
    #scales = mlpy.wavelet.autoscales(N=len(data),dt=1,dj=0.05,wf='morlet',p=omega0)
    #spec = mlpy.wavelet.cwt(data[:,1],dt=1,scales=scales,wf='morlet',p=omega0)
    #spec = numpy.abs(spec)**2
    ax3.imshow(cwtmatr, extent=[-1, 1, 1, 50], cmap='jet', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())



    plt.pause(0.005)
  