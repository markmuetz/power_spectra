import numpy as np
import numpy.fft as fft
import pylab as plt
import iris


plt.ion()

FILE_LOC = '/home/markmuetz/mirrors/archer/work/cylc-run/u-ax548/share/data/history/m500_large_dom_no_wind/w072.nc'
w = iris.load(FILE_LOC)[0]

k = fft.fftshift(fft.fftfreq(w.shape[-1], d=0.5))
l = k.copy()
K, L = np.meshgrid(k, l)

fts = fft.fftshift(fft.fft2(w[:, 15].data), axes=(1, 2))
angle = np.angle(fts[0, 257, 256]) - np.angle(fts[:, 257, 256])
fts_trans = fts * np.exp(1j * angle[:, None, None] * L[None, :, :] / L[257, 256])
w_trans = fft.ifft2(fft.ifftshift(fts_trans, axes=(1, 2)))

for i in range(24):
    plt.clf()
    plt.imshow(w_trans[i].real)
    plt.pause(0.3)
    
    
