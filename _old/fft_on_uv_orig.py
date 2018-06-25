# coding: utf-8
import numpy as np
import scipy
import pylab as plt
from omnium.utils import get_cube
import iris

FILE_LOC = '/home/markmuetz/mirrors/archer/work/cylc-run/u-ax548/share/data/history/km1_large_dom_no_wind_dso_damping/uv072.nc'

TEST = False
SINGLE_TIME = True
LOOP_TIME = False
LOOP_HEIGHT = True
LOOP_FILTER_SCALE = False
PLOTS = [2]
# PLOTS = [2, 4, 6]
# PLOTS = [4, 5, 6]

LEVEL_HEIGHT_INDEX = 15

CUTOFF = 'sigmoid'
FILTER_SCALE = 4
FILTER_SHAPRNESS = 0.5


def radial_profile_orig(data, center):
    # Thank you: https://stackoverflow.com/a/21242776/54557
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # TODO: no need for this - just use histogram in float version of r.
    r_approx = r.astype(np.int)

    tbin = np.bincount(r_approx.ravel(), data.ravel())
    nr = np.bincount(r_approx.ravel())
    radialprofile = tbin / nr

    return radialprofile 


def radial_profile(data, centre):
    # BIG DIFF between this and other function.
    y, x = np.indices((data.shape))
    r = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)

    radialprofile, bins, binindex = scipy.stats.binned_statistic(r.flatten(), 
                                                                 data.flatten(), 
                                                                 bins=int(r.max()),
                                                                 statistic='mean')

    midpoints = (bins[1:] + bins[:-1]) / 2
    return midpoints, radialprofile 


def calc_uv_fft(u_data, v_data, cutoff=CUTOFF, scale=FILTER_SCALE, sharpness=FILTER_SHAPRNESS, pause=3, 
               wait=False):
    speed_data = np.sqrt(u_data**2 + v_data**2)
    ft_speed = np.fft.fft2(speed_data)
    ft_speed_shifted = np.fft.fftshift(ft_speed)

    pow_speed2 = np.abs(ft_speed_shifted)**2
    y, x = np.indices((speed_data.shape))
    r = np.sqrt((x - 128)**2 + (y - 128)**2)

    ft_speed_hi_pass = ft_speed_shifted.copy()
    ft_speed_lo_pass = ft_speed_shifted.copy()

    if cutoff == 'sharp':
        ft_speed_hi_pass[r <= scale] = 0
        ft_speed_lo_pass[r > scale] = 0
    if cutoff == 'sigmoid':
        sigmoid = 1 / (1 + np.exp((r - scale) * sharpness))
        ft_speed_hi_pass *= (1 - sigmoid)
        ft_speed_lo_pass *= sigmoid

    speed_hi_pass = np.fft.ifft2(np.fft.fftshift(ft_speed_hi_pass))
    speed_lo_pass = np.fft.ifft2(np.fft.fftshift(ft_speed_lo_pass))

    if 1 in PLOTS:
        plt.figure(1)
        plt.clf()
        plt.imshow(np.log10(pow_speed2))
        # plt.imshow(pow_speed2)
        plt.pause(0.001)

    if 2 in PLOTS:
        plt.figure(2)
        plt.clf()
        r, r_pow_speed2 = radial_profile(pow_speed2, (128, 128))

        # N.B. don't plot (128, 128) which is zero freq term.
        # Single row along x
        plt.loglog(pow_speed2[128, 129:], 'b-', label='x-dir')
        # Single col along y
        plt.loglog(pow_speed2[128:, 129], 'r-', label='y-dir')
        # Radial sum.
        plt.loglog(r[1:129], r_pow_speed2[1:129], 'g-', label='radial')

        x = np.linspace(4, 128, 2)
        y = 215443469 * x ** (-5 / 3)
        plt.loglog(x, y, 'k--', label='-5/3')
        plt.legend()
        plt.xlabel('scale')
        plt.ylabel('power (m$^2$ s$^{-2}$)')
        plt.pause(0.001)
    
    if 4 in PLOTS:
        plt.figure(4)
        plt.clf()
        plt.imshow(speed_data)
        plt.pause(.001)

    #import ipdb; ipdb.set_trace()

    if 5 in PLOTS:
        assert speed_hi_pass.imag.max() < 1e12
        assert speed_lo_pass.imag.max() < 1e12

        plt.figure(5)
        plt.clf()
        plt.title('High pass (cutoff={}, scale={}, sharpness={})'
                  .format(cutoff, scale, sharpness))
        plt.imshow(np.real(speed_hi_pass))
        plt.pause(.001)

    if 6 in PLOTS:
        plt.figure(6)
        plt.clf()
        plt.title('Low pass (cutoff={}, scale={}, sharpness={})'
                  .format(cutoff, scale, sharpness))
        plt.imshow(np.real(speed_lo_pass))
        plt.pause(.001)

    if wait:
        r = input('Enter to continue, q to quit: ')
        if r == 'q':
            return
    else:
        plt.pause(pause)


def main():
    uv = iris.load(FILE_LOC)
    u = get_cube(uv, 0, 2)
    v = get_cube(uv, 0, 3)
    plt.ion()

    if SINGLE_TIME:
        print('Height: {}'.format(u[0, LEVEL_HEIGHT_INDEX].coord('level_height').points[0]))
        calc_uv_fft(u[-1, LEVEL_HEIGHT_INDEX].data, v[-1, LEVEL_HEIGHT_INDEX].data)
    elif LOOP_TIME:
        print('Height: {}'.format(u[0, LEVEL_HEIGHT_INDEX].coord('level_height').points[0]))
        print('loop_time')
        for i in range(u.shape[0]):
            print(i)
            calc_uv_fft(u[i, LEVEL_HEIGHT_INDEX].data, v[i, LEVEL_HEIGHT_INDEX].data, pause=0.01)
    elif LOOP_HEIGHT:
        print('loop_height')
        for i in range(uv.shape[1]):
            print('Height: {}'.format(uv[0, i].coord('level_height').points[0]))
            calc_uv_fft(u[-1, i].data, v[-1, i].data, pause=0.01)
    elif LOOP_FILTER_SCALE:
        print('loop_filter_scale')
        print('Height: {}'.format(uv[0, LEVEL_HEIGHT_INDEX].coord('level_height').points[0]))
        for i in range(100):
            calc_uv_fft(u[-1, i].data, v[-1, i].data, scale=i, pause=0.01)

    return uv


if __name__ == '__main__':
    uv = main()

