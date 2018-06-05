# coding: utf-8
import numpy as np
import scipy
import pylab as plt
import omnium as om
import iris

FILE_LOC = '/home/markmuetz/mirrors/archer/work/cylc-run/u-ax548/share/data/history/m500_large_dom_no_wind/w072.nc'

TEST = False
SINGLE_TIME = True
LOOP_TIME = False
LOOP_HEIGHT = True
LOOP_FILTER_SCALE = False
PLOTS = [1, 2, 3, 4, 5, 6]
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


def calc_w_fft(w_data, cutoff=CUTOFF, scale=FILTER_SCALE, sharpness=FILTER_SHAPRNESS, pause=3, 
               wait=False):
    ft_w = np.fft.fft2(w_data)
    ft_w_shifted = np.fft.fftshift(ft_w)

    pow_w2 = np.abs(ft_w_shifted)**2
    y, x = np.indices((w_data.shape))
    r = np.sqrt((x - 256)**2 + (y - 256)**2)

    ft_w_hi_pass = ft_w_shifted.copy()
    ft_w_lo_pass = ft_w_shifted.copy()

    if cutoff == 'sharp':
        ft_w_hi_pass[r <= scale] = 0
        ft_w_lo_pass[r > scale] = 0
    if cutoff == 'sigmoid':
        sigmoid = 1 / (1 + np.exp((r - scale) * sharpness))
        ft_w_hi_pass *= (1 - sigmoid)
        ft_w_lo_pass *= sigmoid

    w_hi_pass = np.fft.ifft2(np.fft.fftshift(ft_w_hi_pass))
    w_lo_pass = np.fft.ifft2(np.fft.fftshift(ft_w_lo_pass))

    if 1 in PLOTS:
        plt.figure(1)
        plt.clf()
        plt.imshow(np.log10(pow_w2))
        # plt.imshow(pow_w2)
        plt.pause(0.001)

    if 2 in PLOTS:
        plt.figure(2)
        plt.clf()
        r, r_pow_w2 = radial_profile(pow_w2, (256, 256))

        # N.B. don't plot (256, 256) which is zero freq term.
        # Single row along x
        plt.loglog(pow_w2[256, 257:], 'b-', label='x-dir')
        # Single col along y
        plt.loglog(pow_w2[256:, 257], 'r-', label='y-dir')
        # Radial sum.
        plt.loglog(r[1:257], r_pow_w2[1:257], 'g-', label='radial')

        x = np.linspace(4, 256, 2)
        y = 215443469 * x ** (-5 / 3)
        plt.loglog(x, y, 'k--', label='-5/3')
        plt.legend()
        plt.xlabel('scale')
        plt.ylabel('power (m$^2$ s$^{-2}$)')
        plt.pause(0.001)
    
    if 4 in PLOTS:
        plt.figure(4)
        plt.clf()
        plt.imshow(w_data)
        plt.pause(.001)

    #import ipdb; ipdb.set_trace()

    if 5 in PLOTS:
        assert w_hi_pass.imag.max() < 1e12
        assert w_lo_pass.imag.max() < 1e12

        plt.figure(5)
        plt.clf()
        plt.title('High pass (cutoff={}, scale={}, sharpness={})'
                  .format(cutoff, scale, sharpness))
        plt.imshow(np.real(w_hi_pass))
        plt.pause(.001)

    if 6 in PLOTS:
        plt.figure(6)
        plt.clf()
        plt.title('Low pass (cutoff={}, scale={}, sharpness={})'
                  .format(cutoff, scale, sharpness))
        plt.imshow(np.real(w_lo_pass))
        plt.pause(.001)

    if wait:
        r = input('Enter to continue, q to quit: ')
        if r == 'q':
            return
    else:
        plt.pause(pause)


def tests(w):
    print('All 1')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    w_test[:, :] = 1
    import ipdb; ipdb.set_trace()
    calc_w_fft(w_test)
    input('Enter to continue')

    print('Centre points')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    w_test[255:257, 255:257] = 1
    calc_w_fft(w_test)
    input('Enter to continue')

    print('Band in x-dir')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    w_test[250:260, :] = 1
    calc_w_fft(w_test)
    input('Enter to continue')

    print('Centre circle r=10')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    y, x = np.indices((w_test.shape))
    # import ipdb; ipdb.set_trace()
    w_test[(x - 256)**2 + (y - 256)**2 < 100] = 1
    calc_w_fft(w_test)
    input('Enter to continue')

    print('Some sines/cosines in x/y.')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    y, x = np.indices((w_test.shape))
    sin_x_modes = [3, 7, 11, 17]
    cos_y_modes = [4, 6, 11, 22]
    for sin_x_mode in sin_x_modes:
        w_test += np.sin(sin_x_mode * 2 * np.pi * x / 512)
    for cos_y_mode in cos_y_modes:
        w_test += np.cos(cos_y_mode * 2 * np.pi * y / 512)
    calc_w_fft(w_test)
    input('Enter to continue')

    print('sinc')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    y, x = np.indices((w_test.shape))
    # 0.0001 stops div by zero errors.
    r = np.sqrt((x - 256.0001)**2 + (y - 256.0001)**2)
    w_test = np.sin(r) / r
    calc_w_fft(w_test)
    input('Enter to continue')


def main():
    w = iris.load(FILE_LOC)[0]
    plt.ion()

    if TEST:
        tests(w)

    if SINGLE_TIME:
        print('Height: {}'.format(w[0, LEVEL_HEIGHT_INDEX].coord('level_height').points[0]))
        calc_w_fft(w[-1, LEVEL_HEIGHT_INDEX].data)
    elif LOOP_TIME:
        print('Height: {}'.format(w[0, LEVEL_HEIGHT_INDEX].coord('level_height').points[0]))
        print('loop_time')
        for i in range(w.shape[0]):
            print(i)
            calc_w_fft(w[i, LEVEL_HEIGHT_INDEX].data, pause=0.01)
    elif LOOP_HEIGHT:
        print('loop_height')
        for i in range(w.shape[1]):
            print('Height: {}'.format(w[0, i].coord('level_height').points[0]))
            calc_w_fft(w[-1, i].data)
    elif LOOP_FILTER_SCALE:
        print('loop_filter_scale')
        print('Height: {}'.format(w[0, LEVEL_HEIGHT_INDEX].coord('level_height').points[0]))
        for i in range(100):
            calc_w_fft(w[-1, LEVEL_HEIGHT_INDEX].data, scale=i, pause=0.01)

    return w


if __name__ == '__main__':
    w = main()

