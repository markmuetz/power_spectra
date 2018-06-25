import sys

import numpy as np
import scipy
import pylab as plt
import iris

from omnium.utils import get_cube

FILE_LOC = '/home/markmuetz/mirrors/archer/work/cylc-run/u-ax548/share/data/history/km1_large_dom_no_wind_dso_damping/uv072.nc'
DEFAULT_METHOD = 'loop_time'

FILTER = True
PLOTS = {
    'loop_time': ['mean_pow', 'mean_pow_spectrum'],
    'loop_height': [],
}

ANIMS = {
    'loop_time': [],
    'loop_height': ['height'],
}

LEVEL_HEIGHT_INDEX = 15
TIME_INDEX = -1

CUTOFF = 'sigmoid'
FILTER_SCALE = 4
FILTER_SHAPRNESS = 0.5

def radial_profile(data, centre=None):
    y, x = np.indices((data.shape))
    if not centre:
        centre = (data.shape[0] // 2, data.shape[1] // 2)
    r = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)

    radialprofile, bins, binindex = scipy.stats.binned_statistic(r.flatten(), 
                                                                 data.flatten(), 
                                                                 bins=int(r.max()),
                                                                 statistic='mean')

    midpoints = (bins[1:] + bins[:-1]) / 2
    return midpoints, radialprofile 


class FftProcUV:
    def __init__(self, cubes):
        self.cubes = cubes
        self.u = get_cube(self.cubes, 0, 2)
        self.v = get_cube(self.cubes, 0, 3)


    def run(self, method, **args):
        """Dispatch on method"""
        print('Running {}'.format(method))
        return getattr(self, method)(**args)
    
    def _calc_filtered(self,
                       cutoff=CUTOFF,
                       scale=FILTER_SCALE,
                       sharpness=FILTER_SHAPRNESS):
        y, x = np.indices((self.u[0, 0].shape))
        r = np.sqrt((x - 128)**2 + (y - 128)**2)

        ft_hi_pass = np.fft.fftshift(self.fts.copy(), axes=(1, 2))
        ft_lo_pass = np.fft.fftshift(self.fts.copy(), axes=(1, 2))
        self.cutoff = cutoff
        self.scale = scale
        self.sharpness =sharpness

        if cutoff == 'sharp':
            ft_hi_pass[:, r <= scale] = 0
            ft_lo_pass[:, r > scale] = 0
        if cutoff == 'sigmoid':
            sigmoid = 1 / (1 + np.exp((r - scale) * sharpness))
            ft_hi_pass *= (1 - sigmoid)
            ft_lo_pass *= sigmoid

        self.hi_pass = np.fft.ifft2(np.fft.ifftshift(ft_hi_pass, axes=(1, 2)))
        self.lo_pass = np.fft.ifft2(np.fft.ifftshift(ft_lo_pass, axes=(1, 2)))
    
    def single_time(self, time_index=TIME_INDEX, level_height_index=LEVEL_HEIGHT_INDEX):
        u_data = self.u[time_index, level_height_index].data
        v_data = self.v[time_index, level_height_index].data
        speed_data = np.sqrt(u_data**2 + v_data**2)

        self.ft = np.fft.fft2(speed_data)
        self.mean_pow = np.abs(self.ft)**2

    def loop_time(self, level_height_index=LEVEL_HEIGHT_INDEX):
        self.height = self.u[0, level_height_index].coord('level_height').points[0]
        print('Height: {}'.format(self.height))

        u_data = self.u[:, level_height_index].data
        v_data = self.v[:, level_height_index].data
        speed_data = np.sqrt(u_data**2 + v_data**2)

        self.fts = np.fft.fft2(speed_data)
        self.pows = np.abs(self.fts)**2

        self.mean_ft = self.fts.mean(axis=0)
        self.mean_pow = self.pows.mean(axis=0)

        if FILTER:
            self._calc_filtered()

    def loop_height(self, time_index=TIME_INDEX):
        self.time = self.cube[time_index].coord('time').points[0]
        print('Time: {}'.format(self.time))

        u_data = self.u[time_index, :].data
        v_data = self.v[time_index, :].data
        speed_data = np.sqrt(u_data**2 + v_data**2)

        self.fts = np.fft.fft2(speed_data)
        self.pows = np.abs(self.fts)**2

    def loop_time_height(self):
        raise Exception('Too memory hungry!')
        shape = self.cube.shape
        self.fts = np.zeros(shape, dtype=np.complex)
        self.pows = np.zeros(shape)

        self.fts = np.fft.fft2(self.cube.data)
        self.pows = np.abs(ft)**2

        self.mean_ft = self.fts.mean(axis=0)
        self.mean_pow = self.pows.mean(axis=0)


class Plotter:
    def __init__(self, fft_proc):
        self.fft_proc = fft_proc

    def run(self, plots, anims):
        for i in plots:
            method = 'plot_{}'.format(i)
            print('Running {}'.format(method))
            getattr(self, method)()
        for i in anims:
            method = 'anim_{}'.format(i)
            print('Running {}'.format(method))
            getattr(self, method)()

    def plot_mean_pow(self):
        plt.figure('mean_pow')
        plt.clf()
        plt.imshow(np.log10(np.fft.fftshift(self.fft_proc.mean_pow)))
        # plt.imshow(pow_w2)
        plt.pause(0.001)

    def _plot_pow_spectrum(self, power, height):
        plt.title('Height: {:0.2f} m'.format(height))

        power = np.fft.fftshift(power)
        r, r_pow_w2 = radial_profile(power, (128, 128))

        # N.B. don't plot (128, 128) which is zero freq term.
        # Single row along x
        plt.loglog(power[128, 129:], 'b-', label='x-dir')
        # Single col along y
        plt.loglog(power[128:, 129], 'r-', label='y-dir')
        # Radial sum.
        plt.loglog(r[1:129], r_pow_w2[1:129], 'g-', label='radial')

        x = np.linspace(4, 128, 2)
        y = 215443469 * x ** (-5 / 3)
        plt.loglog(x, y, 'k--', label='-5/3')
        plt.legend()
        plt.xlabel('scale')
        plt.ylabel('power (m$^2$ s$^{-2}$)')

    def plot_mean_pow_spectrum(self):
        plt.figure('mean_pow_spectrum')
        plt.clf()
        self._plot_pow_spectrum(self.fft_proc.mean_pow, self.fft_proc.height)
        plt.pause(0.001)
    
    def anim_time(self):
        assert self.fft_proc.lo_pass.imag.max() < 1e-12

        for i in range(self.fft_proc.cube.shape[0]):
            print(i)
            plt.figure('anim_time')
            plt.clf()
            self._plot_pow_spectrum(self.fft_proc.pows[i], self.fft_proc.height)
            plt.pause(0.001)

            lo_pass = self.fft_proc.lo_pass[i].real
            plt.figure('low_pass')
            plt.clf()
            plt.title('Low pass (cutoff={}, scale={}, sharpness={})'
                      .format(self.fft_proc.cutoff, self.fft_proc.scale, self.fft_proc.sharpness))
            plt.imshow(lo_pass)
            plt.pause(.001)

    def anim_height(self):
        for i in range(self.fft_proc.cube.shape[1]):
            print('Height: {}'.format(self.fft_proc.cube[0, i].coord('level_height').points[0]))
            plt.clf()
            plt.title('Height: {:.2f}'.format(self.fft_proc.cube[0, i].coord('level_height').points[0]))
            plt.figure('anim_height')
            self._plot_pow_spectrum(self.fft_proc.pows[i])
            plt.pause(1)


def main(method, kwargs):
    uv = iris.load(FILE_LOC)

    # Do processing.
    proc = FftProcUV(uv)
    # proc.run('single_time')
    proc.run(method, **kwargs)

    plotter = Plotter(proc)
    plotter.run(PLOTS[method], ANIMS[method])
    return plotter


if __name__ == '__main__':
    if len(sys.argv) > 1:
        method = sys.argv[1]
        kwargs = {}
        for arg in sys.argv[2:]:
            param, val = arg.split(':')
            kwargs[param] = int(val)
    else:
        method = DEFAULT_METHOD

    plotter = main(method, kwargs)
