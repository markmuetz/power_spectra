"""Works out 2D FT of either hor. speed or w

Input data come from UM: shape is: time, height, lat, lon (this is checked).
Because the data domain is from e.g. 0 - 256 km, no need to pre-fftshift.
"""
import os
import sys

import matplotlib
matplotlib.use('agg')

import numpy as np
import scipy
import pylab as plt
import iris

from omnium.utils import get_cube

EXPTS = ['km1_large_dom_no_wind', 'm500_large_dom_no_wind']
MODES = ['uv', 'w']

FILE_LOC_FMT = '/home/n02/n02/mmuetz/work/cylc-run/u-ax548/share/data/history/{}/atmosa_pc072.nc'
DEFAULT_METHOD = 'loop_time'

PLOTS = {
    'loop_time': ['mean_pow_spectrum'],
    'loop_height': [],
}

ANIMS = {
    'loop_time': [],
    'loop_height': ['height'],
}

LEVEL_HEIGHT_INDEX = 15
TIME_INDEX = -1

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


class FftProcUVW:
    def __init__(self, mode, cubes):
        self.mode = mode
        self.cubes = cubes
        self.u = get_cube(self.cubes, 0, 2)
        coords = self.u.coords()
        # N.B. dims are time, height, lat, lon
        assert coords[2].name() == 'grid_latitude'
        assert coords[3].name() == 'grid_longitude'
        self.v = get_cube(self.cubes, 0, 3)
        self.w = get_cube(self.cubes, 0, 150)

        self.nx = self.u.shape[3]
        self.ny = self.u.shape[2]
        assert self.nx == self.ny
        self.half_nx = self.nx // 2

        if mode == 'uv':
            self.cube = self.u
        elif mode == 'w':
            self.cube = self.w

    def run(self, method, **args):
        """Dispatch on method"""
        print('Running {}'.format(method))
        return getattr(self, method)(**args)
    
    def single_time(self, time_index=TIME_INDEX, level_height_index=LEVEL_HEIGHT_INDEX):
        if self.mode == 'uv':
            u_data = self.u[time_index, level_height_index].data
            v_data = self.v[time_index, level_height_index].data
            data = np.sqrt(u_data**2 + v_data**2)
        else:
            data = self.w[time_index, level_height_index].data

        self.ft = np.fft.fft2(data)
        self.mean_pow = np.abs(self.ft)**2

    def loop_time(self, level_height_index=LEVEL_HEIGHT_INDEX):
        self.height = self.cube[0, level_height_index].coord('level_height').points[0]
        print('Height: {:.2f}'.format(self.height))

        if self.mode == 'uv':
            u_data = self.u[:, level_height_index].data
            v_data = self.v[:, level_height_index].data
            data = np.sqrt(u_data**2 + v_data**2)
        else:
            data = self.w[:, level_height_index].data

        # N.B. data.shape[0] is latitude, i.e. y-dir
        # data.shape[1] is longitude, i.e. x-dir
        self.fts = np.fft.fft2(data)
        self.pows = np.abs(self.fts)**2

        self.mean_ft = self.fts.mean(axis=0)
        self.mean_pow = self.pows.mean(axis=0)

    def loop_height(self, time_index=TIME_INDEX):
        self.time = self.cube[time_index].coord('time').points[0]
        print('Time: {}'.format(self.time))

        if self.mode == 'uv':
            u_data = self.u[time_index, :].data
            v_data = self.v[time_index, :].data
            data = np.sqrt(u_data**2 + v_data**2)
        else:
            data = self.w[time_index, :].data

        self.fts = np.fft.fft2(data)
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
    def __init__(self, expt, mode, fft_proc):
        self.expt = expt
        self.mode = mode
        self.fft_proc = fft_proc
        self.half_nx = self.fft_proc.half_nx

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
        plt.title('{}, {}, Height: {:0.2f} m'.format(self.expt, self.mode, height))

        power = np.fft.fftshift(power)
        r, r_pow_w2 = radial_profile(power, (self.half_nx, self.half_nx))

        # N.B. don't plot (self.half_nx, self.half_nx) which is zero freq term.
        # Rem. e.g. power.shape[0] is y.
        # Single col along y
        power_x_slice = power[self.half_nx + 1:, self.half_nx]
        # Single row along x
        power_y_slice = power[self.half_nx, self.half_nx + 1:]
        plt.loglog(range(1, self.half_nx), power_x_slice, 'b-', label='x-dir')
        plt.loglog(range(1, self.half_nx), power_y_slice, 'r-', label='y-dir')
        # Radial sum.
        plt.loglog(r[1:self.half_nx + 1], r_pow_w2[1:self.half_nx + 1], 'g-', label='radial')

        x = np.linspace(4, self.half_nx, 2)
        y = 215443469 * x ** (-5 / 3)
        plt.loglog(x, y, 'k--', label='-5/3')
        plt.legend()
        plt.xlabel('scale')
        plt.ylabel('power (m$^2$ s$^{-2}$)')

    def plot_mean_pow_spectrum(self):
        plt.figure('mean_pow_spectrum')
        plt.clf()
        self._plot_pow_spectrum(self.fft_proc.mean_pow, self.fft_proc.height)

        fig_dir = os.path.join(os.path.dirname(FILE_LOC_FMT.format(self.expt)), 'figs')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, self.expt + '_' + self.mode + '_pow_spec.png'))
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


def main(expt, mode, method, kwargs):
    cubes = iris.load(FILE_LOC_FMT.format(expt))

    # Do processing.
    proc = FftProcUVW(mode, cubes)
    # proc.run('single_time')
    proc.run(method, **kwargs)

    plotter = Plotter(expt, mode, proc)
    plotter.run(PLOTS[method], ANIMS[method])
    return cubes, plotter


if __name__ == '__main__':
    if len(sys.argv) > 1:
        expt = sys.argv[1]
        mode = sys.argv[2]
        method = sys.argv[3]
        kwargs = {}
        for arg in sys.argv[4:]:
            param, val = arg.split(':')
            kwargs[param] = int(val)
        cubes, plotter = main(expt, mode, method, kwargs)
    else:
        kwargs = {}
        method = DEFAULT_METHOD
        for expt in EXPTS:
            for mode in MODES:
                print('{}, {}'.format(expt, mode))
                cubes, plotter = main(expt, mode, method, kwargs)
