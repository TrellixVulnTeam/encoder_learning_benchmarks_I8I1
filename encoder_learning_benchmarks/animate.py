#  Encoder Learning Benchmark
#  Copyright (C) 2020 Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.patches
import matplotlib.lines
import matplotlib.colors

import scipy.interpolate

import numpy as np
import os, shutil
import random
import subprocess
import threading

from . import utils

def pick(dict_, *keys):
    return [dict_[key] for key in keys]


class Animation:
    def __init__(self,
                 n_dim_hidden,
                 n_dim_in,
                 n_dim_out,
                 name=None,
                 dpi=150,
                 fps=30,
                 sec_per_smpl=1e-3,
                 gridres=41,
                 radius=1.5,
                 err_range=0.25):
        # Create a pseudo-random name if none is given
        if name is None:
            import datetime
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
            rnd = "".join(random.choice('0123456789abcdef') for _ in range(8))
            name = "anim_{}_{}".format(now, rnd)

        # Copy the input and output dimensions. For now we only support a
        # limited set input/output dimension configurations
        self.n_dim_hidden = int(n_dim_hidden)
        self.n_dim_in = int(n_dim_in)
        self.n_dim_out = int(n_dim_out)
        assert self.n_dim_hidden > 0
        assert self.n_dim_in == 2
        assert self.n_dim_out == 1

        # Copy all other parameters
        self.name = name
        self.dirname = ".{}_tmp".format(name)
        self.dpi = dpi
        self.fps = fps
        self.sec_per_smpl = sec_per_smpl
        self.smpls_per_frame = int(1.0 / (sec_per_smpl * fps))
        self.gridres = gridres
        self.radius = radius
        self.err_range = err_range
        assert self.smpls_per_frame > 0

        # Reset all internal buffers
        self.reset()

    def reset(self):
        self.i_frame = 0
        self.n_smpls = 0
        self.last_epoch = None
        self.buf_xs_trn_batch = None
        self.buf_ys_trn_batch = None
        self.xs_trn = None
        self.ys_trn = None
        self.xss_trn = None
        self.yss_trn = None

    def __enter__(self):
        self.reset()
        os.makedirs(self.dirname)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            subprocess.run(['ffmpeg',
               '-r', str(self.fps),
                '-i', os.path.join(self.dirname, "frame_%05d.png"),
                '-vf', 'scale=out_color_matrix=bt709:flags=full_chroma_int',
                '-pix_fmt', 'yuv420p',
                '-preset', 'slow',
                '-crf', '22',
                self.name + '.mp4'])
        shutil.rmtree(self.dirname)

    def _render_frame(self, title, xs_trn_batch, ys_trn_batch, D, network,
                      eval_net_and_errs):
        # Compute the output error for the background contour plot
        _, errs = eval_net_and_errs(self.xss_trn, self.yss_trn, D)
        errs = errs.reshape((self.xs_trn.size, self.ys_trn.size))

        # Plot the current approximation error as a coloured contour plot
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        ax.contourf(self.xs_trn,
                    self.ys_trn,
                    errs,
                    vmin=-self.err_range,
                    vmax=self.err_range,
                    cmap='RdBu')

        # Plot all the input samples as points. Their colour corresponds to the
        # error at that point in the network
        As, errs = eval_net_and_errs(xs_trn_batch, ys_trn_batch, D)
        ax.scatter(xs_trn_batch[:, 0],
                   xs_trn_batch[:, 1],
                   c=errs[:, 0],
                   marker='o',
                   cmap='RdBu',
                   edgecolors='k',
                   linewidths=0.75,
                   vmin=-self.err_range,
                   vmax=self.err_range)

        # Visualize the network
        utils.visualise_network(ax, network, np.mean(As, axis=0))

        # Render a bit more than just the data range
        r = self.radius
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)

        # Decorate the axes
        ax.set_frame_on(False)
        rr = np.floor(2 * r) / 2
        ticks = np.linspace(-rr, rr, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect(1)
        ax.set_title(title)

        # Save the plotted data to the output
        out = os.path.join(self.dirname, 'frame_{:05d}.png'.format(self.i_frame))
        fig.patch.set_facecolor('white')
        fig.savefig(out, 
            dpi=self.dpi,
            transparent=False)
        plt.close(fig)

        self.i_frame += 1


    def __call__(self, data):
        # Destructure the given data
        i_epoch, i_batch, n_epochs, n_batches, xs_trn_batch, ys_trn_batch, \
        xs_trn, ys_trn, D, network, eval_net, eval_net_and_errs = pick(
            data,
            "i_epoch", "i_batch", "n_epochs", "n_batches",
            "xs_trn_batch", "ys_trn_batch", "xs_trn", "ys_trn",
            "D", "network", "eval_net", "eval_net_and_errs")

        # Make sure that all arrays have the right dimensionality
        assert (xs_trn_batch.ndim == 2) and (xs_trn_batch.shape[1]
                                             == self.n_dim_in)
        assert (ys_trn_batch.ndim == 2) and (ys_trn_batch.shape[1]
                                             == self.n_dim_out)
        assert (xs_trn.ndim == 2) and (xs_trn.shape[1] == self.n_dim_in)
        assert (ys_trn.ndim == 2) and (ys_trn.shape[1] == self.n_dim_out)

        # Split the given training samples into batches of size smpl_per_frame
        def append(tar, src):
            if tar is None:
                return np.copy(src)
            return np.concatenate((tar, src), axis=0)

        # When switching epochs, interpolate the full training set onto a
        # grid
        if i_epoch != self.last_epoch:
            r = self.radius / np.sqrt(2)
            self.xs_trn = np.linspace(-r, r, self.gridres)
            self.ys_trn = np.linspace(-r, r, self.gridres)
            xss, yss = np.meshgrid(self.xs_trn, self.ys_trn)
            self.xss_trn = np.array((xss.flatten(), yss.flatten())).T
            self.yss_trn = scipy.interpolate.griddata(xs_trn, ys_trn,
                                                      self.xss_trn)
            self.last_epoch = i_epoch

        # Append the incoming samples to the buffer
        self.buf_xs_trn_batch = append(self.buf_xs_trn_batch, xs_trn_batch)
        self.buf_ys_trn_batch = append(self.buf_ys_trn_batch, ys_trn_batch)

        # Iterate over all frames
        N = self.smpls_per_frame
        while self.buf_xs_trn_batch.shape[0] > N:
            # Fetch training samples for the next frame
            xs_trn_batch = self.buf_xs_trn_batch[:N]
            ys_trn_batch = self.buf_ys_trn_batch[:N]
            self.buf_xs_trn_batch = self.buf_xs_trn_batch[N:]
            self.buf_ys_trn_batch = self.buf_ys_trn_batch[N:]

            # Assemble the frame title
            self.n_smpls += N
            title = "Epoch: {}/{}; Time: {:0.2f}s".format(
                i_epoch + 1, n_epochs, self.n_smpls * self.sec_per_smpl)

            # Render the frame
            self._render_frame(title, xs_trn_batch, ys_trn_batch, D, network,
                               eval_net_and_errs)

