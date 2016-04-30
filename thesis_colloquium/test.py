""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy.constants import G
from astropy import log as logger
from astropy.coordinates.angles import rotation_matrix
import astropy.units as u
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle
import numpy as np
import h5py

import gary.potential as gp
import gary.dynamics as gd
from gary.units import galactic, UnitSystem

# Project
from scf import SCFSimulation

orbit_style = dict(marker=None, linestyle='-', color='#2ca25f', alpha=0.5, lw=1.5)

def read_w_from_snap(snap_index, snap_file="snap.h5"):
    with h5py.File(snap_file, 'r') as f:
        _pos = f['snapshots/0/pos'][:]
        pos = np.zeros(_pos.shape)
        vel = np.zeros(_pos.shape)

        usys = UnitSystem([u.Unit(x) for x in f['units'].attrs.values()] + [u.radian])

        i = int(snap_index)
#         time[i] = f['snapshots/{}'.format(i)].attrs['t']
        pos = (f['snapshots/{}/pos'.format(i)][:]*usys['length']).to(u.kpc).value
        vel = (f['snapshots/{}/vel'.format(i)][:]*usys['length']/usys['time']).to(u.kpc/u.Myr).value
        tub = (f['snapshots/{}/tub'.format(i)][:]*usys['time']).to(u.Myr).value

    return gd.CartesianPhaseSpacePosition(pos*u.kpc, vel*u.kpc/u.Myr), tub

def read_times(snap_file="snap.h5"):
    with h5py.File(snap_file, 'r') as f:
        usys = UnitSystem([u.Unit(x) for x in f['units'].attrs.values()] + [u.radian])

        times = np.zeros(len(f['snapshots']))
        for i in range(len(f['snapshots'])):
            times[i] = f['snapshots/{}'.format(i)].attrs['t']

    times = (times*usys['time']).to(u.Gyr)
    return times

def read_cen(snap_file="snap.h5"):
    with h5py.File(snap_file, 'r') as f:
        usys = UnitSystem([u.Unit(x) for x in f['units'].attrs.values()] + [u.radian])
        sim_dt = f['parameters'].attrs['dt']
        dt = (sim_dt*usys['time']).to(u.Myr)

        cen_pos = (f['cen/pos'][:]*usys['length']).to(u.kpc)
        cen_vel = (f['cen/vel'][:]*usys['length']/usys['time']).to(u.km/u.s)
        cen_t = np.arange(cen_vel.shape[1])*dt

    return gd.CartesianOrbit(cen_pos, cen_vel, t=cen_t)

def make_action_animation(potential, bound_style, unbound_style, n_snaps, cen):
    lead_style = unbound_style.copy()
    lead_style['color'] = '#ca0020'

    trail_style = unbound_style.copy()
    trail_style['color'] = '#0571b0'

    _orbit_style = orbit_style.copy()
    _orbit_style['alpha'] = 0.4
    # _orbit_style['marker'] = 'o'
    # _orbit_style['linestyle'] = 'none'

    # snapshot times
    # snap_times = read_times()

    # energy of progenitor
    cen_E = cen.energy(potential)[0]
    # cen_actions,cen_angles,cen_freqs = potential.action_angle(cen)

    fig,axes = pl.subplots(1,3,figsize=(15,5))

    axes[0].set_xlim(4,24)
    axes[0].set_ylim(10.25, 12.25)
    axes[0].set_xlabel(r'$J_R$')
    axes[0].set_ylabel(r'$L_z$')

    axes[1].set_xlim(10.,10.8)
    axes[1].set_ylim(6.3,6.65)
    axes[1].set_xlabel(r'$\Omega_R$')
    axes[1].set_ylabel(r'$\Omega_\phi$')

    axes[2].set_xlim(0,2*np.pi)
    axes[2].set_ylim(0,2*np.pi)
    axes[2].set_xlabel(r'$\theta_R$')
    axes[2].set_ylabel(r'$\phi$')

    fig.tight_layout()

    tmp = dict(freq_ratio=None)

    def update(num, *lines):
        w,tub = read_w_from_snap(num)
        E = w.energy(potential)
        dE = E - cen_E
        bound = tub == 0
        lead = (tub > 0) & (dE < 0.)
        trail = (tub > 0) & (dE > 0.)

        actions,angles,freqs = potential.action_angle(w)
        actions = actions.to(u.Msun*u.kpc*u.km/u.s).value
        actions[1] /= 100.
        angles = angles.to(u.radian).value
        freqs = freqs.to(1/u.Gyr)

        lines[0].set_data(actions[0:2,bound])
        lines[1].set_data(actions[0:2,lead])
        lines[2].set_data(actions[0:2,trail])

        lines[3].set_data(freqs[0:2,bound])
        lines[4].set_data(freqs[0:2,lead])
        lines[5].set_data(freqs[0:2,trail])

        lines[6].set_data(angles[0:2,bound])
        lines[7].set_data(angles[0:2,lead])
        lines[8].set_data(angles[0:2,trail])

        # plot the centroid orbit
        # if len(tmp['cen_angles']) < 128:
        #     tmp['cen_angles'].pop(0)
        # tmp['cen_angles'].append(np.median(angles[0:2,bound], axis=1).tolist())
        # print(len(tmp['cen_angles']))
        # lines[9].set_data(np.array(tmp['cen_angles']).T)

        if tmp['freq_ratio'] is None:
            med_freqs = np.median(freqs[0:2,bound], axis=1)
            tmp['freq_ratio'] = med_freqs[1] / med_freqs[0]

            x = np.linspace(0,2*np.pi,32)
            for offset in np.arange(-5, 10, 1.):
                axes[2].plot(x, tmp['freq_ratio']*x + offset, zorder=-100, **_orbit_style)

        print("plotting {}".format(num), end="\r")

        return lines

    lines = []
    for j in [0,1,2]:
        l1, = axes[j].plot([], [], **bound_style)
        lines.append(l1)
        l2, = axes[j].plot([], [], **lead_style)
        lines.append(l2)
        l3, = axes[j].plot([], [], **trail_style)
        lines.append(l3)

    line_ani = animation.FuncAnimation(fig, update, n_snaps, fargs=lines,
                                       interval=60, blit=True)
    line_ani.save('stream_actionangle.mp4', codec="libx264", extra_args=['-pix_fmt', 'yuv420p'], dpi=150, bitrate=-1)

def make_xyz_animation(potential, bound_style, unbound_style, n_snaps, cen):

    lead_style = unbound_style.copy()
    lead_style['color'] = '#ca0020'

    trail_style = unbound_style.copy()
    trail_style['color'] = '#0571b0'

    # snapshot times
    snap_times = read_times()

    # energy of progenitor
    cen_E = cen.energy(potential)[0]

    fig,axes = pl.subplots(1,2,figsize=(10,5))

    # grid = np.linspace(-20,20,64)
    # potential.plot_contours(grid=(grid,grid,0), cmap='Greys', ax=axes[0])
    axes[0].add_patch(Ellipse([0,0], 20., 2., facecolor='#aaaaaa'))
    axes[0].add_patch(Circle([0,0], 2., facecolor='#aaaaaa'))

    axes[0].set_xlim(-20, 20)
    axes[0].set_ylim(-20, 20)
    axes[0].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$y$')

    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(-10, 10)
    axes[1].set_xlabel(r'$x-x_p$')
    axes[1].set_ylabel(r'$y-y_p$')

    fig.tight_layout()

    def update(num, *lines):
        w,tub = read_w_from_snap(num)
        E = w.energy(potential)
        dE = E - cen_E
        bound = tub == 0
        lead = (tub > 0) & (dE < 0.)
        trail = (tub > 0) & (dE > 0.)

        lines[0].set_data(w.pos[0:2,bound].value)
        lines[1].set_data(w.pos[0:2,lead].value)
        lines[2].set_data(w.pos[0:2,trail].value)

        # _cen = w.pos[0:2,bound].value.mean(axis=1)

        # plot the progenitor orbit
        idx = np.abs(snap_times[num] - cen.t).argmin()
        lines[6].set_data(cen.pos[0:2,:idx].value)

        _cen_pos = cen.pos[0:2,idx].value
        _cen_vel = cen.vel[0:2,idx].value
        _cen_vel = _cen_vel / np.linalg.norm(_cen_vel)
        ang = np.arctan2(_cen_vel[1], _cen_vel[0])
        R = rotation_matrix(ang*u.radian, 'z')[:2,:2]

        dcen = R.dot(w.pos[0:2,bound].value - _cen_pos[:,None])
        lines[3].set_data(dcen)

        dcen = R.dot(w.pos[0:2,lead].value - _cen_pos[:,None])
        lines[4].set_data(dcen)

        dcen = R.dot(w.pos[0:2,trail].value - _cen_pos[:,None])
        lines[5].set_data(dcen)

        return lines

    lines = []
    for j in [0,1]:
        l1, = axes[j].plot([], [], **bound_style)
        lines.append(l1)

        l2, = axes[j].plot([], [], **lead_style)
        lines.append(l2)

        l3, = axes[j].plot([], [], **trail_style)
        lines.append(l3)

    l, = axes[0].plot([], [], zorder=-1000, **orbit_style)
    lines.append(l)

    line_ani = animation.FuncAnimation(fig, update, n_snaps, fargs=lines,
                                       interval=60, blit=True)
    line_ani.save('stream_xy.mp4', codec="libx264", extra_args=['-pix_fmt', 'yuv420p'], dpi=150, bitrate=-1)

def main():
    # read SCFBI file
    skip = 1
    names = ['m','x','y','z','vx','vy','vz']
    bodies_filename = '/Users/adrian/projects/scf/fortran/SCFBI'
    bodies = np.genfromtxt(bodies_filename, dtype=None, names=names,
                           skip_header=skip)

    b = gd.CartesianPhaseSpacePosition(pos=bodies[['x','y','z']].view(np.float64).reshape(-1,3).T,
                                       vel=bodies[['vx','vy','vz']].view(np.float64).reshape(-1,3).T)
    b = b[:10000]

    w0 = gd.CartesianPhaseSpacePosition(pos=[15.,0,0]*u.kpc,
                                        vel=[0,75.,0]*u.km/u.s)

    rs = 10.*u.kpc
    M = ((220.*u.km/u.s)**2 * rs / G).to(u.Msun)
    potential = gp.IsochronePotential(m=M, b=rs, units=galactic)

    if not os.path.exists('snap.h5'):
        sim = SCFSimulation(b, potential, 2.5e4*u.Msun, 10*u.pc,
                            snapshot_filename='snap.h5')
        sim.run(w0, 0.25*sim.units['time'], 16384, n_snapshot=32,
                n_recenter=128, n_tidal=256)

    with h5py.File('snap.h5', 'r') as f:
        n_snaps = len(f['snapshots/'])
        print("{} snapshots".format(n_snaps))

    cen = read_cen()

    pl.style.use('apw-notebook')

    bound_style = dict(marker=',', alpha=0.75, color='#666666', linestyle='none')
    unbound_style = dict(marker='o', alpha=0.8, linestyle='none', markersize=2)

    make_action_animation(potential, bound_style, unbound_style, n_snaps, cen)
    # make_xyz_animation(potential, bound_style, unbound_style, n_snaps, cen)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main()
