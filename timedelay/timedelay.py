# -*- coding: utf-8 -*-

from lightkurve import KeplerLightCurveFile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from astropy.stats import LombScargle

__all__ = ["timedelay"]

def timedelay(target, file=False):

    if file:
        t, y = np.loadtxt(target, usecols=(0,1)).T
    else:
        t, y = get_lightcurve(target)

    # Get frequency estimate. Should be mixed with Lomb Scargle below
    nu = estimate_frequencies(t, y)

    uHz_conv = 1e-6 * 24 * 60 * 60  # Factor to convert between day^-1 and uHz

    # Fuzzy nyquist
    nyquist = 0.5 / np.median(np.diff(t))
    nyquist = nyquist / uHz_conv

    # Overshoot by 2 * nyquist
    freq_uHz = np.linspace(1e-2, nyquist * 2, 100000)
    freq = freq_uHz * uHz_conv

    model = LombScargle(t, y)
    power = model.power(freq, method="fast", normalization="psd")

    fig, ax = plt.subplots(2,1,figsize=[10,10])
    ax[0].plot(freq_uHz * uHz_conv, np.sqrt(power), "k", linewidth=0.5)
    ax[0].set_xlim(freq_uHz[0] * uHz_conv, freq_uHz[-1] * uHz_conv)
    ax[0].set_xlabel(r"frequency $[d^{-1}]$")
    ax[0].axvline(nyquist * uHz_conv, c='r')
    ax[0].set_ylabel("Amplitude")

    # This is the absolute worst thing I have ever written
    for current_freq in nu[:7]:
        times_0 = t[0]
        ax[0].scatter(current_freq, 0.3)

        phase = []
        time_mid = []

        mod = []
        time_mod = []
        for i, j in zip(t, y):
            mod.append(j)
            time_mod.append(i)

            if i-times_0 > 10:
                sampling = 1/np.median(np.diff(time_mod))
                phase.append(ft_single(time_mod, mod, current_freq))
                time_mid.append(np.mean(time_mod))
                times_0 = i
                mod = []
                time_mod = []

        phase -= np.mean(phase)
        ax[1].scatter(time_mid, phase / (2*np.pi*(current_freq / uHz_conv * 1e-6)),
                    alpha=1, s=8)

    ax[1].set_xlabel('Time')
    ax[1].set_ylabel(r'$\tau$')
    plt.show()

def ft_single(x, y, freq, verbose=False):
    x = np.asarray(x)
    y = np.asarray(y)

    ft_real, ft_imag, power, fr = [], [], [], []
    len_x = len(x)
    ft_real.append(0.0)
    ft_imag.append(0.0)
    omega = 2.0 * np.pi * freq

    # Kinda slow. Should be vectorized
    for i in range(len_x):
        expo = omega * x[i]
        c = np.cos(expo)
        s = np.sin(expo)
        ft_real[-1] += y[i] * c
        ft_imag[-1] += y[i] * s
    return np.arctan(ft_imag[0]/ft_real[0])

def get_lightcurve(target):
    lcs = KeplerLightCurveFile.from_archive(target, quarter='all', 
                                                    cadence='long')
    lc = lcs[0].PDCSAP_FLUX.remove_nans()
    lc.flux = -2.5 * np.log10(lc.flux)
    lc.flux = lc.flux - np.average(lc.flux)
    for i in lcs[1:]:
        i = i.PDCSAP_FLUX.remove_nans()
        i.flux = -2.5 * np.log10(i.flux)
        i.flux = i.flux - np.average(i.flux)
        lc = lc.append(i)
    return lc.time, lc.flux

def estimate_frequencies(x, y, max_peaks=9, oversample=4.0):
    tmax = x.max()
    tmin = x.min()
    dt = np.median(np.diff(x))
    df = 1.0 / (tmax - tmin)
    ny = 0.5 / dt

    freq = np.arange(df, 2 * ny, df / oversample)
    power = LombScargle(x, y).power(freq)

    # Find peaks
    peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_inds = np.arange(1, len(power)-1)[peak_inds]
    peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
    peaks = []
    for j in range(max_peaks):
        i = peak_inds[0]
        freq0 = freq[i]
        alias = 2.0*ny - freq0

        m = np.abs(freq[peak_inds] - alias) > 25*df
        m &= np.abs(freq[peak_inds] - freq0) > 25*df

        peak_inds = peak_inds[m]
        peaks.append(freq0)
    peaks = np.array(peaks)

    # Optimize the model
    T = tf.float64
    t = tf.constant(x, dtype=T)
    f = tf.constant(y, dtype=T)
    nu = tf.Variable(peaks, dtype=T)
    arg = 2*np.pi*nu[None, :]*t[:, None]
    D = tf.concat([tf.cos(arg), tf.sin(arg),
                   tf.ones((len(x), 1), dtype=T)],
                  axis=1)

    # Solve for the amplitudes and phases of the oscillations
    DTD = tf.matmul(D, D, transpose_a=True)
    DTy = tf.matmul(D, f[:, None], transpose_a=True)
    w = tf.linalg.solve(DTD, DTy)
    model = tf.squeeze(tf.matmul(D, w))
    chi2 = tf.reduce_sum(tf.square(f - model))

    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2, [nu],
                                                 method="L-BFGS-B")
    with tf.Session() as sess:
        sess.run(nu.initializer)
        opt.minimize(sess)
        return sess.run(nu)

def timedelay_main():
    import argparse
    parser = argparse.ArgumentParser(
            description=('Calculate and show time delays for Kepler photometry'))
    parser.add_argument('target', help='Name of input target or filepath', type=str)
    parser.add_argument('--file', action='store_true')
    args = parser.parse_args()
    timedelay(args.target, args.file)