import numpy as np
from scipy.linalg import hankel
from scipy.signal import convolve2d
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib


def get_bispectral_features(rrs, **kwargs):
    ll_lower, ll_upper = 0.04, 0.15
    roi_lower, roi_upper = 0.04, 0.4
    psd, freq = bispectrum(rrs, **kwargs)

    omega_idx = []
    omega_diag_idx = []
    ll_idx = []
    roi_idx = []
    for i, f1 in enumerate(freq):
        for j, f2 in enumerate(freq):
            if f1 >= f2:

                if (ll_lower <= f1 <= ll_upper) and (ll_lower <= f2 <= ll_upper):
                    ll_idx.append((i, j))

                if (roi_lower <= f1 <= roi_upper) and (roi_lower <= f2 <= roi_upper):
                    roi_idx.append((i, j))

                if f1 + f2 <= 1:
                    omega_idx.append((i, j))
                    if i == j:
                        omega_diag_idx.append((i, j))

    bisp_features = {
        'bispen': bisp_entropy(psd, omega_idx),
        'bispen2': bisp_entropy_squared(psd, omega_idx),
        'h1': bisp_log_sum(psd, omega_idx),
        'h2': bisp_log_sum(psd, omega_diag_idx),
        'h3': spec_mom_first_order(psd, omega_diag_idx),
        'h4': spec_mom_second_order(psd, omega_diag_idx),
        'll_h1': bisp_log_sum(psd, ll_idx),
        'roi_wcob_f2': wcob_f2(psd, roi_idx),
    }

    return bisp_features


def bisp_entropy(psd, region):
    norm = 0.0
    for (i, j) in region:
        norm += psd[j, i]

    bispen = 0.0
    for (i, j) in region:
        p_n = psd[j, i] / norm
        bispen += p_n * np.log2(p_n)
    return -(bispen / np.log2(len(region)))


def bisp_entropy_squared(psd, region):
    norm = 0.0
    for (i, j) in region:
        norm += psd[j, i] ** 2

    bispen = 0.0
    for (i, j) in region:
        q_n = psd[j, i] ** 2 / norm
        bispen += q_n * np.log2(q_n)

    return -(bispen / np.log2(len(region)))


def bisp_log_sum(psd, region):
    """Sum of logarithmic amplitudes of the bispectrum"""
    h1 = 0.0
    for (i, j) in region:
        h1 += np.log2(psd[j, i])
    return h1


def spec_mom_first_order(psd, region):
    """First order spectral moment of the amplitudes of diagonal elements in the bispectrum"""
    h3 = 0.0
    for (i, j) in region:
        if i == j:
            h3 += i * np.log2(psd[j, i])
    return h3


def spec_mom_second_order(psd, region):
    """Second order spectral moment of the amplitudes of diagonal elements in the bispectrum"""
    h3 = spec_mom_first_order(psd, region)
    h4 = 0.0
    for (i, j) in region:
        if i == j:
            h4 += (i - h3) ** 2 * np.log2(psd[j, i])
    return h4


def wcob_f1(psd, region):
    norm = 0.0
    f1_center = 0.0
    for (i, j) in region:
        norm += psd[j, i]
        f1_center += i * psd[j, i]
    return f1_center / norm


def wcob_f2(psd, region):
    norm = 0.0
    f2_center = 0.0
    for (i, j) in region:
        norm += psd[j, i]
        f2_center += j * psd[j, i]
    return f2_center / norm


# https://github.com/synergetics/spectrum/blob/master/src/conventional/bispectrumi.py
# Ported matlab code from :
#   https://nl.mathworks.com/matlabcentral/fileexchange/3013-hosa-higher-order-spectral-analysis-toolbox
def bispectrum(y, nlag=None, nsamp=None, overlap=None, flag='biased', nfft=None, wind=None, normalize=False, plot=False):
    """
    Parameters:
        y             - records vector or time-series
        nlag        - number of lags to compute [must be specified]
        segsamp - samples per segment        [default: row dimension of y]
        overlap - percentage overlap         [default = 0]
        flag        - 'biased' or 'unbiased' [default is 'unbiased']
        nfft        - FFT length to use            [default = 128]
        wind        - window function to apply:
                            if wind=0, the Parzen window is applied (default)
                            otherwise the hexagonal window with unity values is applied.

    Output:
        Bspec     - estimated bispectrum    it is an nfft x nfft array
                            with origin at the center, and axes pointing down and to the right
        waxis     - frequency-domain axis associated with the bispectrum.
                        - the i-th row (or column) of Bspec corresponds to f1 (or f2)
                            value of waxis(i).
    """

    if len(y.shape) == 1:
        y = y.reshape((1, y.shape[0]))

    (ly, nrecs) = y.shape
    if ly == 1:
        y = y.reshape(1, -1)
        ly = nrecs
        nrecs = 1

    if not overlap:
        overlap = 0
    overlap = min(99, max(overlap, 0))
    if nrecs > 1:
        overlap = 0
    if not nsamp:
        nsamp = ly
    if nsamp > ly or nsamp <= 0:
        nsamp = ly
    if not 'flag':
        flag = 'biased'
    if not nfft:
        nfft = 128
    if not wind:
        wind = 0

    nlag = min(nlag, nsamp - 1)
    if nfft < 2 * nlag + 1:
        nfft = nextpow2(nsamp)

    # create the lag window
    Bspec = np.zeros([nfft, nfft])
    if wind == 0:
        indx = np.array([range(1, nlag + 1)]).T
        window = make_arr((1, np.sin(np.pi * indx / nlag) / (np.pi * indx / nlag)), axis=0)
    else:
        window = np.ones([nlag + 1, 1])
    window = make_arr((window, np.zeros([nlag, 1])), axis=0)

    # cumulants in non-redundant region
    overlap = np.fix(nsamp * overlap / 100)
    nadvance = nsamp - overlap
    nrecord = np.fix((ly * nrecs - overlap) / nadvance)

    c3 = np.zeros([nlag + 1, nlag + 1])
    ind = np.arange(nsamp)
    y = y.ravel(order='F')

    s = 0
    for k in range(int(nrecord)):
        x = y[ind].ravel(order='F')
        x = x - np.mean(x)
        ind = ind + int(nadvance)

        for j in range(int(nlag + 1)):
            z = x[range(int(nsamp - j))] * x[range(j, int(nsamp))]
            for i in range(j, nlag + 1):
                Sum = np.dot(z[range(int(nsamp - i))].T, x[range(i, int(nsamp))])
                if flag == 'biased':
                    Sum = Sum / nsamp
                else:
                    Sum = Sum / (nsamp - i)
                c3[i, j] = c3[i, j] + Sum

    c3 = c3 / nrecord

    # cumulants elsewhere by symmetry
    c3 = c3 + np.tril(c3, -1).T  # complete I quadrant
    c31 = c3[1:nlag + 1, 1:nlag + 1]
    c32 = np.zeros([nlag, nlag])
    c33 = np.zeros([nlag, nlag])
    c34 = np.zeros([nlag, nlag])
    for i in range(nlag):
        x = c31[i:nlag, i]
        c32[nlag - 1 - i, 0:nlag - i] = x.T
        c34[0:nlag - i, nlag - 1 - i] = x
        if i + 1 < nlag:
            x = np.flipud(x[1:len(x)])
            c33 = c33 + np.diag(x, i + 1) + np.diag(x, -(i + 1))

    c33 = c33 + np.diag(c3[0, nlag:0:-1])

    cmat = make_arr(
        (make_arr((c33, c32, np.zeros([nlag, 1])), axis=1),
         make_arr((make_arr((c34, np.zeros([1, nlag])), axis=0), c3), axis=1)),
        axis=0
    )

    # apply lag-domain window
    wcmat = cmat
    if wind != -1:
        indx = np.arange(-1 * nlag, nlag + 1).T
        window = window.reshape(-1, 1)
        for k in range(int(-nlag), int(nlag + 1)):
            wcmat[:, k + nlag] = (cmat[:, k + nlag].reshape(-1, 1) * \
                                  window[abs(indx - k)] * \
                                  window[abs(indx)] * \
                                  window[abs(k)]).reshape(-1, )

    # compute 2d-fft, and shift and rotate for proper orientation
    Bspec = np.fft.fft2(wcmat, (nfft, nfft))
    Bspec = np.fft.fftshift(Bspec)  # axes d and r; orig at ctr

    if nfft % 2 == 0:
        waxis = np.transpose(np.arange(-1 * nfft / 2, nfft / 2)) / nfft
    else:
        waxis = np.transpose(np.arange(-1 * (nfft - 1) / 2, (nfft - 1) / 2 + 1)) / nfft

    bisp = abs(Bspec)
    freq = np.linspace(0, 1, len(bisp))
    if normalize:
        bisp = bisp / np.max(bisp)
    if plot:
        plot_psd(bisp, freq)
    return (bisp, freq)

def plot_psd(psd, freq):
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cs = plt.contourf(freq, freq, abs(psd), 4, cmap=plt.cm.Spectral_r)
    norm = matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
    sm.set_array([])
    plt.colorbar(sm, format=fmt)

    plt.xlabel('f1 (Hz)')
    plt.ylabel('f2 (Hz)')
    plt.show()

import numpy as np
import sys
import os


def nextpow2(num):
    '''
    Returns the next highest power of 2 from the given value.
    Example
    -------
    >>>nextpow2(1000)
    1024
    >>nextpow2(1024)
    2048

    Taken from: https://github.com/alaiacano/frfft/blob/master/frfft.py
    '''

    npow = 2
    while npow <= num:
        npow = npow * 2
    return npow


def flat_eq(x, y):
    """
    Emulate MATLAB's assignment of the form
    x(:) = y
    """
    z = x.reshape(1, -1)
    z = y
    return z.reshape(x.shape)


def make_arr(arrs, axis=0):
    """
    Create arrays like MATLAB does
    python                                                 MATLAB
    make_arr((4, range(1,10)), axis=0) => [4; 1:9]
    """
    a = []
    ctr = 0
    for x in arrs:
        if len(np.shape(x)) == 0:
            a.append(np.array([[x]]))
        elif len(np.shape(x)) == 1:
            a.append(np.array([x]))
        else:
            a.append(x)
        ctr += 1
    return np.concatenate(a, axis)


def shape(o, n):
    """
    Behave like MATLAB's shape
    """
    s = o.shape
    if len(s) < n:
        x = tuple(np.ones(n-len(s)))
        return s + x
    else:
        return s


def here(f=__file__):
    """
    This script's directory
    """
    return os.path.dirname(os.path.realpath(f))