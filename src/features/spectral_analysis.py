from typing import List, Tuple
from collections import namedtuple
import numpy as np
from scipy import interpolate
from scipy import signal
import scipy.linalg as sl
from astropy.stats import LombScargle
import spectrum
import matplotlib.pyplot as plt

from hrvanalysis import (
    get_nn_intervals, remove_outliers, remove_ectopic_beats,
    interpolate_nan_values, get_time_domain_features,
    get_csi_cvi_features, get_poincare_plot_features)

# Frequency Methods name
WELCH_METHOD = 'welch'
LOMB_METHOD = 'lomb'
AR_BURG_METHOD = 'burg'

# Detrend methods
CONSTANT_DETREND = 'constant'
SPA_DETREND = 'spa'

# Interpolation methods
# For a full list, see 'kind' parameter of https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
LINEAR_INTER = 'linear'
QUAD_INTER = 'quadratic'
CUBIC_INTER = 'cubic'

# Named Tuple for different frequency bands
VlfBand = namedtuple('Vlf_band', ['low', 'high'])
LfBand = namedtuple('Lf_band', ['low', 'high'])
HfBand = namedtuple('Hf_band', ['low', 'high'])


def get_frequency_domain_features(nn_intervals: List[float], method: str = WELCH_METHOD,
                                  sampling_frequency: float = 7.0,
                                  interpolation_method: str = LINEAR_INTER,
                                  detrend_method: str = CONSTANT_DETREND,
                                  spa_regularizer: float = 10.0,
                                  ar_order: int = 16,
                                  plot=False,
                                  vlf_band: namedtuple = VlfBand(0.0033, 0.04),
                                  lf_band: namedtuple = LfBand(0.04, 0.15),
                                  hf_band: namedtuple = HfBand(0.15, 0.40)) -> dict:
    # ----------  Compute frequency & Power of signal  ---------- #
    freq, psd = _get_freq_psd_from_nn_intervals(nn_intervals=nn_intervals, method=method,
                                                sampling_frequency=sampling_frequency,
                                                interpolation_method=interpolation_method,
                                                detrend_method=detrend_method,
                                                spa_regularizer=spa_regularizer,
                                                ar_order=ar_order,
                                                vlf_band=vlf_band, hf_band=hf_band)

    if plot:
        plt.plot(freq, psd)
        plt.show()
        plt.clf()

    # ---------- Features calculation ---------- #
    freqency_domain_features = _get_features_from_psd(freq=freq, psd=psd,
                                                      vlf_band=vlf_band,
                                                      lf_band=lf_band,
                                                      hf_band=hf_band)

    return freqency_domain_features


def _get_features_from_psd(freq: List[float], psd: List[float], vlf_band: namedtuple = VlfBand(0.0033, 0.04),
                           lf_band: namedtuple = LfBand(0.04, 0.15),
                           hf_band: namedtuple = HfBand(0.15, 0.40)) -> dict:
    # Calcul of indices between desired frequency bands
    vlf_indexes = np.logical_and(freq >= vlf_band[0], freq < vlf_band[1])
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    hf_indexes = np.logical_and(freq >= hf_band[0], freq < hf_band[1])

    # STANDARDS

    # Integrate using the composite trapezoidal rule
    lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])
    hf = np.trapz(y=psd[hf_indexes], x=freq[hf_indexes])

    # total power & vlf : Feature often used for  'long term recordings' analysis
    vlf = np.trapz(y=psd[vlf_indexes], x=freq[vlf_indexes])
    total_power = vlf + lf + hf

    lf_hf_ratio = lf / hf
    lfnu = (lf / (lf + hf)) * 100
    hfnu = (hf / (lf + hf)) * 100

    freqency_domain_features = {
        'lf': lf,
        'hf': hf,
        'lf_hf_ratio': lf_hf_ratio,
        'lfnu': lfnu,
        'hfnu': hfnu,
        'total_power': total_power,
        'vlf': vlf
    }

    return freqency_domain_features


def spa_detrending(r, mu=10):
    # https://arxiv.org/pdf/2002.06509.pdf
    N = len(r)
    D = np.zeros((N - 2, N))
    for n in range(N - 2):
        D[n, n], D[n, n + 1], D[n, n + 2] = 1.0, -2.0, 1.0
    D = mu * np.dot(D.T, D)
    for n in range(len(D)):
        D[n, n] += 1.0
    L = sl.cholesky(D, lower=True)
    Y = sl.solve_triangular(L, r, trans='N', lower=True)
    y = sl.solve_triangular(L, Y, trans='T', lower=True)
    return y, r - y


def _get_freq_psd_from_nn_intervals(nn_intervals: List[float], method: str,
                                    sampling_frequency: float = 1.0,
                                    interpolation_method: str = LINEAR_INTER,
                                    detrend_method: str = CONSTANT_DETREND,
                                    spa_regularizer: float = 10.0,
                                    ar_order: int = 16,
                                    vlf_band: namedtuple = VlfBand(0.0033, 0.04),
                                    hf_band: namedtuple = HfBand(0.15, 0.40)) -> Tuple:
    timestamps = _create_time_info(nn_intervals)

    if method in [WELCH_METHOD, AR_BURG_METHOD]:
        # ---------- Interpolation of signal ---------- #

        if interpolation_method is not None:
            funct = interpolate.interp1d(x=timestamps, y=nn_intervals, kind=interpolation_method)

            timestamps_interpolation = _create_interpolation_time(nn_intervals, sampling_frequency)
            nn_intervals = funct(timestamps_interpolation)

            plt.plot(nn_intervals)
            plt.show()

        # ---------- Remove DC Component ---------- #
        if detrend_method == CONSTANT_DETREND:
            nn_intervals = nn_intervals - np.mean(nn_intervals)
        elif detrend_method == SPA_DETREND:
            nn_intervals = spa_detrending(nn_intervals, spa_regularizer)[1]

        #  ----------  Compute Power Spectral Density  ---------- #
        if method == WELCH_METHOD:
            freq, psd = signal.welch(x=nn_intervals, fs=sampling_frequency, window='hann',
                                     nfft=4096)
        elif method == AR_BURG_METHOD:
            # psd = np.array(spectrum.pburg(nn_intervals, ar_order, NFFT=4096).psd)
            # freq = np.array(spectrum.pburg(nn_intervals, ar_order, NFFT=4096).frequencies())
            p = spectrum.pburg(nn_intervals, ar_order, NFFT=4096)
            psd = p.psd
            psd = np.log(psd)
            psd = psd/np.max(psd)

            freq = np.array(p.frequencies())

            freq_band_indices = freq <= 0.5
            pxx_band = psd[freq_band_indices]
            freq_band = freq[freq_band_indices]

            freq, psd = freq_band, pxx_band


    elif method == LOMB_METHOD:
        freq, psd = LombScargle(timestamps, nn_intervals,
                                normalization='psd').autopower(minimum_frequency=vlf_band[0],
                                                               maximum_frequency=hf_band[1])
    else:
        raise ValueError('Not a valid method. Choose between "lomb" and "welch"')

    return freq, psd


def _create_time_info(nn_intervals: List[float]) -> List[float]:
    # Convert in seconds
    nni_tmstp = np.cumsum(nn_intervals) / 1000

    # Force to start at 0
    return nni_tmstp - nni_tmstp[0]


def _create_interpolation_time(nn_intervals: List[float], sampling_frequency: int = 7) -> List[float]:
    time_nni = _create_time_info(nn_intervals)
    # Create timestamp for interpolation
    nni_interpolation_tmstp = np.arange(0, time_nni[-1], 1 / float(sampling_frequency))
    return nni_interpolation_tmstp
