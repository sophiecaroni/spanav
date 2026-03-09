"""
********************************************************************************
    Title: Spectral analysis utilities

    Author: Sophie Caroni
    Date of creation: 06.10.2025

    Description:
    This script contains helper functions for EEG spectral analysis.
********************************************************************************
"""
import mne
import numpy as np

from fooof import FOOOF
from yasa import bandpower_from_psd


def get_band_freqs(
        band: str,
) -> tuple[float, float]:
    """

    :param band:
    :return:
    """
    if band == 'theta':
        return 4.0, 8.0
    elif band == 'alpha':
        return 8.0, 12.0
    elif band == 'beta':
        return 12.0, 30.0
    else:
        freqs = band.split('-')
        return float(freqs[0]), float(freqs[1])


def compute_psd(
        rec: mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked,
        log_space: bool = False,
        fmin: float = 0.0,
        fmax: float = np.inf,
        test: bool = False,
        **kwargs: any,
) -> mne.time_frequency.Spectrum | tuple[np.ndarray, np.ndarray]:
    """

    :param rec:
    :param log_space:
    :param fmin:
    :param fmax: default takes all frequencies (up to Nyquist frequency)
    :param test:
    :return: PSD
    """
    kwargs.update({
        'fmin': fmin,
        'fmax': fmax,
        'picks': 'all',
    })
    if isinstance(rec, mne.io.BaseRaw):
        kwargs['reject_by_annotation'] = None  # annotations make it slower so ignore them
        if 'n_fft' not in kwargs.keys():
            times = rec.times
            n_fft = len(times) // 1000 if test else len(times)
            kwargs['n_fft'] = n_fft

    psd_obj = rec.compute_psd(**kwargs)
    if log_space:
        psd_data, freqs = psd_obj.get_data(return_freqs=True)
        psd_log = np.log10(psd_data)
        return psd_log, freqs
    else:
        return psd_obj


def get_band_power(
        psd,
        freqs,
        band: str,
        rel: bool = False,
):
    custom_bands = [
        (4, 8, "Theta"),
        (8, 12, "Alpha"),
        (38, 42, "38-42"),
    ]
    band_pws_df = bandpower_from_psd(psd, freqs, bands=custom_bands, relative=rel)
    return band_pws_df[band.capitalize()][0]


def compute_osc_snr(
        model: FOOOF,
        band: str,
):
    # Get power spectra of oscillatory and background components
    osc_psd = model.get_data(component='peak', space='linear')
    osc_psd = np.clip(osc_psd, 0, None)  # Prevent negative oscillatory power (which means "no peaks")
    bg_psd = model.get_data(component='aperiodic', space='linear')

    # Estimate oscillatory-band power
    freqs = model.freqs
    absolute = True

    osc_pw = get_band_power(osc_psd, freqs, band, rel=not absolute)
    osc_bg = get_band_power(bg_psd, freqs, band, rel=not absolute)

    # Compute and return SNR (in linear space)
    snr_linear = osc_pw / osc_bg
    return snr_linear


def model_psd(
        psd_data: np.ndarray,
        psd_freqs: np.ndarray,
        fmin_fmax: list | None = None,
        max_n_peaks: int = 6.0
):
    """

    :param psd_data: PSD powers in *linear* space
    :param psd_freqs: PSD frequency in *linear* space
    :param fmin_fmax:
    :param max_n_peaks: maximum number of peaks to fit, default 6 because it's typical for EEG (1-40 Hz)
    :return:
    """
    # Define FOOOF object
    freq_res = psd_freqs[1] - psd_freqs[0]
    peak_width_limits = (2*freq_res, 12)
    fm = FOOOF(  # expects PSD in linear space!
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks
    )

    # Fit the spectrum
    fm.fit(
        freqs=psd_freqs,
        power_spectrum=psd_data,
        freq_range=fmin_fmax
    )
    return fm



