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
import pandas as pd
from mne.time_frequency import EpochsSpectrum, Spectrum, BaseTFR
from fooof import FOOOF
from yasa import bandpower_from_psd

PSD = EpochsSpectrum | Spectrum


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
) -> PSD:
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
        psd_obj = psd_obj.copy()
        psd_obj._data = np.log10(psd_obj._data)
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


def get_psd_kwargs(sfreq: float = 250) -> dict:
    return dict(
        fmin=2,
        fmax=60,
        method="welch",
        n_fft=sfreq,
        n_per_seg=sfreq,
        # windows_length will be n_per_seg / sfreq, so setting n_per_seg=sfreq will make windows_length 1s
        n_overlap=int(sfreq / 2),  # 50% overlap
        window="hamming"  # common
    )


def spectral_bl_corr_from_df(input_df: pd.DataFrame, objs_name: str, objs_col: str, bl_name: str) -> dict:
    """
    Use one MNE's spectral object to baseline-correct all other spectral objects in a dataframe.
    :param input_df: input dataframe
    :param objs_name: column name defining the different types of spectral objects
    :param objs_col: column name containing the spectral objects (TFR or Spectrum)
    :param bl_name: type of the object to use as baseline (value in col_name)
    :return: output dictionary of records containing baseline-corrected objects along with their new name (bl + old object name)
    """
    # Assert needed columns are present in input_df
    needed_cols = [objs_name, objs_col]
    missing = [col for col in needed_cols if col not in input_df.columns]
    if missing:
        raise ValueError(
            f"Expected both {objs_name} and {objs_col} to be columns of input_df, "
            f"missing {missing}"
        )

    # Assert type spectral objects in objs_col are all of the same type of either BaseTFR or Spectrum
    valid_types = (BaseTFR, Spectrum)
    present = set(type(x) for x in input_df[objs_col])
    invalid = [x for x in input_df[objs_col] if not isinstance(x, valid_types)]
    if invalid or len(present) > 1:
        raise TypeError(
            f"Expected all objects in '{objs_col}' to be instances of either {valid_types}, "
            f"got {present} present types"
        )

    # Assert one and only one object to use as baseline is present
    bl_obj_rows = input_df[input_df[objs_name] == bl_name][objs_col]
    if len(bl_obj_rows) != 1:
        raise ValueError(
            f"Expected one object to use as baseline, got '{len(bl_obj_rows)}'"
            f"\n\t -> {bl_obj_rows = }"
        )

    # Extract object to use as baseline
    bl_obj = bl_obj_rows.iloc[0]

    # Compute baseline
    bl = bl_obj._data.mean(axis=-1, keepdims=True)  # last dimension is time in TFR and frequencies in Spectrum objects

    # Baseline correct all other objects using bl_obj
    bl_corr_records = {objs_name: [], objs_col: []}
    for _, row in input_df[input_df[objs_name] != bl_name].iterrows():
        obj_name = row[objs_name]
        obj = row[objs_col]

        # Baseline-correct (similar to mode='percent' of mne.baseline.rescale https://mne.tools/stable/generated/mne.baseline.rescale.html)
        obj_bl = obj.copy()
        obj_bl._data = (obj._data - bl) / bl * 100

        # Append new object and its name to dict
        bl_corr_records[objs_name].append(f'bl{obj_name}')
        bl_corr_records[objs_col].append(obj_bl)

    return bl_corr_records
