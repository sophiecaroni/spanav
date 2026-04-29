"""
    Title: Cluser-based permutation tests

    Author: Sophie Caroni
    Date of creation: 16.04.2026

    Description:
    Functions for mass-univariate cluster-based permutation tests on EEG spectral data (TFR and PSD).
"""
import warnings
import mne
import numpy as np
import pandas as pd
from mne.stats import f_mway_rm, permutation_cluster_test
from mne.time_frequency import Spectrum, BaseTFR
from spanav_eeg_utils.config_utils import get_seed
from spanav_eeg_utils.spectral_utils import get_band_freqs
from spanav_tbi.processing.psd import get_sid_level_psd_df
from spanav_tbi.processing.tfr import get_sid_level_tfr_df
from spanav_tbi.analysis.stat_utils import get_tfr_epo_types, get_psd_epo_types

SEED = get_seed()
_TFR_EPO_TYPES = get_tfr_epo_types()
_PSD_EPO_TYPES = get_psd_epo_types()


def _pivot_to_array(
        df: pd.DataFrame,
        data_col: str,
        factor_cols: list[str],
) -> tuple[np.ndarray, list, dict[str, list]]:
    """
    Pivot a subject x factors DataFrame into (n_sids, n_factor_levels_combos, *data_shape).

    Data is extracted via obj._data[0], assuming single-channel spectral objects (average_channels=True at load time).

    :param df: pd.DataFrame, must have column 'sid', all columns in factor_cols, and data_col.
    :param data_col: str, column holding the spectral objects.
    :param factor_cols: list[str], columns that define the within-subjects factors, ordered fromnslowest- to
        fastest-varying (e.g. ['cond', 'epo_type']).
    :return: tuple of (array of shape (n_sids, n_factor_levels_combos, *data_shape), included_sids list, factor_levels
        dict)
    """
    factor_levels = {col: sorted(df[col].unique()) for col in factor_cols}

    # Extract all factor levels combinations according f_mway_rm convention: first factor varies slowest, last factor varies fastest.
    expected_combos = set(df.groupby(factor_cols).groups.keys())

    rows = []
    included_sids = []
    for sid, sid_df in df.groupby('sid'):
        sid_groups = sid_df.groupby(factor_cols)

        # Exclude a subject completely if they don't have a spectral object for all the factor levels combinations
        missing = expected_combos - set(sid_groups.groups.keys())
        if missing:
            warnings.warn(f"Skipping {sid = } because of missing data for the following factor-level combinations: {missing}")
            continue

        combo_data = []
        for combo, combo_df in sid_groups:

            # Check that every subject only has one spectral object per combination
            if len(combo_df) > 1:
                raise ValueError(
                    f"Expected one observation for subject, got {len(combo_df)} rows for {sid = }, {combo = }. "
                )
            obj = combo_df.iloc[0][data_col]

            is_mne_obj = isinstance(obj, Spectrum) or isinstance(obj, BaseTFR)
            obj_data = obj if not is_mne_obj else obj._data[0]  # to extract the average data and drop the channel dimension
            combo_data.append(obj_data)
        rows.append(combo_data)
        included_sids.append(sid)  # keep track of included subjects

    return np.array(rows), included_sids, factor_levels


def run_cluster_test(
        data: np.ndarray,
        effect: str,
        factor_levels: list[int],
        n_permutations: int = 1000,
        seed: int = SEED,
) -> dict:
    """
    Run a spectral cluster permutation test on channel-averaged EEG spectral data.

    Handles (n_freqs,) for PSD or (n_freqs, n_times) for TFR inputs. Adjacency is built purely over the
    spectral objects via mne.stats.combine_adjacency (chain adjacency per dimension).

    :param data: np.ndarray of shape (n_sids, n_factor_levels_combos, *spectral_dims).
        spectral_dims is (n_freqs,) for PSD or (n_freqs, n_times) for TFR.
        n_factor_levels_combos is the product of all factor level counts.
    :param effect: str, factor statistical effect to test.
    :param factor_levels: list[int], level counts per within-subjects factor passed to f_mway_rm.
        For a cond × epo_type design pass e.g. [n_conds, n_epo_types].
    :param n_permutations: int, number of permutations for the null distribution.
    :param seed: int, random seed for reproducibility.
    :return: dict with keys F_obs (ndarray of shape spectral_dims), clusters (list), cluster_pv (ndarray),
        H0 (ndarray), significant (ndarray[bool] of shape spectral_dims — True where a significant cluster exists).
    """
    n_sids, n_factor_levels_combos = data.shape[:2]

    # Define (time-)frequency adjacency matrix
    spectral_dims = data.shape[2:]
    adjacency = mne.stats.combine_adjacency(*spectral_dims)

    # Define matrix of (n_sids, n_space) arrays, one per factor level combination
    n_space = int(np.prod(spectral_dims))
    data_flat = data.reshape(n_sids, n_factor_levels_combos, n_space)
    X = [data_flat[:, i_c, :] for i_c in range(n_factor_levels_combos)]

    stat_fun = make_rm_stat_fun(factor_levels, effect)

    F_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
        X,
        stat_fun=stat_fun,
        adjacency=adjacency,
        n_permutations=n_permutations,
        seed=seed,
        out_type='mask',
    )

    sig_mask = np.zeros(F_obs.shape, dtype=bool)
    for cluster, is_sig in zip(clusters, cluster_pv < 0.05):
        if is_sig:
            sig_mask[cluster] = True

    return dict(
        F_obs=F_obs.reshape(spectral_dims),
        clusters=clusters,
        cluster_pv=cluster_pv,
        H0=H0,
        significant=sig_mask.reshape(spectral_dims),
    )


def make_rm_stat_fun(factor_levels: list[int], effect: str):
    """
    Return a stat_fun closure for repeated-measures F-test via f_mway_rm.
    :param factor_levels: list[int], number of levels per factor, passed to f_mway_rm.
    :param effect: str, effect string for f_mway_rm ('A', 'B', 'A:B', 'A*B').
    :return: callable, stat_fun compatible with permutation_cluster_test.
    """
    def stat_fun(*args):
        return f_mway_rm(
            np.swapaxes(args, 1, 0),
            factor_levels=factor_levels,
            effects=effect,
            return_pvals=False,
        )[0]
    return stat_fun


def _get_effect_label(effect: str, factor_cols: list[str]) -> str:
    """ Derive human-readable label for a stastistical effect string from the factor column names."""
    factor_map = {chr(ord('A') + i): col for i, col in enumerate(factor_cols)}
    return ' × '.join(factor_map.get(f, f) for f in effect.split(':'))


def run_cluster_test_tfr(
        group: str,
        factor_cols: list[str],
        effects: list[str],
        dev: bool = False,
        **kwargs,
) -> tuple[dict[str, dict], list]:
    """
    Extract TFR data and run run_cluster_test for each effect of interest.

    Expects the 'tfr' column to contain single-channel AverageTFR objects (average_channels=True at load time).

    :param group:
    :param factor_cols: list[str], within-subjects factors.
    :param effects: list[str], factors statistical effects to test.
    :param kwargs: forwarded to run_cluster_test (e.g. n_permutations).
    :param dev:
    :return: tuple of (results dict mapping effect str to results dict, included_sids list)
    """
    tfr_df = get_sid_level_tfr_df(test=dev, save=False, load=True, average_channels=True)
    tfr_df = tfr_df[tfr_df['group'] == group]
    tfr_df = tfr_df[tfr_df['epo_type'].isin(_TFR_EPO_TYPES)]

    data, included_sids, factor_levels = _pivot_to_array(tfr_df, 'tfr', factor_cols)
    factor_levels_counts = [len(v) for v in factor_levels.values()]
    results = {}
    for effect in effects:
        res = run_cluster_test(data, factor_levels=factor_levels_counts, effect=effect, **kwargs)
        res['effect_label'] = _get_effect_label(effect, factor_cols)
        results[effect] = res
    return results, included_sids


def run_cluster_test_psd(
        group: str,
        factor_cols: list[str],
        effects: list[str],
        dev: bool = False,
        **kwargs,
) -> tuple[dict[str, dict], list]:
    """
    Extract PSD data and run run_cluster_test for each effect of interest.

    Expects the 'psd' column to contain single-channel Spectrum objects (average_channels=True at load time).

    :param group:
    :param factor_cols: list[str] or None, within-subjects factors.
    :param effects: list[str], factors statistical effects to test.
    :param kwargs: forwarded to run_cluster_test (e.g. n_permutations).
    :param dev:
    :return: tuple of (results dict mapping effect str to results dict, included_sids list)
    """
    psd_df = get_sid_level_psd_df(test=dev, save=False, load=True, average_channels=True)
    psd_df = psd_df[psd_df['group'] == group]
    psd_df = psd_df[psd_df['epo_type'].isin(_PSD_EPO_TYPES)]

    data, included_sids, factor_levels = _pivot_to_array(psd_df, 'psd', factor_cols)
    factor_levels_counts = [len(v) for v in factor_levels.values()]
    results = {}
    for effect in effects:
        res = run_cluster_test(data, factor_levels=factor_levels_counts, effect=effect, **kwargs)
        res['effect_label'] = _get_effect_label(effect, factor_cols)
        results[effect] = res
    return results, included_sids


def run_psd_ch_cluster_test(
        group: str,
        factor_cols: list[str],
        effects: list[str],
        band: str = 'theta',
        dev: bool = False,
        **kwargs,
):
    psd_df = get_sid_level_psd_df(test=dev, save=False, load=True, average_channels=False)
    psd_df = psd_df[psd_df['group'] == group]
    psd_df = psd_df[psd_df['epo_type'].isin(_PSD_EPO_TYPES)]

    # Select PSD in the interval of the band of interest
    fmin, fmax = get_band_freqs(band)
    psd_df['psd'] = psd_df['psd'].apply(lambda psd: psd.get_data(fmin=fmin, fmax=fmax))
    return run_ch_cluster_test(psd_df, factor_cols, effects, **kwargs)


def run_ch_cluster_test(
        df: pd.DataFrame,
        factor_cols: list[str],
        effects: list[str],
        data_col: str = 'psd',
        **kwargs,
):
    data, included_sids, factor_levels = _pivot_to_array(df, data_col, factor_cols)
    factor_levels_counts = [len(v) for v in factor_levels.values()]
    results = {}
    for effect in effects:
        res = run_cluster_test(data, factor_levels=factor_levels_counts, effect=effect, **kwargs)
        res['effect_label'] = _get_effect_label(effect, factor_cols)
        results[effect] = res
    return results, included_sids

