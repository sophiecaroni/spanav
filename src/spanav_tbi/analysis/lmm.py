"""
    Title: Linear mixed models (LMM)

    Author: Sophie Caroni
    Date of creation: 10.03.2026

    Description:
    This script contains functions that allow to parametrize and run LMMs.
"""
import subprocess
import pandas as pd
import numpy as np
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.parsing_utils as prs
from pathlib import Path
from spanav_eeg_utils.config_utils import get_seed


def run_rscript(rscript_fname: str, args: list, verbose: bool = True) -> subprocess.CompletedProcess:
    r_script_path = Path(__file__).resolve().parent / rscript_fname
    r_executable = Path("/Library/Frameworks/R.framework/Resources/bin/Rscript")

    args_str = [str(arg).lower() if isinstance(arg, bool) else str(arg) for arg in args]

    try:
        result = subprocess.run(
            args=[str(r_executable), str(r_script_path)] + args_str,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("R script failed")
        print("stdout:")
        print(e.stdout)
        print("stderr:")
        print(e.stderr)
        raise

    if verbose:
        print("\n####################################### R SCRIPT OUTPUT #######################################\n")
        print(result.stdout)
        if result.stderr:
            print("####################################### R SCRIPT WARNINGS ######################################\n")
            print(result.stderr)
    return result


def _simulate_sid(existing_sids: list[str], new_sid_group: str) -> str:
    id_n = 1
    while True:
        new_sid = f'73{new_sid_group.upper()}{id_n:02d}'
        if new_sid not in existing_sids:
            return new_sid
        id_n += 1


def _estimate_lmm_components(in_df: pd.DataFrame, feature: str, lmm_factors: list[str]) -> tuple[pd.Series, float, float]:
    df = in_df.copy()

    # 1) Fixed-effects: mean response per factor combination, pooled across subjects
    factor_comb_means = df.groupby(lmm_factors)[feature].mean()
    factor_comb_means_long = df.groupby(lmm_factors)[feature].transform("mean")  # .transform() to return means in shape of df len (instead of compacting as mean)
    df['factor_combs_resid'] = df[feature] - factor_comb_means_long

    # 2) Random intercept: per-subject mean of the cell-centered residuals (between-subject SD)
    sid_intercept = df.groupby('sid')['factor_combs_resid'].mean()
    sigma_sid = sid_intercept.std(ddof=1)  # ddof=1 computes SD using n-1 denominator (ie sample SD, since we estimate a sample's unknown underlying spread)

    # 3) Residual: leftover spread once both the cell mean and the subject intercept are removed (within-subject residual SD)
    per_sid_intercept = df['sid'].map(sid_intercept)  # trasforms sid_intercept in the df["sid"] shape
    resid = df['factor_combs_resid'] - per_sid_intercept
    sigma_res = resid.std(ddof=1)  # ddof=1 computes SD using n-1 denominator (ie sample SD, since we estimate a sample's unknown underlying spread)

    return factor_comb_means, sigma_sid, sigma_res


def _simulate_lmm_dataframe(in_df: pd.DataFrame, fname: str, new_sids_by_group_n: int = 15) -> None:
    sim_df = in_df.copy()
    rng = np.random.default_rng(get_seed())
    lmm_factors = ["group", "cond", "epo_type", "band"]

    # Pick one subject by group as examples and subset the input df to their data
    example_sids = [group_sids[0] for group_sids in in_df.groupby('group')['sid'].unique()]
    example_sids_df = in_df.copy()[in_df['sid'].isin(example_sids)]

    # Use data from the exemplar subjects as a basis to create two new subjects (one per group) with fake data
    for i in range(new_sids_by_group_n):
        existing_sids = sim_df['sid'].unique()
        new_sids_df = example_sids_df.copy()

        # First replace subject IDs
        for sid in new_sids_df['sid'].unique():
            new_sid = _simulate_sid(existing_sids, new_sid_group=prs.get_group_letter(sid))
            new_sids_df.replace({sid: new_sid}, inplace=True)

        # Then simulate each feature from the model as: factor-combinations mean + per-subject random intercept + residual noise
        simulate_features = ["abs_pw_log", "rel_pw_log"]
        for feature in simulate_features:

            # Use real data (from in_df) to estimate the regression-model components of the feature to simulate
            fixed_eff, sigma_sid, sigma_res = _estimate_lmm_components(in_df, feature, lmm_factors)

            # Fixed effects for each row (based on factor values combination)
            new_sids_df_mi = new_sids_df.set_index(lmm_factors)  # convert factor columns to multiindex (as in fixed_eff)
            row_fixed_eff = new_sids_df_mi.index.map(fixed_eff).to_numpy()  # extract the mean estimated by fixed_eff

            # One random-intercept per subject
            sid_intercepts = {sid: rng.normal(0, sigma_sid) for sid in new_sids_df['sid'].unique()}  # N(0, σ_sid)
            row_intercept = new_sids_df['sid'].map(sid_intercepts).to_numpy()  # broadcast dict to an array, one value per row

            # Create a vector of residual gaussian noise (normal distribution with mean=0 and SD=sigma_res) for each row
            noise = rng.normal(0, sigma_res, size=len(new_sids_df))  # ε ~ N(0, σ_resid)

            # Use estimated values to simulate feature values for new subjects
            new_sids_df[feature] = row_fixed_eff + row_intercept + noise

        sim_df = pd.concat([sim_df, new_sids_df], ignore_index=True)

    sim_df.reset_index(inplace=True)
    sim_df.to_csv(io.get_tables_path() / fname, index=False)


def _subset_lmm_dataframe(in_df: pd.DataFrame, fname: str) -> None:
    subset_df = in_df.copy()
    subset_df.drop(subset_df[subset_df['epo_n'] > 3].index, inplace=True)
    subset_df.reset_index(inplace=True)
    subset_df.to_csv(io.get_tables_path() / fname, index=False)


def get_lmm_table_path(test: bool, sim: bool, overwrite: bool = False) -> Path:
    df_fname = "osc_df_epo_level.csv"
    if sim:
        df_fname_sim = "osc_df_epo_level_SIM.csv"
        if overwrite or not (io.get_tables_path() / df_fname_sim).exists():
            df = pd.read_csv(io.get_tables_path() / df_fname, index_col=0)
            _simulate_lmm_dataframe(df, df_fname_sim)
        df_fname = df_fname_sim
    elif test:
        df_fname_test = "osc_df_epo_level_TEST.csv"
        if overwrite or not (io.get_tables_path() / df_fname_test).exists():
            df = pd.read_csv(io.get_tables_path() / df_fname, index_col=0)
            _subset_lmm_dataframe(df, df_fname_test)
        df_fname = df_fname_test

    return io.get_tables_path() / df_fname


def _parse_rscript_output(output: str, pattern: str) -> str:
    for line in output.splitlines():
        if pattern in line:
            return line.split(pattern, 1)[1].strip()
    raise RuntimeError(f"Could not find value with prefix '{pattern}' from R output.")


def select_best_fit_method(
        metric: str,
        lmm_formula: str,
        test: bool,
        sim: bool,
        band: str | None = None
) -> str:
    df_path = get_lmm_table_path(test=test, sim=sim)
    script_out = run_rscript(
        "eval_lmm_fits.R",
        [test, sim, band, metric, df_path, lmm_formula],
        verbose=True,
    )
    return _parse_rscript_output(script_out.stdout, "PROPOSED_BEST_FIT_METHOD=")

