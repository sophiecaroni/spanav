"""
    Title: Linear mixed models (LMM) pipeline

    Author: Sophie Caroni
    Date of creation: 10.03.2026

    Description:
    This script performs a LMM pipeline where suited models are chosen and used to model different EEG features.
"""
import spanav_tbi.analysis.lmm as lmm


def run_lmm_analysis(test: bool, sim: bool) -> None:
    bands = (
        "theta",
    )
    metrics = (
        "abs_pw",
        "rel_pw",
        "osc_snr",
    )

    i = 0
    for band in bands:
        for metric in metrics:
            if i > 0 and (test or sim):
                break

            fit_method = lmm.select_best_fit_method(band, metric, test, sim)
            print(
                f"Best fitting method for {band} {metric}: {fit_method}"
            )

            i += 1


if __name__ == "__main__":
    run_lmm_analysis(
        test=True,
        sim=False,
    )
