"""
    Title: Linear mixed models (LMM) pipeline

    Author: Sophie Caroni
    Date of creation: 10.03.2026

    Description:
    This script performs a LMM pipeline where suited models are chosen and used to model different EEG features.
"""
import spanav_tbi.analysis.lmm as lmm


def run_lmm_explorative(test: bool, sim: bool) -> None:
    formula = "y ~ group * cond * epo_type * band + (1 | sid)"
    metrics = (
        "abs_pw_log",
        "rel_pw_lin",
    )

    for metric in metrics:
        print(f"\n####################################### EXPLORATIVE: {metric.upper()} #######################################\n")
        fit_method = lmm.select_best_fit_method(metric, formula, test, sim, "explorative")
        print(
            f"Best fitting method for {metric}: {fit_method}"
        )


def run_lmm_hypo_test(test: bool, sim: bool) -> None:
    formula = "y ~ group * epo_type + (1 | sid)"
    metrics = (
        "abs_pw_log",
        "rel_pw_lin",
    )

    for metric in metrics:
        print(f"\n################################## HYPOTHESES TESTING: THETA {metric.upper()} ##################################\n")
        fit_method = lmm.select_best_fit_method(metric, formula, test, sim, band="theta", pipeline_label="hypo-testing")
        print(
            f"Best fitting method for {metric}: {fit_method}"
        )


def run_lmms(**kwargs):
    run_lmm_hypo_test(**kwargs)
    run_lmm_explorative(**kwargs)


if __name__ == "__main__":
    test = False
    sim = not test
    run_lmms(
        test=test,
        sim=sim  ,
    )
