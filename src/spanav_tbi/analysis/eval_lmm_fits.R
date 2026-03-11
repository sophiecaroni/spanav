# """
#     Title: Evaluate linear mixed models (LMM) fitting methods
#
#     Author: Sophie Caroni
#     Date of creation: 10.03.2026
#
#     Description:
#     This script compares different fitting methods for LMM on some data using DHARMa diagnostics and suggests the best one.
# """
library(glmmTMB)
library(DHARMa)
library(dplyr)

wd <- "/Users/sophiecaroni/epfl_hes/spanav-tbi/data"
setwd(wd)

# -------------------------
# 0. Preparation
# -------------------------

# Retrieve arguments
args <- commandArgs(trailingOnly=TRUE)

testing_mode <- tolower(args[1]) == "true"
band_arg <- args[2]
metric <- args[3]
df_fpath <- args[4]

# Define plotting functions
save_residual_plot <- function(res, fpath) {
    png(fpath, width=1200, height=900, res=140)
    plot(res)
    dev.off()
}
save_dharma_plot <- function(plot_fun, fpath) {
    png(fpath, width=1200, height=900, res=140)
    on.exit(dev.off())
    plot_fun()
}

# Define plot paths/fnames
pdir <- file.path("outputs", "plots", "LMM", "diagnostics")
dir.create(pdir, recursive=TRUE, showWarnings=FALSE)
pname_pref <- paste0(band_arg, "_", metric, if (testing_mode) "_test" else "")

# -------------------------
# 1. Load and prepare dataframe
# -------------------------
raw_df <- read.csv(file=df_fpath, row.names=1, colClasses=c(group="character"))

df <- raw_df %>%
    filter(band == band_arg) %>%   # filter out rows not corresponding to the band of interest
    select(sid, group, cond, epo_type, all_of(metric)) %>%  # select columns of interest
    rename(y=all_of(metric)) %>%  # rename into 'y' the column containing the metric of interest
    mutate(  # turn columns into factors
        sid=factor(sid),
        group=factor(group),
        cond=factor(cond),
        epo_type=factor(epo_type)
    )

cat("\nRows in analysis dataframe:", nrow(df), "\n")
print(head(df))

# -------------------------
# 2. Run candidate models
# -------------------------
# Define common formula for the models
form_lmm <- if (testing_mode) {
    y ~ cond * epo_type + (1 | sid)
} else {
    y ~ group * cond * epo_type + (1 | sid)
}

# Verify wether the y column contains only positive values (needed for the lognormal family)
all_positive <- all(df$y > 0, na.rm=TRUE)

# Run GLMM models with different fitting methods and add them to list
models <- list()

cat("\n======================================== Fitting models ======================================\n")

# Gaussian fitting family
cat("\n---------------------------------------- Gaussian model ---------------------------------------\n")
models$gaussian <- glmmTMB(
    form_lmm,
    data=df,
    family=gaussian()
)

# Lognormal fitting family
if (all_positive) {
    cat("\n-------------------------------------- Lognormal model --------------------------------------\n")
    models$lognormal <- glmmTMB(
    form_lmm,
    data=df,
    family=lognormal()
    )
}

# # Tweedie fitting family - often much slower for DHARMa simulation, so run DHARMa only if the fit succeeds
# cat("\n----------------------------------------- Tweedie model -----------------------------------------\n")
# models$tweedie <- tryCatch(
#   glmmTMB(
#     form_lmm,
#     data=df,
#     family=tweedie(link="log")
#   ),
#   error=function(e) {
#     message("Tweedie fitting failed: ", e$message)
#     NULL
#   }
# )

# -------------------------
# 3. Run diagnostics to evaluate models
# -------------------------
diagn_table <- data.frame(
    fit_method=character(),
    disp_p=numeric(),
    unif_p=numeric(),
    out_p=numeric(),
    AIC=numeric(),
    n_fail_diagn=numeric(),
    stringsAsFactors=FALSE
)
cat("\n================================== Running DHARMa diagnostics ==================================\n")
for (fit_name in names(models)) {

    # Get valid models (that converged when fitted)
    model_obj <- models[[fit_name]]
    if (is.null(model_obj)) next
    cat("\n--- On", fit_name, "model ---\n")

    # Attempt computeing residuals
    res <- tryCatch(
    {
        n_sim <- if (testing_mode) 50 else 250  # lower number of stimulations when running in testing mode
        simulateResiduals(model_obj, n=n_sim, plot=FALSE)
    },
    error = function(e) {
        message("DHARMa failed for ", fit_name, ": ", e$message)
        NULL
    }
    )

    # For successfully computed residuals
    if (!is.null(res)) {

        # Plot residulas
        ppath <- file.path(pdir, paste0(pname_pref, "_", fit_name))
        save_residual_plot(res, paste0(ppath, "_DHARMa.png"))

        # Plot dispersion and store dispersion test p-value into variable
        cat("\n------------------------------------------ Dispersion -----------------------------------------\n")
        save_dharma_plot(
            function() disp_res <<- testDispersion(res, plot=TRUE),
            paste0(ppath, "_dispersion.png")
        )
        disp_p <- disp_res$p.value

        # Plot uniformity and store uniformity test p-value into variable
        cat("\n------------------------------------------ Uniformity -----------------------------------------\n")
        save_dharma_plot(
            function() unif_res <<- testUniformity(res, plot=TRUE),
            paste0(ppath, "_uniformity.png")
        )
        unif_p <- unif_res$p.value

        # Plot outliers and store outliers test p-value into variable
        cat("\n------------------------------------------- Outliers ------------------------------------------\n")
        save_dharma_plot(
            function() out_res <<- testOutliers(res, plot=TRUE),
            paste0(ppath, "_outliers.png")
        )
        out_p <- out_res$p.value
    }

    # Save diagn_table of this model in df
    this_row <- data.frame(
        fit_method = fit_name,
        disp_p = disp_p,
        unif_p = unif_p,
        out_p = out_p,
        AIC = AIC(model_obj),
        n_fail_diagn = sum(c(disp_p, unif_p, out_p) < 0.05, na.rm = TRUE),
        stringsAsFactors = FALSE
    )
    diagn_table <- rbind(diagn_table, this_row)
}

# -------------------------
# 4. Print diagn_table and proposed best fit method
# -------------------------
cat("\n============================================= Result ===========================================\n")
cat(
    "How to evaluate this table:
    \t a) Higher p-values (non-significant tests) indicate better fits!
    \t b) In case multiple fit methods achieve good diagnostics: choose the one with lowest AIC
    ! The table already ordered based on number of failed diagnostics and AIC !\n\n"
)
diagn_table <- diagn_table[order(diagn_table$n_fail_diagn, diagn_table$AIC), ]  # order table based on failed diagnostics and AIC
rownames(diagn_table) <- NULL  # reset index
print(diagn_table)


proposed_method <- diagn_table$fit_method[1]
cat("\n==> PROPOSED_BEST_FIT_METHOD=", proposed_method, "\n", sep="")
cat("\n============================================== END =============================================\n")
