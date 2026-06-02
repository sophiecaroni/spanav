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

wd <- "/Users/sophiecaroni/epfl_hes/spanav/local"
setwd(wd)

# -------------------------
# 0. Preparation
# -------------------------

# Retrieve arguments
args <- commandArgs(trailingOnly=TRUE)
testing_mode <- tolower(args[1]) == "true"
simulation_mode <- tolower(args[2]) == "true"
band_arg <- args[3]
metric <- args[4]
df_fpath <- args[5]
formula_arg <- args[6]
pipeline_arg <- args[7]

# The calling script passes "None" when band is a factor and not used to filter the metric in a specific band
if (band_arg == "None") band_arg <- NA

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
pdir <- file.path("outputs", "plots", "LMM", pipeline_arg, "diagnostics")
dir.create(pdir, recursive=TRUE, showWarnings=FALSE)
band_label <- if (is.na(band_arg)) "" else paste0(band_arg, "_")  # omit band from fname when metric is not filtered in a band
pname_pref <- paste0(if (testing_mode) "TEST_" else (if (simulation_mode) "SIM_" else ""), band_label, metric)

# -------------------------
# 1. Load and prepare dataframe
# -------------------------
raw_df <- read.csv(file=df_fpath, row.names=1, colClasses=c(group="character"))

df <- raw_df %>%
    filter(is.na(band_arg) | band == band_arg) %>%  # if band is input when calling the script exclude rows not corresponding to it
    select(sid, group, cond, epo_type, band, all_of(metric)) %>%  # select columns of interest
    rename(y=all_of(metric)) %>%  # rename into 'y' the column containing the metric of interest
    mutate(  # turn columns into factors
        sid=factor(sid),
        group=factor(group),
        cond=factor(cond),
        epo_type=factor(epo_type),
        band=factor(band)
    )

cat("\nInput dataframe: ")
cat("\n\tsubjects: ", as.character(unique(df$sid)))
cat("\n\trows: ", nrow(df))
cat("\n\thead: ", "\n")
print(head(df))

# -------------------------
# 2. Run candidate models
# -------------------------
# Define common formula for the models
formula <- as.formula(formula_arg)

# Verify wether the y column contains only positive values (needed for the lognormal family)
all_positive <- all(df$y > 0, na.rm=TRUE)

# Run GLMM models with different fitting methods and add them to list
models <- list()

cat("\n======================================== Fitting methods ======================================\n")

# Gaussian fitting family
cat("\n----------------------------------------- Gaussian fit ----------------------------------------\n")
models$gaussian <- glmmTMB(
    formula,
    data=df,
    family=gaussian()
)

# Lognormal fitting family
if (all_positive) {
    cat("\n---------------------------------------- Lognormal fit ---------------------------------------\n")
    models$lognormal <- glmmTMB(
    formula,
    data=df,
    family=lognormal()
    )
}

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

        # Plot DHARMa residulas
        ppath <- file.path(pdir, paste0(pname_pref, "_", fit_name))
        save_residual_plot(res, paste0(ppath, "_dharma.png"))

        # Run uniformity test (already plotted in DHARMa residulas "_dharma.png") to add p-value into variable and table later
        unif_res <- testUniformity(res, plot=TRUE)
        unif_p <- unif_res$p.value

        # Plot dispersion and store dispersion test p-value into variable
        save_dharma_plot(
            function() disp_res <<- testDispersion(res, plot=TRUE),
            paste0(ppath, "_dispersion.png")
        )
        disp_p <- disp_res$p.value

        # Plot outliers and store outliers test p-value into variable
        save_dharma_plot(
            function() out_res <<- testOutliers(res, plot=TRUE),
            paste0(ppath, "_outliers.png")
        )
        out_p <- out_res$p.value
    }

    # Save diagn_table of this model in df
    round_dig <- 2
    this_row <- data.frame(
        fit_method = fit_name,
        disp_p = round(disp_p, digits=round_dig),
        unif_p = round(unif_p, digits=round_dig),
        out_p = round(out_p, digits=round_dig),
        AIC = round(AIC(model_obj), digits=round_dig),
        n_fail_diagn = sum(c(disp_p, unif_p, out_p) < 0.05, na.rm = TRUE),
        stringsAsFactors = FALSE
    )
    diagn_table <- rbind(diagn_table, this_row)
}

# -------------------------
# 4. Print diagn_table and proposed best fit method
# -------------------------
cat("\n============================================ Result ============================================\n")
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
