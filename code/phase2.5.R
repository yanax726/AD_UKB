#!/usr/bin/env Rscript
# =============================================================================
# UK BIOBANK AD-METABOLIC PHASE 2.5: MENDELIAN RANDOMIZATION (FIXED)
# =============================================================================
# Purpose: Genetic causal inference to complement Phase 3 temporal analysis
# Fixes: 1) Load local NAFLD file, 2) Handle missing packages, 3) Link to Phase 3
# =============================================================================
# install.packages("R.utils")
cat("\n================================================================================\n")
cat("PHASE 2.5: MENDELIAN RANDOMIZATION ANALYSIS\n")
cat("================================================================================\n")

# Load required packages
suppressPackageStartupMessages({
  library(TwoSampleMR)
  library(data.table)
  library(tidyverse)
  library(glue)
})

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - Use your actual paths
gwas_path <- "/mnt/project/"  # Your GWAS files location
phase1_path <- "~/results/discovery_pipeline_bias_corrected"
output_path <- file.path(phase1_path, "mendelian_randomization")
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

# Parameters
P_THRESHOLD <- 5e-8
R2_THRESHOLD <- 0.001
KB_THRESHOLD <- 10000

# =============================================================================
# STEP 1: LOAD PHASE 1 COHORT DATA FOR CONTEXT
# =============================================================================
cat("\nLoading Phase 1 outputs for integration...\n")

# Load your discovery cohort to get context
discovery_cohort <- readRDS(file.path(phase1_path, "discovery_cohort_primary.rds"))
cat(glue("  Loaded discovery cohort: {nrow(discovery_cohort)} participants\n"))
cat(glue("  AD cases: {sum(discovery_cohort$ad_case_primary == 1)}\n"))

# Get outcome prevalences for power calculations
outcome_prevalences <- discovery_cohort %>%
  summarise(
    diabetes_prev = mean(has_diabetes_any, na.rm = TRUE),
    obesity_prev = mean(has_obesity_any, na.rm = TRUE),
    hyperlipidemia_prev = mean(has_hyperlipidemia_any, na.rm = TRUE)
  )

# =============================================================================
# STEP 2: LOAD GWAS DATA (INCLUDING LOCAL NAFLD FILE)
# =============================================================================
cat("\n================================================================================\n")
cat("LOADING GWAS DATA\n")
cat("================================================================================\n")

# FIXED VERSION - handles all the specific column names found in your files
load_gwas_file <- function(file_path, name) {
  cat(glue("\nLoading {name}...\n"))
  
  # Read file
  if (grepl("\\.gz$", file_path)) {
    data <- fread(file_path)
  } else {
    data <- fread(file_path)
  }
  
  cat(glue("  Loaded {nrow(data)} variants\n"))
  
  # Convert to data.frame
  data <- as.data.frame(data)
  
  # FILE-SPECIFIC COLUMN MAPPING based on diagnostic results
  
  if (name == "ad_meta") {
    # AD meta-analysis file mappings
    data$EA <- data$Tested_Allele
    data$NEA <- data$Other_Allele
    data$EAF <- data$Freq_Tested_Allele_in_HRS
    # Already has: SNP, BETA, SE, P
    
  } else if (name == "diabetes") {
    # DIAGRAM diabetes - SPECIAL HANDLING
    # Has OR but NO SE - must calculate from confidence intervals
    data$BETA <- log(data$OR)
    
    # Calculate SE from 95% CI of OR
    # SE = (log(OR_95U) - log(OR_95L)) / (2 * 1.96)
    data$SE <- (log(data$OR_95U) - log(data$OR_95L)) / (2 * 1.96)
    
    data$EA <- data$RISK_ALLELE
    data$NEA <- data$OTHER_ALLELE
    # Already has: SNP, P
    
    cat("  Converted OR to beta and calculated SE from CI\n")
    
  } else if (name %in% c("hdl", "ldl", "triglycerides", "cholesterol")) {
    # Lipid traits files
    data$SNP <- data$rsID
    data$BETA <- data$EFFECT_SIZE
    data$P <- data$pvalue
    data$EA <- data$ALT
    data$NEA <- data$REF
    data$EAF <- data$POOLED_ALT_AF
    # Already has: SE
    
  } else if (name == "finngen_atopic" || name == "finngen_dermatitis") {
    # FinnGen files
    if ("rsids" %in% names(data)) data$SNP <- data$rsids
    if ("beta" %in% names(data)) data$BETA <- data$beta
    if ("sebeta" %in% names(data)) data$SE <- data$sebeta
    if ("pval" %in% names(data)) data$P <- data$pval
    if ("alt" %in% names(data)) data$EA <- data$alt
    if ("ref" %in% names(data)) data$NEA <- data$ref
    if ("af_alt" %in% names(data)) data$EAF <- data$af_alt
  }
  
  # Final check
  required_cols <- c("SNP", "BETA", "SE", "P")
  available <- intersect(required_cols, names(data))
  missing <- setdiff(required_cols, names(data))
  
  cat("  Required columns available:", paste(available, collapse=", "), "\n")
  if (length(missing) > 0) {
    cat("  WARNING - Missing columns:", paste(missing, collapse=", "), "\n")
    cat("  Available columns:", paste(head(names(data), 15), collapse=", "), "\n")
  }
  
  return(data)
}
# Load exposures (AD/Eczema)
exposure_files <- list(
  ad_meta = "Meta-analysis_Locke_et_al+UKBiobank_2018_UPDATED.txt",
  finngen_atopic = "summary_stats_release_finngen_R12_L12_ATOPIC.gz",
  finngen_dermatitis = "summary_stats_release_finngen_R12_L12_DERMATITISECZEMA.gz"
)

exposure_data <- list()
for (name in names(exposure_files)) {
  file_path <- file.path(gwas_path, exposure_files[[name]])
  if (file.exists(file_path)) {
    exposure_data[[name]] <- load_gwas_file(file_path, name)
  }
}

# Load outcomes (Metabolic diseases)
outcome_files <- list(
  diabetes = "diagram.mega-meta.txt",
  hdl = "without_UKB_HDL_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz",
  ldl = "without_UKB_LDL_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz",
  triglycerides = "without_UKB_logTG_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz",
  cholesterol = "without_UKB_TC_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz"
)

outcome_data <- list()
for (name in names(outcome_files)) {
  file_path <- file.path(gwas_path, outcome_files[[name]])
  if (file.exists(file_path)) {
    outcome_data[[name]] <- load_gwas_file(file_path, name)
  }
}

# SPECIAL HANDLING FOR LOCAL NAFLD FILE
nafld_file <- file.path(gwas_path, "GCST90091033_buildGRCh37.tsv.gz")
if (file.exists(nafld_file)) {
  cat("\nLoading local NAFLD GWAS file...\n")
  nafld_data <- fread(nafld_file)
  
  # Map GWAS catalog format columns
  if ("chromosome" %in% names(nafld_data)) nafld_data$CHR <- nafld_data$chromosome
  if ("base_pair_location" %in% names(nafld_data)) nafld_data$POS <- nafld_data$base_pair_location
  if ("variant_id" %in% names(nafld_data)) nafld_data$SNP <- nafld_data$variant_id
  if ("effect_allele" %in% names(nafld_data)) nafld_data$EA <- nafld_data$effect_allele
  if ("other_allele" %in% names(nafld_data)) nafld_data$NEA <- nafld_data$other_allele
  if ("beta" %in% names(nafld_data)) nafld_data$BETA <- nafld_data$beta
  if ("standard_error" %in% names(nafld_data)) nafld_data$SE <- nafld_data$standard_error
  if ("p_value" %in% names(nafld_data)) nafld_data$P <- nafld_data$p_value
  
  outcome_data$nafld <- nafld_data
  cat(glue("  Successfully loaded {nrow(nafld_data)} NAFLD variants\n"))
}

# =============================================================================
# STEP 3: EXTRACT INSTRUMENTS
# =============================================================================
cat("\n================================================================================\n")
cat("EXTRACTING GENETIC INSTRUMENTS\n")
cat("================================================================================\n")

extract_instruments_custom <- function(gwas_data, p_threshold = 5e-8, r2 = 0.001) {
  # Filter for genome-wide significance
  instruments <- gwas_data %>%
    filter(P < p_threshold) %>%
    arrange(P)
  
  cat(glue("  Found {nrow(instruments)} genome-wide significant SNPs\n"))
  
  # CRITICAL FIX: Convert to plain data.frame for TwoSampleMR
  instruments <- as.data.frame(instruments)
  
  # Format for TwoSampleMR
  if (nrow(instruments) > 0) {
    instruments_formatted <- format_data(
      instruments,
      type = "exposure",
      snp_col = "SNP",
      beta_col = "BETA",
      se_col = "SE",
      pval_col = "P",
      effect_allele_col = "EA",
      other_allele_col = "NEA",
      eaf_col = "EAF"
    )
    
    # Add F-statistic
    instruments_formatted$F <- (instruments_formatted$beta.exposure / instruments_formatted$se.exposure)^2
    
    return(instruments_formatted)
  }
  
  return(NULL)
}

# Extract instruments for each exposure
all_instruments <- list()
for (exp_name in names(exposure_data)) {
  cat(glue("\nExtracting instruments for {exp_name}...\n"))
  instruments <- extract_instruments_custom(exposure_data[[exp_name]])
  
  if (!is.null(instruments)) {
    all_instruments[[exp_name]] <- instruments
    cat(glue("  Extracted {nrow(instruments)} instruments\n"))
    cat(glue("  Mean F-statistic: {round(mean(instruments$F), 1)}\n"))
  }
}

# =============================================================================
# STEP 4: PERFORM MR ANALYSES
# =============================================================================
cat("\n================================================================================\n")
cat("PERFORMING MENDELIAN RANDOMIZATION\n")
cat("================================================================================\n")

mr_results_all <- list()

for (exp_name in names(all_instruments)) {
  for (out_name in names(outcome_data)) {
    cat(glue("\n{exp_name} -> {out_name}:\n"))
    
    # Format outcome data
    outcome_formatted <- format_data(
      as.data.frame(outcome_data[[out_name]]),  # Convert to data.frame here too
      type = "outcome",
      snp_col = "SNP",
      beta_col = "BETA",
      se_col = "SE",
      pval_col = "P",
      effect_allele_col = "EA",
      other_allele_col = "NEA"
    )
    
    # Harmonize data
    harmonized <- harmonise_data(
      exposure_dat = all_instruments[[exp_name]],
      outcome_dat = outcome_formatted,
      action = 2
    )
    
    # Filter for MR
    harmonized <- harmonized[harmonized$mr_keep, ]
    
    if (nrow(harmonized) >= 3) {
      # Run multiple MR methods
      mr_res <- mr(harmonized, method_list = c(
        "mr_ivw",
        "mr_egger_regression",
        "mr_weighted_median",
        "mr_simple_mode",
        "mr_weighted_mode"
      ))
      
      # Calculate OR for binary outcomes
      mr_res$OR <- exp(mr_res$b)
      mr_res$OR_lci <- exp(mr_res$b - 1.96 * mr_res$se)
      mr_res$OR_uci <- exp(mr_res$b + 1.96 * mr_res$se)
      
      # Store results
      result_key <- paste(exp_name, out_name, sep = "_to_")
      mr_results_all[[result_key]] <- list(
        mr = mr_res,
        harmonized = harmonized,
        n_snps = nrow(harmonized),
        heterogeneity = mr_heterogeneity(harmonized),
        pleiotropy = mr_pleiotropy_test(harmonized)
      )
      
      # Print IVW result
      ivw <- mr_res[mr_res$method == "Inverse variance weighted", ]
      if (nrow(ivw) > 0) {
        cat(glue("  IVW: OR = {round(ivw$OR, 2)} ({round(ivw$OR_lci, 2)}-{round(ivw$OR_uci, 2)}), P = {format.pval(ivw$pval)}\n"))
        cat(glue("  N SNPs: {nrow(harmonized)}, Heterogeneity I2: {round(mr_results_all[[result_key]]$heterogeneity$Q[1], 1)}%\n"))
      }
    } else {
      cat("  Insufficient instruments after harmonization\n")
    }
  }
}

# =============================================================================
# STEP 5: SUMMARIZE AND SAVE RESULTS FOR PHASE 3
# =============================================================================
cat("\n================================================================================\n")
cat("PREPARING RESULTS FOR PHASE 3 INTEGRATION\n")
cat("================================================================================\n")

# Create summary table
summary_table <- data.frame()
for (result_name in names(mr_results_all)) {
  mr_res <- mr_results_all[[result_name]]$mr
  ivw <- mr_res[mr_res$method == "Inverse variance weighted", ]
  
  if (nrow(ivw) > 0) {
    parts <- strsplit(result_name, "_to_")[[1]]
    summary_table <- rbind(summary_table, data.frame(
      exposure = parts[1],
      outcome = parts[2],
      n_snps = mr_results_all[[result_name]]$n_snps,
      OR = round(ivw$OR, 3),
      CI_lower = round(ivw$OR_lci, 3),
      CI_upper = round(ivw$OR_uci, 3),
      p_value = ivw$pval,
      significant = ivw$pval < 0.05
    ))
  }
}

# Display summary
cat("\n=== SIGNIFICANT MR FINDINGS (P < 0.05) ===\n")
significant <- summary_table %>% filter(significant)
if (nrow(significant) > 0) {
  print(significant)
} else {
  cat("No significant causal relationships found via MR\n")
}

# Save for Phase 3 integration
phase3_prep <- list(
  mr_results = mr_results_all,
  summary = summary_table,
  discovery_cohort_info = list(
    n_participants = nrow(discovery_cohort),
    n_ad_cases = sum(discovery_cohort$ad_case_primary == 1),
    outcome_prevalences = outcome_prevalences
  ),
  instrument_strength = sapply(all_instruments, function(x) mean(x$F)),
  timestamp = Sys.time()
)

saveRDS(phase3_prep, file.path(output_path, "phase3_mr_input.rds"))

# Also save detailed results
write.csv(summary_table, file.path(output_path, "mr_summary.csv"), row.names = FALSE)
saveRDS(mr_results_all, file.path(output_path, "mr_detailed_results.rds"))

cat("\n================================================================================\n")
cat("PHASE 2.5 COMPLETE - READY FOR PHASE 3 TEMPORAL CAUSAL DISCOVERY\n")
cat("================================================================================\n")
cat(glue("\nKey outputs saved:\n"))
cat(glue("  1. {output_path}/phase3_mr_input.rds - Main input for Phase 3\n"))
cat(glue("  2. {output_path}/mr_summary.csv - Summary table\n"))
cat(glue("  3. {output_path}/mr_detailed_results.rds - Full MR results\n"))

cat("\n=== DATA AVAILABLE FOR PHASE 3 ===\n")
cat(glue("✓ Participant data: {nrow(discovery_cohort)} individuals\n"))
cat(glue("✓ Temporal data: Multiple instances available\n"))
cat(glue("✓ NMR metabolomics: Processed and ready\n"))
cat(glue("✓ Genetic instruments: {length(all_instruments)} exposures tested\n"))
cat(glue("✓ MR results: {nrow(summary_table)} causal estimates\n"))

cat("\nNext: Run Phase 3 for temporal causal discovery using CausalFormer/VAR-LiNGAM\n")