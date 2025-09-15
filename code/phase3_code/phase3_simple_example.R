# phase3_simple_example.R - Simple example of running phase 3 analysis step by step

library(reticulate)

# =============================================================================
# OPTION 1: RUN EVERYTHING AT ONCE (NOT RECOMMENDED FOR LARGE DATA)
# =============================================================================

run_all_at_once <- function() {
  # Load all modules
  source_python("phase3_00_config.py")
  source_python("phase3_01_data_processor.py")
  
  # Run complete pipeline
  # source_python("phase3_complete_pipeline.py")  # This would be the full pipeline
}

# =============================================================================
# OPTION 2: RUN STEP BY STEP (RECOMMENDED)
# =============================================================================

# Step 1: Just load configuration
cat("Step 1: Loading configuration only...\n")
source_python("phase3_00_config.py")

# Check it worked
config <- py$Config
cat("Output will be saved to:", config$OUTPUT_PATH, "\n\n")

# Step 2: Just process data
cat("Step 2: Processing data only...\n")
source_python("phase3_01_data_processor.py")

# Create processor
processor <- py$ComprehensiveDataProcessor()

# If you already have processed data, load it instead
if (file.exists(file.path(config$OUTPUT_PATH, "results", "processed_data.pkl"))) {
  cat("Loading previously processed data...\n")
  analysis_data <- processor$load_processed_data()
} else {
  cat("Processing data from scratch...\n")
  # Your file paths
  file_paths <- list(
    primary = "path/to/your/data.csv"  # Update this
  )
  
  datasets <- processor$load_and_validate_data(py$dict(file_paths))
  analysis_data <- processor$prepare_analysis_dataset(datasets)
  processor$save_processed_data(analysis_data)
}

# Step 3: Run just MR analysis
cat("\nStep 3: Running just MR analysis...\n")
source_python("phase3_03_mr_analysis.py")

# Create some example data
n_snps <- 50
mr_data <- list(
  beta_exposure = rnorm(n_snps, 0, 0.1),
  se_exposure = abs(rnorm(n_snps, 0, 0.02)) + 0.01,
  beta_outcome = rnorm(n_snps, 0.05, 0.1),
  se_outcome = abs(rnorm(n_snps, 0, 0.02)) + 0.01
)

# Run MR
mr_model <- py$ContaminationMixtureMR(n_components = 3L)
mr_model$fit(
  beta_exposure = mr_data$beta_exposure,
  beta_outcome = mr_data$beta_outcome,
  se_exposure = mr_data$se_exposure,
  se_outcome = mr_data$se_outcome
)

cat("MR Effect estimate:", mr_model$causal_effect, "\n")

# =============================================================================
# OPTION 3: RUN SPECIFIC ANALYSES WITH CHECKPOINTS
# =============================================================================

run_with_checkpoints <- function(restart_from = NULL) {
  
  steps <- c("config", "data", "mr", "mediation", "heterogeneity")
  
  # Determine starting point
  if (!is.null(restart_from)) {
    start_idx <- which(steps == restart_from)
    steps <- steps[start_idx:length(steps)]
  }
  
  # Run each step
  for (step in steps) {
    
    cat("\n========== Running:", step, "==========\n")
    
    if (step == "config") {
      source_python("phase3_00_config.py")
      saveRDS(TRUE, file.path(py$Config$OUTPUT_PATH, "checkpoints", "config_done.rds"))
      
    } else if (step == "data") {
      source_python("phase3_01_data_processor.py")
      # Process data...
      saveRDS(TRUE, file.path(py$Config$OUTPUT_PATH, "checkpoints", "data_done.rds"))
      
    } else if (step == "mr") {
      source_python("phase3_03_mr_analysis.py")
      # Run MR...
      saveRDS(TRUE, file.path(py$Config$OUTPUT_PATH, "checkpoints", "mr_done.rds"))
      
    } else if (step == "mediation") {
      source_python("phase3_04_mediation.py")
      # Run mediation...
      saveRDS(TRUE, file.path(py$Config$OUTPUT_PATH, "checkpoints", "mediation_done.rds"))
      
    } else if (step == "heterogeneity") {
      source_python("phase3_05_heterogeneity.py")
      # Run heterogeneity...
      saveRDS(TRUE, file.path(py$Config$OUTPUT_PATH, "checkpoints", "heterogeneity_done.rds"))
    }
    
    # Save intermediate results
    cat("Checkpoint saved for:", step, "\n")
    
    # Optional: Clear memory between steps
    gc()
  }
}

# =============================================================================
# OPTION 4: RUN INDIVIDUAL FUNCTIONS
# =============================================================================

# Just validate data without full processing
validate_data_only <- function(file_path) {
  source_python("phase3_00_config.py")
  source_python("phase3_01_data_processor.py")
  
  processor <- py$ComprehensiveDataProcessor()
  
  # Read just one file
  if (endsWith(file_path, ".csv")) {
    df <- read.csv(file_path)
  } else if (endsWith(file_path, ".rds")) {
    df <- readRDS(file_path)
  }
  
  # Convert to pandas
  df_py <- r_to_py(df)
  
  # Validate
  processor$`_validate_cohort`(df_py, "test_cohort")
  
  # Get validation metrics
  return(processor$validation_metrics)
}

# Example: Just create outcome variables
create_outcomes_only <- function(df) {
  source_python("phase3_01_data_processor.py")
  processor <- py$ComprehensiveDataProcessor()
  
  df_py <- r_to_py(df)
  outcomes <- processor$`_create_outcome_variables`(df_py)
  
  return(py_to_r(outcomes))
}

# =============================================================================
# MEMORY MANAGEMENT TIPS
# =============================================================================

# Function to run analysis in chunks
run_in_chunks <- function(data, chunk_size = 1000) {
  n_total <- nrow(data)
  n_chunks <- ceiling(n_total / chunk_size)
  
  results <- list()
  
  for (i in 1:n_chunks) {
    cat("Processing chunk", i, "of", n_chunks, "\n")
    
    start_idx <- (i - 1) * chunk_size + 1
    end_idx <- min(i * chunk_size, n_total)
    
    chunk_data <- data[start_idx:end_idx, ]
    
    # Process chunk
    # ... your analysis here ...
    
    # Save chunk results
    results[[i]] <- "chunk_results"
    
    # Clear memory
    gc()
  }
  
  return(results)
}

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# Just run step 2 and 3
cat("\n=== EXAMPLE: Running just data processing and MR ===\n")

# Step 1: Config (always needed)
source_python("phase3_00_config.py")

# Step 2: Data processing
source_python("phase3_01_data_processor.py")

# Step 3: MR only
source_python("phase3_03_mr_analysis.py")

cat("\nModules loaded successfully!\n")
cat("You can now use any functions from these modules.\n")