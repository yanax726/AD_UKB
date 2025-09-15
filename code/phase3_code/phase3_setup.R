# UKB RAP RStudio Environment Setup Script
# This script installs all necessary R and Python packages for your analysis

# Function to install R packages if not already installed
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) {
    install.packages(new_packages, repos = "https://cloud.r-project.org/")
  }
}

# Install essential R packages
cat("Installing R packages...\n")
r_packages <- c("reticulate", "devtools", "tidyverse", "data.table")
install_if_missing(r_packages)

# Load reticulate
library(reticulate)

# Check Python availability and create/use conda environment
cat("\nSetting up Python environment...\n")

# Try to use conda if available, otherwise use virtualenv
tryCatch({
  # Check if conda is available
  conda_exe <- conda_binary()
  
  # Create a conda environment specifically for this project
  env_name <- "ukb_rap_env"
  
  # Check if environment already exists
  envs <- conda_list()
  if(!(env_name %in% envs$name)) {
    cat("Creating new conda environment:", env_name, "\n")
    conda_create(env_name, python_version = "3.9")
  } else {
    cat("Using existing conda environment:", env_name, "\n")
  }
  
  # Use the conda environment
  use_condaenv(env_name, required = TRUE)
  
}, error = function(e) {
  cat("Conda not available, using virtualenv instead\n")
  
  # Create a virtual environment
  env_path <- "~/ukb_rap_venv"
  
  if(!virtualenv_exists(env_path)) {
    cat("Creating virtual environment at:", env_path, "\n")
    virtualenv_create(env_path, python = NULL)
  }
  
  # Use the virtual environment
  use_virtualenv(env_path, required = TRUE)
})

# Get the Python configuration
py_config()

# Install Python packages
cat("\nInstalling Python packages... This may take several minutes.\n")

# Core packages
core_packages <- c(
  "pandas==1.5.3",  # Specific version for compatibility
  "numpy",
  "scipy",
  "scikit-learn",
  "matplotlib",
  "seaborn",
  "tqdm",
  "statsmodels",
  "joblib",
  "networkx"
)

# Install core packages
for(pkg in core_packages) {
  cat("Installing:", pkg, "\n")
  py_install(pkg, pip = TRUE)
}

# Install PyTorch (CPU version for UKB RAP)
cat("\nInstalling PyTorch (CPU version)...\n")
py_install("torch", pip = TRUE, pip_options = "--index-url https://download.pytorch.org/whl/cpu")

# Install optional packages with error handling
optional_packages <- c(
  "pyreadr",
  "lifelines",
  "pingouin"
)

for(pkg in optional_packages) {
  tryCatch({
    cat("Installing optional package:", pkg, "\n")
    py_install(pkg, pip = TRUE)
  }, error = function(e) {
    cat("Warning: Could not install", pkg, "- continuing without it\n")
  })
}

# Verify installations
cat("\n\nVerifying Python package installations...\n")
cat("====================================\n")

# Function to check if a Python module is available
check_python_module <- function(module_name, import_name = NULL) {
  if(is.null(import_name)) import_name <- module_name
  
  tryCatch({
    py_module <- import(import_name)
    version <- py_module$`__version__`
    cat(sprintf("✓ %-20s : %s\n", module_name, version))
    return(TRUE)
  }, error = function(e) {
    cat(sprintf("✗ %-20s : Not installed\n", module_name))
    return(FALSE)
  })
}

# Check all required modules
modules_to_check <- list(
  c("pandas", "pd"),
  c("numpy", "np"),
  c("scipy", "scipy"),
  c("sklearn", "sklearn"),
  c("torch", "torch"),
  c("matplotlib", "matplotlib"),
  c("seaborn", "sns"),
  c("tqdm", "tqdm"),
  c("statsmodels", "statsmodels"),
  c("joblib", "joblib"),
  c("networkx", "nx")
)

all_installed <- TRUE
for(module_info in modules_to_check) {
  module_name <- module_info[1]
  import_name <- ifelse(length(module_info) > 1, module_info[2], module_info[1])
  if(!check_python_module(module_name, import_name)) {
    all_installed <- FALSE
  }
}

# Check optional modules
cat("\nOptional modules:\n")
for(module in optional_packages) {
  check_python_module(module, module)
}

# Test the pandas compatibility fix
cat("\n\nTesting pandas compatibility fix...\n")
py_run_string("
import pandas
import pandas.core.arrays

# Apply compatibility fix
if not hasattr(pandas.arrays, 'NumpyExtensionArray'):
    if hasattr(pandas.core.arrays, 'NumpyExtensionArray'):
        pandas.arrays.NumpyExtensionArray = pandas.core.arrays.NumpyExtensionArray
        print('✓ Pandas compatibility fix applied successfully')
    else:
        class NumpyExtensionArray:
            pass
        pandas.arrays.NumpyExtensionArray = NumpyExtensionArray
        print('✓ Pandas compatibility fix applied with dummy class')
else:
    print('✓ Pandas already has NumpyExtensionArray')
")

# Create a test script to verify everything works
cat("\n\nCreating test script...\n")

test_code <- '
# Test script to verify all imports work
import pandas
import pandas.core.arrays

# Apply compatibility fix
if not hasattr(pandas.arrays, "NumpyExtensionArray"):
    if hasattr(pandas.core.arrays, "NumpyExtensionArray"):
        pandas.arrays.NumpyExtensionArray = pandas.core.arrays.NumpyExtensionArray
    else:
        class NumpyExtensionArray:
            pass
        pandas.arrays.NumpyExtensionArray = NumpyExtensionArray

# Now test all imports
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.stats as stats
from scipy import linalg
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from datetime import datetime
import gc
import time
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.nonparametric.smoothers_lowess import lowess
import joblib
from functools import lru_cache
import networkx as nx
from collections import defaultdict
import itertools
from pathlib import Path

print("\\n✓ All core imports successful!")

# Test optional imports
try:
    import pyreadr
    print("✓ pyreadr available")
except ImportError:
    print("⚠ pyreadr not available")

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    print("✓ lifelines available")
except ImportError:
    print("⚠ lifelines not available")

try:
    import pingouin as pg
    print("✓ pingouin available")
except ImportError:
    print("⚠ pingouin not available")

# Quick functionality test
print("\\nRunning quick functionality test...")
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Pandas DataFrame shape: {df.shape}")
print(f"PyTorch tensor shape: {X.shape}")
print("\\n✓ Basic functionality verified!")
'

# Run the test
cat("\nRunning import test...\n")
tryCatch({
  py_run_string(test_code)
  cat("\n✅ All imports successful! Your environment is ready.\n")
}, error = function(e) {
  cat("\n❌ Error during import test:\n")
  print(e)
  cat("\nPlease check the error message above and re-run the setup if needed.\n")
})

# Save the setup information
cat("\nSaving environment information...\n")
env_info <- list(
  python_version = py_version(),
  python_path = py_exe(),
  timestamp = Sys.time(),
  packages_installed = core_packages
)

saveRDS(env_info, "ukb_rap_env_info.rds")

# Print usage instructions
cat("\n" ,strrep("=", 60), "\n")
cat("SETUP COMPLETE!\n")
cat(strrep("=", 60), "\n")
cat("\nTo use this environment in future R sessions:\n")
cat("1. Load reticulate: library(reticulate)\n")

if(exists("env_name")) {
  cat("2. Activate conda environment: use_condaenv('", env_name, "')\n", sep="")
} else if(exists("env_path")) {
  cat("2. Activate virtual environment: use_virtualenv('", env_path, "')\n", sep="")
}

cat("\n3. Your Python code should now run without import errors!\n")
cat("\nEnvironment info saved to: ukb_rap_env_info.rds\n")