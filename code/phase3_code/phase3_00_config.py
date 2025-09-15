#!/usr/bin/env python3
"""
phase3_00_config.py - Configuration and Setup for UK Biobank AD-Metabolic Analysis
REVISED VERSION: Incorporates all fixes from Phase 1 & 2.5
"""

# COMPATIBILITY FIXES - Handle various pandas versions
import sys
import pandas
import pandas.core.arrays

# Fix for pandas compatibility across versions
if not hasattr(pandas.arrays, 'NumpyExtensionArray'):
    if hasattr(pandas.core.arrays, 'NumpyExtensionArray'):
        pandas.arrays.NumpyExtensionArray = pandas.core.arrays.NumpyExtensionArray
    else:
        class NumpyExtensionArray:
            pass
        pandas.arrays.NumpyExtensionArray = NumpyExtensionArray

# Standard imports
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import gc
import joblib
from pathlib import Path

# Handle PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("✓ PyTorch available")
except ImportError:
    print("⚠ PyTorch not available - will use alternative methods")
    TORCH_AVAILABLE = False
    torch = None

# Essential imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.impute import KNNImputer, SimpleImputer

# Handle optional sklearn components
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    print("⚠ IterativeImputer not available - will use simple imputation")
    ITERATIVE_IMPUTER_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats
from scipy import linalg
from statsmodels.stats.multitest import multipletests

# Optional advanced packages
try:
    import pyreadr
    print("✓ pyreadr available for R data files")
    PYREADR_AVAILABLE = True
except ImportError:
    print("⚠ pyreadr not available - will use alternative formats")
    PYREADR_AVAILABLE = False

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    print("✓ lifelines available for survival analysis")
    LIFELINES_AVAILABLE = True
except ImportError:
    print("⚠ lifelines not available - survival analysis will be limited")
    LIFELINES_AVAILABLE = False

try:
    import pingouin as pg
    print("✓ pingouin available for statistical tests")
    PINGOUIN_AVAILABLE = True
except ImportError:
    print("⚠ pingouin not available - will use scipy stats")
    PINGOUIN_AVAILABLE = False

# Set seeds for reproducibility
np.random.seed(42)
if TORCH_AVAILABLE and torch is not None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

# =============================================================================
# CONFIGURATION CLASS - REVISED WITH PHASE 1&2 FIXES
# =============================================================================

class Config:
    """Enhanced configuration incorporating all previous fixes"""
    
    # Paths - Match actual Phase 1 output structure
    BASE_PATH = os.path.expanduser("~/results/discovery_pipeline_bias_corrected")
    OUTPUT_PATH = os.path.join(BASE_PATH, "phase3_temporal_causal")
    MR_PATH = os.path.join(BASE_PATH, "mendelian_randomization")
    
    # FIXED: Correct metabolite pattern from Phase 1
    METABOLITE_PATTERN = r'^p\d{5}_i\d+$'  # 5 digits, not 23xxx
    
    # Model parameters - Adjusted for actual data
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = None
    
    CAUSALFORMER_CONFIG = {
        'd_model': 128,          # Reduced for smaller temporal cohort
        'n_heads': 4,           
        'n_layers': 3,           # Shallower due to limited data
        'dropout': 0.2,          # Higher dropout for small sample
        'max_seq_length': 4,     
        'kernel_sizes': [3, 5, 7],
        'dilation_rates': [1, 2, 4]
    }
    
    # Training parameters
    BATCH_SIZE = 32          # Smaller batch for limited data
    LEARNING_RATE = 5e-5     # Lower LR for stability
    MAX_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 7
    GRADIENT_CLIP = 1.0
    WEIGHT_DECAY = 1e-4
    
    # Causal discovery parameters - Adjusted for MR heterogeneity
    STABILITY_SELECTION_RUNS = 50   # Reduced for speed
    SUBSAMPLE_RATIO = 0.8           # Higher ratio for small samples
    ALPHA_THRESHOLD = 0.05          # Less stringent due to multiple testing
    FDR_THRESHOLD = 0.1             # Relaxed for discovery
    PC_ALPHA = 0.05
    MIN_EDGE_STRENGTH = 0.2         # Lower threshold
    
    # MR-specific parameters for handling heterogeneity
    MR_HETEROGENEITY_THRESHOLD = 1000  # Flag extreme I² values
    MR_METHODS = ['mr_ivw', 'mr_egger_regression', 'mr_weighted_median']
    
    # Clinical thresholds - UK Biobank specific
    CLINICAL_THRESHOLDS = {
        'hba1c_diabetes': 48,        # mmol/mol (UK units)
        'glucose_diabetes': 7.0,     # mmol/L
        'bmi_obesity': 30.0,
        'ldl_high': 3.0,            # mmol/L (UK guidelines)
        'hdl_low': 1.0,             # mmol/L
        'triglycerides_high': 1.7,  # mmol/L
        'crp_high': 3.0,            # mg/L
        'alt_high': 40,             # U/L
        'urate_high': 360           # μmol/L (UK units)
    }
    
    # ADDED: Include NAFLD/fatty liver
    METABOLIC_OUTCOMES = [
        'has_diabetes_any',
        'has_hypertension_any', 
        'has_obesity_any',
        'has_hyperlipidemia_any',
        'has_hyperuricemia',
        'has_gout_any',
        'has_nafld',              # Added from Phase 2.5
        'has_metabolic_syndrome',
        'has_ihd_any',
        'has_stroke_any'
    ]
    
    # Computational settings
    N_JOBS = -1
    RANDOM_STATE = 42
    
    # Output subdirectories
    SUBDIRS = [
        'models', 'results', 'figures', 'tables', 'checkpoints',
        'validation', 'temporal', 'mediation', 'heterogeneity',
        'mr_integration', 'clinical', 'supplementary'
    ]

# Create output directories
for subdir in Config.SUBDIRS:
    os.makedirs(os.path.join(Config.OUTPUT_PATH, subdir), exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS - ENHANCED
# =============================================================================

def print_header(text, width=80):
    """Print formatted header"""
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width)

def check_environment():
    """Check and report environment status"""
    print_header("ENVIRONMENT CHECK")
    
    env_status = {
        'PyTorch': TORCH_AVAILABLE,
        'pyreadr': PYREADR_AVAILABLE,
        'lifelines': LIFELINES_AVAILABLE,
        'pingouin': PINGOUIN_AVAILABLE,
        'IterativeImputer': ITERATIVE_IMPUTER_AVAILABLE
    }
    
    print("\nPackage availability:")
    for package, available in env_status.items():
        status = "✓" if available else "✗"
        print(f"  {status} {package}: {available}")
    
    if TORCH_AVAILABLE:
        print(f"\nCompute device: {Config.DEVICE}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return env_status

def validate_mr_results(mr_data):
    """Validate and flag MR results with extreme heterogeneity"""
    if mr_data is None or len(mr_data) == 0:
        return None
    
    # Flag extreme heterogeneity
    if 'heterogeneity_i2' in mr_data.columns:
        mr_data['extreme_heterogeneity'] = mr_data['heterogeneity_i2'] > Config.MR_HETEROGENEITY_THRESHOLD
        
        n_extreme = mr_data['extreme_heterogeneity'].sum()
        if n_extreme > 0:
            print(f"\n⚠ WARNING: {n_extreme} MR analyses show extreme heterogeneity (I²>{Config.MR_HETEROGENEITY_THRESHOLD}%)")
            print("  These results should be interpreted with caution")
    
    return mr_data

def save_results(results, filename, subdir='results'):
    """Enhanced save function with type handling"""
    output_dir = os.path.join(Config.OUTPUT_PATH, subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    # Handle different file types
    if filename.endswith('.csv') and isinstance(results, pd.DataFrame):
        results.to_csv(filepath, index=False)
    elif filename.endswith('.pkl'):
        joblib.dump(results, filepath)
    elif filename.endswith('.json'):
        # Convert numpy/pandas types to native Python
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            # FIXED: Check for scalar NaN AFTER checking for lists/arrays
            elif np.isscalar(obj) and pd.isna(obj):
                return None
            else:
                return obj
        
        results_clean = convert_to_native(results)
        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=2)
    
    print(f"Saved: {filepath}")
    return filepath

# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    print_header("UK BIOBANK AD-METABOLIC PHASE 3: TEMPORAL CAUSAL DISCOVERY")
    print(f"Version: 3.0-REVISED")
    print(f"Date: {datetime.now()}")
    print(f"Output path: {Config.OUTPUT_PATH}")
    
    # Check environment
    env_status = check_environment()
    
    # Save configuration
    config_dict = {
        'paths': {
            'base': Config.BASE_PATH,
            'output': Config.OUTPUT_PATH,
            'mr': Config.MR_PATH
        },
        'environment': env_status,
        'parameters': {
            'causal_discovery': {
                'alpha': Config.ALPHA_THRESHOLD,
                'fdr': Config.FDR_THRESHOLD,
                'min_edge_strength': Config.MIN_EDGE_STRENGTH
            },
            'mr_integration': {
                'heterogeneity_threshold': Config.MR_HETEROGENEITY_THRESHOLD,
                'methods': Config.MR_METHODS
            }
        },
        'outcomes': Config.METABOLIC_OUTCOMES,
        'timestamp': datetime.now().isoformat()
    }
    
    save_results(config_dict, 'phase3_configuration.json')
    print("\n✓ Configuration complete and saved")
