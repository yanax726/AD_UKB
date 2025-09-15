# Run in order:
source_python("phase3_00_config.py")
source_python("phase3_01_data_processor.py")
source_python("phase3_02_causalformer.py")
source_python("phase3_02_diagnostic_and_fix.py")
source_python("phase3_02_comprehensive_data_finder.py")
source_python("phase3_03_mr_analysis.py")
source_python("phase3_04_mediation.py")
source_python("phase3_05_heterogeneity.py")
source_python("phase3_06_temporal.py")
source_python()
python phase3_01_data_processor.py  # Process and save data
python phase3_02_causalformer.py    # Train neural network
python phase3_03_mr_analysis.py     # Enhanced MR analysis
python phase3_04_mediation.py       # Mediation analysis
python phase3_05_heterogeneity.py   # Heterogeneous effects
python phase3_06_temporal.py        # Temporal causality


# In RStudio, run this:
library(reticulate)
use_python("/usr/bin/python3")
writeLines(readLines("causalformer_analysis_only.py"), "causalformer_analysis_only.py")
# Run just the analysis
py_run_file("causalformer_analysis_only.py")
py_run_file("pipeline.py")
py_run_file("dp.py")
py_run_file("causal_insight.py")
py_run_file("analysis.py")