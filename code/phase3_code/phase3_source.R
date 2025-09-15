# First check what pandas version is REALLY installed
system("python3 -c 'import pandas; print(pandas.__version__)'")

# Clean the corrupted installation
system("rm -rf ~/.local/lib/python3.10/site-packages/*tatsmodels*")

# If pandas is not 1.5.3, force reinstall
system("python3 -m pip install --user --force-reinstall --no-deps pandas==1.5.3")
library(reticulate)
# Reinstall lifelines
system("python3 -m pip install --user lifelines")
setwd("~/results/discovery_pipeline_bias_corrected")
source_python("phase3_analysis.py")
source_python("phase3_00_config.py")
source_python("phase3_01_data_processor.py")