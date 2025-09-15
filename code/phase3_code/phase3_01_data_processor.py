#!/usr/bin/env python3
"""
phase3_01_data_processor.py - REVISED Data Processing with Phase 1&2 Fixes
"""

# Import configuration
from phase3_00_config import *

# =============================================================================
# COMPREHENSIVE DATA PROCESSOR - REVISED
# =============================================================================

class TemporalCausalDataProcessor:
    """Revised data processor incorporating all fixes"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = []
        self.validation_report = {}
        self.mr_results = None
        
    def load_phase1_outputs(self):
        """Load Phase 1 outputs with proper error handling"""
        
        print_header("LOADING PHASE 1 OUTPUTS")
        
        datasets = {}
        
        # Try multiple formats for each cohort
        cohorts = ['primary', 'temporal', 'bias_corrected']
        
        for cohort in cohorts:
            loaded = False
            
            # Try RDS first if pyreadr available
            if PYREADR_AVAILABLE:
                rds_path = os.path.join(Config.BASE_PATH, f"discovery_cohort_{cohort}.rds")
                if os.path.exists(rds_path):
                    try:
                        result = pyreadr.read_r(rds_path)
                        datasets[cohort] = list(result.values())[0]
                        loaded = True
                        print(f"✓ Loaded {cohort} from RDS: {datasets[cohort].shape}")
                    except Exception as e:
                        print(f"  Failed to load RDS: {e}")
            
            # Try CSV as fallback
            if not loaded:
                csv_path = os.path.join(Config.BASE_PATH, f"discovery_cohort_{cohort}.csv")
                if os.path.exists(csv_path):
                    try:
                        datasets[cohort] = pd.read_csv(csv_path)
                        loaded = True
                        print(f"✓ Loaded {cohort} from CSV: {datasets[cohort].shape}")
                    except Exception as e:
                        print(f"  Failed to load CSV: {e}")
            
            # Try pickle as last resort
            if not loaded:
                pkl_path = os.path.join(Config.BASE_PATH, f"discovery_cohort_{cohort}.pkl")
                if os.path.exists(pkl_path):
                    try:
                        datasets[cohort] = joblib.load(pkl_path)
                        loaded = True
                        print(f"✓ Loaded {cohort} from PKL: {datasets[cohort].shape}")
                    except Exception as e:
                        print(f"  Failed to load PKL: {e}")
            
            if not loaded:
                print(f"✗ Could not load {cohort} cohort")
        
        # Validate loaded data
        for name, df in datasets.items():
            self._validate_cohort(df, name)
        
        return datasets
    
    def load_mr_results(self):
        """Load Phase 2.5 MR results with validation"""
        
        print("\nLoading MR results...")
        
        mr_path = os.path.join(Config.MR_PATH, "mr_summary.csv")
        
        if os.path.exists(mr_path):
            try:
                mr_data = pd.read_csv(mr_path)
                print(f"✓ Loaded MR results: {len(mr_data)} associations")
                
                # Validate and flag heterogeneity issues
                self.mr_results = validate_mr_results(mr_data)
                
                # Show significant findings
                if 'significant' in mr_data.columns:
                    sig_results = mr_data[mr_data['significant']]
                    print(f"  Significant associations: {len(sig_results)}")
                    
                    for _, row in sig_results.head(5).iterrows():
                        print(f"    {row['exposure']} → {row['outcome']}: OR={row['OR']:.3f}")
                
                return self.mr_results
                
            except Exception as e:
                print(f"✗ Failed to load MR results: {e}")
                return None
        else:
            print("✗ MR results not found")
            return None
    
    def _validate_cohort(self, df, name):
        """Validate cohort with fixed metabolite detection"""
        
        print(f"\nValidating {name} cohort:")
        
        validation = {
            'n_samples': len(df),
            'n_features': df.shape[1]
        }
        
        # Check essential variables
        essential = ['eid', 'ad_case_primary', 'age_baseline', 'sex']
        validation['essential_present'] = all(col in df.columns for col in essential)
        
        if not validation['essential_present']:
            missing = [col for col in essential if col not in df.columns]
            print(f"  ⚠ Missing essential columns: {missing}")
        
        # FIXED: Use correct metabolite pattern
        import re
        metabolite_pattern = re.compile(Config.METABOLITE_PATTERN)
        met_cols = [col for col in df.columns if metabolite_pattern.match(col)]
        
        validation['n_metabolites'] = len(met_cols)
        
        # Check temporal structure
        met_i0 = [col for col in met_cols if col.endswith('_i0')]
        met_i1 = [col for col in met_cols if col.endswith('_i1')]
        
        validation['n_met_i0'] = len(met_i0)
        validation['n_met_i1'] = len(met_i1)
        validation['has_temporal'] = len(met_i1) > 0
        
        # Check for NAFLD (added from Phase 2.5)
        validation['has_nafld'] = 'has_nafld' in df.columns
        
        # Calculate outcome prevalences
        if validation['essential_present']:
            validation['ad_prevalence'] = df['ad_case_primary'].mean()
            
            for outcome in Config.METABOLIC_OUTCOMES:
                if outcome in df.columns:
                    validation[f'{outcome}_prevalence'] = df[outcome].mean()
        
        # Check data quality
        validation['missing_rate'] = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        validation['duplicate_ids'] = df['eid'].duplicated().sum() if 'eid' in df.columns else 0
        
        self.validation_report[name] = validation
        
        # Print summary
        print(f"  Samples: {validation['n_samples']:,}")
        print(f"  Features: {validation['n_features']:,}")
        print(f"  Metabolites: {validation['n_metabolites']} (i0: {validation['n_met_i0']}, i1: {validation['n_met_i1']})")
        print(f"  Missing rate: {validation['missing_rate']:.1%}")
        print(f"  NAFLD data: {'Yes' if validation['has_nafld'] else 'No'}")
        
        if validation['duplicate_ids'] > 0:
            print(f"  ⚠ WARNING: {validation['duplicate_ids']} duplicate IDs")
        
        return validation
    
    def prepare_temporal_analysis(self, datasets):
        """Prepare data specifically for temporal causal analysis"""
        
        print_header("PREPARING TEMPORAL CAUSAL ANALYSIS DATA")
        
        # Select best available cohort
        if 'temporal' in datasets and datasets['temporal'] is not None:
            df = datasets['temporal'].copy()
            print("Using temporal cohort")
        elif 'bias_corrected' in datasets:
            df = datasets['bias_corrected'].copy()
            print("Using bias-corrected cohort")
        else:
            df = datasets['primary'].copy()
            print("Using primary cohort")
        
        # Extract temporal sequences
        temporal_data = self._extract_temporal_sequences(df)
        
        if temporal_data is None:
            print("⚠ No temporal data available - will use cross-sectional analysis")
            return self._prepare_cross_sectional(df)
        
        # Prepare features and outcomes
        features = self._extract_features(df, temporal_data)
        outcomes = self._extract_outcomes(df)
        
        # Add clinical biomarkers
        clinical = self._extract_clinical_markers(df)
        
        # Create analysis dataset
        analysis_data = {
            'temporal': temporal_data,
            'features': features,
            'outcomes': outcomes,
            'clinical': clinical,
            'demographics': self._extract_demographics(df),
            'mr_results': self.mr_results,
            'metadata': {
                'n_samples': len(df),
                'n_metabolites': len(temporal_data['metabolite_names']) if temporal_data else 0,
                'n_timepoints': temporal_data['n_timepoints'] if temporal_data else 1,
                'has_nafld': 'has_nafld' in outcomes.columns
            }
        }
        
        return analysis_data
    
    def _extract_temporal_sequences(self, df):
        """Extract temporal sequences with proper pattern matching"""
        
        print("\nExtracting temporal sequences...")
        
        # FIXED: Use correct pattern
        import re
        met_pattern = re.compile(r'^(p\d{5})_i(\d+)$')
        
        # Find all metabolite base names and instances
        metabolite_instances = {}
        
        for col in df.columns:
            match = met_pattern.match(col)
            if match:
                base_name = match.group(1)
                instance = int(match.group(2))
                
                if base_name not in metabolite_instances:
                    metabolite_instances[base_name] = []
                metabolite_instances[base_name].append(instance)
        
        # Find metabolites with at least 2 timepoints
        temporal_metabolites = {
            base: sorted(instances) 
            for base, instances in metabolite_instances.items()
            if len(instances) >= 2
        }
        
        if not temporal_metabolites:
            print("  No temporal metabolites found")
            return None
        
        print(f"  Found {len(temporal_metabolites)} metabolites with temporal data")
        
        # Determine common timepoints
        all_instances = set()
        for instances in temporal_metabolites.values():
            all_instances.update(instances)
        
        common_instances = sorted(all_instances)[:2]  # Use first 2 timepoints
        print(f"  Using instances: {common_instances}")
        
        # Extract temporal data
        valid_metabolites = []
        for base, instances in temporal_metabolites.items():
            if all(i in instances for i in common_instances):
                valid_metabolites.append(base)
        
        print(f"  {len(valid_metabolites)} metabolites have complete data")
        
        # Create temporal array
        n_samples = len(df)
        n_metabolites = len(valid_metabolites)
        n_time = len(common_instances)
        
        temporal_array = np.zeros((n_samples, n_metabolites, n_time))
        
        for j, base in enumerate(valid_metabolites):
            for t, instance in enumerate(common_instances):
                col = f'{base}_i{instance}'
                if col in df.columns:
                    temporal_array[:, j, t] = df[col].fillna(0).values
        
        # Calculate changes
        delta = temporal_array[:, :, 1] - temporal_array[:, :, 0]
        
        # Handle division by zero for percent change
        baseline = temporal_array[:, :, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_change = np.where(
                baseline != 0,
                (delta / baseline) * 100,
                0
            )
        
        temporal_data = {
            'sequences': temporal_array,
            'metabolite_names': valid_metabolites,
            'n_timepoints': n_time,
            'instances': common_instances,
            'delta': delta,
            'percent_change': percent_change
        }
        
        return temporal_data
    
    def _extract_features(self, df, temporal_data=None):
        """Extract features for analysis"""
        
        features = {}
        
        # Basic features
        basic_features = ['age_baseline', 'bmi_i0']
        for feat in basic_features:
            if feat in df.columns:
                features[feat] = df[feat].fillna(df[feat].median()).values
        
        # Sex encoding
        if 'sex' in df.columns:
            features['sex'] = (df['sex'] == 'Male').astype(int).values
        elif 'sex_binary' in df.columns:
            features['sex'] = df['sex_binary'].values
        
        # Townsend index
        if 'townsend_index' in df.columns:
            features['townsend_index'] = df['townsend_index'].fillna(0).values
        
        # Genetic PCs
        pc_cols = [f'pc{i}' for i in range(1, 11)]
        available_pcs = [col for col in pc_cols if col in df.columns]
        if available_pcs:
            pc_data = df[available_pcs].fillna(0).values
            features['genetic_pcs'] = pc_data
        
        # Add temporal features if available
        if temporal_data is not None:
            # Use baseline metabolites as features
            features['metabolites_baseline'] = temporal_data['sequences'][:, :, 0]
            # Use change as features
            features['metabolites_delta'] = temporal_data['delta']
        
        return features
    
    def _extract_outcomes(self, df):
        """Extract outcome variables including NAFLD"""
        
        outcomes = pd.DataFrame(index=df.index)
        
        # Primary exposure
        outcomes['ad_case'] = df['ad_case_primary'].astype(int)
        
        # All metabolic outcomes including NAFLD
        for outcome in Config.METABOLIC_OUTCOMES:
            if outcome in df.columns:
                outcomes[outcome] = df[outcome].astype(int)
            else:
                # Try alternative names
                alt_names = {
                    'has_nafld': ['has_fatty_liver', 'nafld', 'fatty_liver'],
                    'has_gout_any': ['has_gout', 'gout'],
                    'has_hyperuricemia': ['hyperuricemia', 'high_uric_acid']
                }
                
                if outcome in alt_names:
                    for alt in alt_names[outcome]:
                        if alt in df.columns:
                            outcomes[outcome] = df[alt].astype(int)
                            break
        
        # Create composite outcomes
        metabolic_cols = [col for col in outcomes.columns if col.startswith('has_')]
        if len(metabolic_cols) >= 2:
            outcomes['n_metabolic'] = outcomes[metabolic_cols].sum(axis=1)
            outcomes['multiple_metabolic'] = (outcomes['n_metabolic'] >= 2).astype(int)
        
        print(f"\nExtracted {len(outcomes.columns)} outcome variables")
        print(f"  Including NAFLD: {'has_nafld' in outcomes.columns}")
        
        return outcomes
    
    def _extract_clinical_markers(self, df):
        """Extract clinical biomarkers with UK-specific thresholds"""
        
        clinical = pd.DataFrame(index=df.index)
        
        # Map of biomarker names to possible column names
        biomarker_map = {
            'glucose': ['glucose_i0', 'glucose', 'p30740_i0'],
            'hba1c': ['hba1c_i0', 'hba1c', 'p30750_i0'],
            'cholesterol': ['cholesterol_i0', 'p30690_i0'],
            'ldl': ['ldl_i0', 'p30780_i0'],
            'hdl': ['hdl_i0', 'p30760_i0'],
            'triglycerides': ['triglycerides_i0', 'p30870_i0'],
            'crp': ['crp_i0', 'p30710_i0'],
            'alt': ['alt_i0', 'p30620_i0'],
            'urate': ['urate_i0', 'uric_acid_i0', 'p30880_i0']
        }
        
        for biomarker, possible_cols in biomarker_map.items():
            for col in possible_cols:
                if col in df.columns:
                    clinical[biomarker] = df[col]
                    
                    # Create abnormal flags using UK thresholds
                    if biomarker in Config.CLINICAL_THRESHOLDS:
                        threshold = Config.CLINICAL_THRESHOLDS[biomarker]
                        
                        if biomarker == 'hdl_low':
                            clinical[f'{biomarker}_abnormal'] = (df[col] < threshold).astype(int)
                        else:
                            clinical[f'{biomarker}_abnormal'] = (df[col] > threshold).astype(int)
                    break
        
        print(f"Extracted {len(clinical.columns)} clinical markers")
        
        return clinical
    
    def _extract_demographics(self, df):
        """Extract demographic variables"""
        
        demographics = pd.DataFrame(index=df.index)
        
        demo_vars = ['age_baseline', 'sex', 'ethnicity', 'townsend_index', 
                    'education', 'assessment_centre']
        
        for var in demo_vars:
            if var in df.columns:
                demographics[var] = df[var]
        
        # Create age groups
        if 'age_baseline' in demographics.columns:
            demographics['age_group'] = pd.cut(
                demographics['age_baseline'],
                bins=[0, 50, 60, 70, 100],
                labels=['<50', '50-59', '60-69', '≥70']
            )
        
        # Create deprivation quintiles
        if 'townsend_index' in demographics.columns:
            demographics['deprivation_quintile'] = pd.qcut(
                demographics['townsend_index'],
                q=5,
                labels=['Q1-Least', 'Q2', 'Q3', 'Q4', 'Q5-Most']
            )
        
        return demographics
    
    def _prepare_cross_sectional(self, df):
        """Fallback for cross-sectional analysis if no temporal data"""
        
        print("\nPreparing cross-sectional analysis...")
        
        # Extract baseline metabolites only
        import re
        met_pattern = re.compile(r'^p\d{5}_i0$')
        met_cols = [col for col in df.columns if met_pattern.match(col)]
        
        if met_cols:
            metabolite_data = df[met_cols].fillna(0).values
            metabolite_names = [col.replace('_i0', '') for col in met_cols]
        else:
            metabolite_data = None
            metabolite_names = []
        
        features = self._extract_features(df, None)
        outcomes = self._extract_outcomes(df)
        clinical = self._extract_clinical_markers(df)
        demographics = self._extract_demographics(df)
        
        return {
            'features': features,
            'outcomes': outcomes,
            'clinical': clinical,
            'demographics': demographics,
            'metabolite_data': metabolite_data,
            'metabolite_names': metabolite_names,
            'mr_results': self.mr_results,
            'metadata': {
                'n_samples': len(df),
                'n_metabolites': len(metabolite_names),
                'analysis_type': 'cross_sectional',
                'has_nafld': 'has_nafld' in outcomes.columns
            }
        }
    
    def save_processed_data(self, data, filename="temporal_analysis_data.pkl"):
        """Save processed data"""
        
        filepath = os.path.join(Config.OUTPUT_PATH, 'results', filename)
        joblib.dump(data, filepath)
        print(f"\n✓ Saved processed data: {filepath}")
        
        # Also save summary
        summary = {
            'n_samples': data['metadata']['n_samples'],
            'analysis_type': data['metadata'].get('analysis_type', 'temporal'),
            'n_metabolites': data['metadata']['n_metabolites'],
            'n_timepoints': data['metadata'].get('n_timepoints', 1),
            'has_nafld': data['metadata']['has_nafld'],
            'outcomes': list(data['outcomes'].columns) if 'outcomes' in data else [],
            'timestamp': datetime.now().isoformat()
        }
        
        save_results(summary, 'data_processing_summary.json')
        
        return filepath

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_header("PHASE 3 DATA PROCESSING - REVISED")
    
    # Initialize processor
    processor = TemporalCausalDataProcessor()
    
    # Load Phase 1 outputs
    datasets = processor.load_phase1_outputs()
    
    if not datasets:
        print("\n✗ ERROR: No datasets loaded from Phase 1")
        sys.exit(1)
    
    # Load MR results from Phase 2.5
    mr_results = processor.load_mr_results()
    
    # Prepare temporal analysis data
    analysis_data = processor.prepare_temporal_analysis(datasets)
    
    # Save processed data
    processor.save_processed_data(analysis_data)
    
    # Save validation report
    save_results(processor.validation_report, 'validation_report.json')
    
    # Print summary
    print_header("PROCESSING COMPLETE")
    
    print("\n📊 Data Summary:")
    print(f"  Samples: {analysis_data['metadata']['n_samples']:,}")
    print(f"  Metabolites: {analysis_data['metadata']['n_metabolites']}")
    
    if 'temporal' in analysis_data and analysis_data['temporal']:
        print(f"  Timepoints: {analysis_data['temporal']['n_timepoints']}")
        print(f"  Temporal metabolites: {len(analysis_data['temporal']['metabolite_names'])}")
    
    if 'has_nafld' in analysis_data['metadata']:
        print(f"  NAFLD data: {'Available' if analysis_data['metadata']['has_nafld'] else 'Not available'}")
    
    if mr_results is not None:
        print(f"\n🧬 MR Integration:")
        print(f"  Total MR tests: {len(mr_results)}")
        if 'significant' in mr_results.columns:
            print(f"  Significant: {mr_results['significant'].sum()}")
        if 'extreme_heterogeneity' in mr_results.columns:
            print(f"  With extreme heterogeneity: {mr_results['extreme_heterogeneity'].sum()}")
    
    print(f"\n✓ Ready for temporal causal discovery analysis")
    print(f"  Output directory: {Config.OUTPUT_PATH}")
