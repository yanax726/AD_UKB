#!/usr/bin/env python3
"""
phase3_05_heterogeneity.py - Heterogeneous Treatment Effects Analysis
"""

# Import configuration
try:
    from phase3_00_config import *
except ImportError:
    print("ERROR: Please run phase3_00_config.py first!")
    raise

import pickle

# =============================================================================
# HETEROGENEOUS EFFECTS ANALYZER
# =============================================================================

class HeterogeneousEffectsAnalyzer:
    """Analyze heterogeneous treatment effects with machine learning methods"""
    
    def __init__(self, method='causal_forest', n_estimators=1000, max_depth=None):
        self.method = method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.results = {}
        
    def analyze_heterogeneity(self, treatment, outcome, features, feature_names=None,
                            subgroups=None, test_interactions=True):
        """
        Comprehensive heterogeneous treatment effects analysis
        """
        
        print(f"\n  Analyzing heterogeneity with {features.shape[1]} features...")
        
        # Estimate conditional average treatment effects (CATE)
        if self.method == 'causal_forest':
            cate_estimates, model = self._estimate_cate_forest(
                treatment, outcome, features
            )
        elif self.method == 'meta_learners':
            cate_estimates, model = self._estimate_cate_metalearners(
                treatment, outcome, features
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Test for heterogeneity
        het_test = self._test_heterogeneity(cate_estimates, features)
        
        # Variable importance for effect modification
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(features.shape[1])]
        
        modifier_importance = self._identify_effect_modifiers(
            cate_estimates, features, feature_names, model
        )
        
        # Subgroup analysis
        subgroup_effects = {}
        if subgroups is not None:
            for name, mask in subgroups.items():
                if mask.sum() > 30:
                    subgroup_effects[name] = self._analyze_subgroup(
                        cate_estimates[mask], treatment[mask], outcome[mask]
                    )
        
        # Test specific interactions if requested
        interaction_tests = {}
        if test_interactions:
            interaction_tests = self._test_interactions(
                treatment, outcome, features, feature_names
            )
        
        # Optimal treatment rules
        treatment_rules = self._derive_treatment_rules(
            cate_estimates, features, feature_names
        )
        
        # Policy evaluation
        policy_value = self._evaluate_policy(
            cate_estimates, treatment, outcome
        )
        
        self.results = {
            'cate_estimates': cate_estimates,
            'cate_mean': cate_estimates.mean(),
            'cate_std': cate_estimates.std(),
            'cate_quantiles': np.percentile(cate_estimates, [10, 25, 50, 75, 90]),
            'heterogeneity_test': het_test,
            'modifier_importance': modifier_importance,
            'subgroup_effects': subgroup_effects,
            'interaction_tests': interaction_tests,
            'treatment_rules': treatment_rules,
            'policy_value': policy_value,
            'model': model
        }
        
        return self
    
    def _estimate_cate_forest(self, treatment, outcome, features):
        """Estimate CATE using random forests (R-learner approach)"""
        
        # First stage: estimate propensity scores
        print("    Estimating propensity scores...")
        prop_model = RandomForestClassifier(
            n_estimators=self.n_estimators // 2,
            max_depth=self.max_depth,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        )
        prop_model.fit(features, treatment)
        prop_scores = prop_model.predict_proba(features)[:, 1]
        
        # Clip propensity scores for stability
        prop_scores = np.clip(prop_scores, 0.01, 0.99)
        
        # Second stage: outcome models
        print("    Estimating outcome models...")
        
        # Better binary outcome detection
        unique_outcomes = np.unique(outcome)
        is_binary = (
            outcome.dtype == bool or 
            (len(unique_outcomes) == 2 and 
             set(unique_outcomes).issubset({0, 1, True, False}))
        )
        
        # Model for treated
        treated_idx = treatment == 1
        
        if is_binary:
            # Binary outcome - use classifier
            outcome_model_1 = RandomForestClassifier(
                n_estimators=self.n_estimators // 2,
                max_depth=self.max_depth,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42
            )
        else:
            # Continuous outcome - use regressor
            outcome_model_1 = RandomForestRegressor(
                n_estimators=self.n_estimators // 2,
                max_depth=self.max_depth,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42
            )
        
        outcome_model_1.fit(features[treated_idx], outcome[treated_idx])
        
        # Model for control
        control_idx = treatment == 0
        
        if is_binary:
            outcome_model_0 = RandomForestClassifier(
                n_estimators=self.n_estimators // 2,
                max_depth=self.max_depth,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42
            )
        else:
            outcome_model_0 = RandomForestRegressor(
                n_estimators=self.n_estimators // 2,
                max_depth=self.max_depth,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42
            )
        
        outcome_model_0.fit(features[control_idx], outcome[control_idx])
        
        # Predict potential outcomes
        if hasattr(outcome_model_1, 'predict_proba'):
            y1_pred = outcome_model_1.predict_proba(features)[:, 1]
            y0_pred = outcome_model_0.predict_proba(features)[:, 1]
        else:
            y1_pred = outcome_model_1.predict(features)
            y0_pred = outcome_model_0.predict(features)
        
        # CATE = E[Y|T=1,X] - E[Y|T=0,X]
        cate = y1_pred - y0_pred
        
        # Store models
        model = {
            'propensity_model': prop_model,
            'outcome_model_treated': outcome_model_1,
            'outcome_model_control': outcome_model_0,
            'propensity_scores': prop_scores
        }
        
        return cate, model
    
    def _estimate_cate_metalearners(self, treatment, outcome, features):
        """Estimate CATE using meta-learners (S-learner, T-learner, X-learner)"""
        
        # Determine if outcome is binary
        unique_outcomes = np.unique(outcome)
        is_binary = (
            outcome.dtype == bool or 
            (len(unique_outcomes) == 2 and 
             set(unique_outcomes).issubset({0, 1, True, False}))
        )
        
        # S-learner
        print("    S-learner...")
        if is_binary:
            s_learner = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=-1,
                random_state=42
            )
        else:
            s_learner = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=-1,
                random_state=42
            )
        
        X_with_treatment = np.column_stack([features, treatment])
        s_learner.fit(X_with_treatment, outcome)
        
        # Predict with and without treatment
        X_treated = np.column_stack([features, np.ones(len(features))])
        X_control = np.column_stack([features, np.zeros(len(features))])
        
        if hasattr(s_learner, 'predict_proba'):
            cate_s = (s_learner.predict_proba(X_treated)[:, 1] - 
                     s_learner.predict_proba(X_control)[:, 1])
        else:
            cate_s = s_learner.predict(X_treated) - s_learner.predict(X_control)
        
        # T-learner (already implemented above as forest method)
        cate_t, t_models = self._estimate_cate_forest(treatment, outcome, features)
        
        # X-learner
        print("    X-learner...")
        
        # Stage 1: Same as T-learner
        if hasattr(t_models['outcome_model_treated'], 'predict_proba'):
            y1_pred = t_models['outcome_model_treated'].predict_proba(features)[:, 1]
            y0_pred = t_models['outcome_model_control'].predict_proba(features)[:, 1]
        else:
            y1_pred = t_models['outcome_model_treated'].predict(features)
            y0_pred = t_models['outcome_model_control'].predict(features)
        
        # Stage 2: Imputed treatment effects
        treated_idx = treatment == 1
        control_idx = treatment == 0
        
        tau_treated = outcome[treated_idx] - y0_pred[treated_idx]
        tau_control = y1_pred[control_idx] - outcome[control_idx]
        
        # Stage 3: CATE models
        cate_model_treated = RandomForestRegressor(
            n_estimators=self.n_estimators // 2,
            n_jobs=-1,
            random_state=42
        )
        cate_model_treated.fit(features[treated_idx], tau_treated)
        
        cate_model_control = RandomForestRegressor(
            n_estimators=self.n_estimators // 2,
            n_jobs=-1,
            random_state=42
        )
        cate_model_control.fit(features[control_idx], tau_control)
        
        # Combine using propensity scores
        prop_scores = t_models['propensity_scores']
        cate_x = (prop_scores * cate_model_treated.predict(features) + 
                 (1 - prop_scores) * cate_model_control.predict(features))
        
        # Average across learners
        cate_ensemble = (cate_s + cate_t + cate_x) / 3
        
        model = {
            's_learner': s_learner,
            't_models': t_models,
            'x_models': {
                'cate_treated': cate_model_treated,
                'cate_control': cate_model_control
            },
            'cate_s': cate_s,
            'cate_t': cate_t,
            'cate_x': cate_x
        }
        
        return cate_ensemble, model
    
    def _test_heterogeneity(self, cate_estimates, features):
        """Test for presence of heterogeneous effects"""
        
        # Method 1: Variance test
        cate_var = np.var(cate_estimates)
        
        # Bootstrap to get null distribution
        n_boot = 1000
        null_vars = []
        
        for _ in range(n_boot):
            # Permute CATE estimates
            cate_perm = np.random.permutation(cate_estimates)
            null_vars.append(np.var(cate_perm))
        
        p_variance = np.mean(null_vars >= cate_var)
        
        # Method 2: Best linear predictor test
        # Fit model to predict CATE
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # Cross-validated R²
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(rf, features, cate_estimates, cv=5, 
                                  scoring='r2', n_jobs=-1)
        mean_r2 = cv_scores.mean()
        
        # Permutation test for R²
        null_r2s = []
        for _ in range(100):  # Fewer permutations for speed
            cate_perm = np.random.permutation(cate_estimates)
            score = cross_val_score(rf, features, cate_perm, cv=3, 
                                  scoring='r2', n_jobs=-1).mean()
            null_r2s.append(score)
        
        p_r2 = np.mean(null_r2s >= mean_r2)
        
        return {
            'cate_variance': cate_var,
            'p_variance': p_variance,
            'r_squared': mean_r2,
            'p_r_squared': p_r2,
            'significant': p_variance < 0.05 or p_r2 < 0.05,
            'heterogeneity_score': -np.log10(min(p_variance, p_r2) + 1e-10)
        }
    
    def _identify_effect_modifiers(self, cate_estimates, features, feature_names, model):
        """Identify important effect modifiers"""
        
        # Method 1: Variable importance from CATE prediction
        rf_cate = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_cate.fit(features, cate_estimates)
        
        importance_cate = rf_cate.feature_importances_
        
        # Method 2: Variable importance from outcome models (if available)
        importance_outcome = None
        if isinstance(model, dict) and 'outcome_model_treated' in model:
            imp_treated = model['outcome_model_treated'].feature_importances_
            imp_control = model['outcome_model_control'].feature_importances_
            importance_outcome = (imp_treated + imp_control) / 2
        
        # Method 3: Univariate interaction tests
        interaction_scores = []
        
        for j in range(features.shape[1]):
            # Test X_j * Treatment interaction
            X = np.column_stack([
                features[:, j],
                cate_estimates,
                features[:, j] * cate_estimates
            ])
            
            # Correlation between feature and CATE
            corr = np.corrcoef(features[:, j], cate_estimates)[0, 1]
            interaction_scores.append(abs(corr))
        
        interaction_scores = np.array(interaction_scores)
        
        # Combine scores
        if importance_outcome is not None:
            combined_importance = (
                importance_cate + 
                importance_outcome + 
                interaction_scores / interaction_scores.max()
            ) / 3
        else:
            combined_importance = (
                importance_cate + 
                interaction_scores / interaction_scores.max()
            ) / 2
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_cate': importance_cate,
            'importance_outcome': importance_outcome if importance_outcome is not None else 0,
            'interaction_score': interaction_scores,
            'combined_importance': combined_importance
        }).sort_values('combined_importance', ascending=False)
        
        return importance_df
    
    def _analyze_subgroup(self, cate_subgroup, treatment_subgroup, outcome_subgroup):
        """Analyze treatment effects in a specific subgroup"""
        
        # Basic statistics
        stats_dict = {
            'n': len(cate_subgroup),
            'mean_cate': cate_subgroup.mean(),
            'std_cate': cate_subgroup.std(),
            'se_cate': cate_subgroup.std() / np.sqrt(len(cate_subgroup)),
            'ci_lower': np.percentile(cate_subgroup, 2.5),
            'ci_upper': np.percentile(cate_subgroup, 97.5)
        }
        
        # Observed ATE in subgroup
        treated_outcomes = outcome_subgroup[treatment_subgroup == 1]
        control_outcomes = outcome_subgroup[treatment_subgroup == 0]
        
        if len(treated_outcomes) > 0 and len(control_outcomes) > 0:
            stats_dict['observed_ate'] = treated_outcomes.mean() - control_outcomes.mean()
            stats_dict['n_treated'] = len(treated_outcomes)
            stats_dict['n_control'] = len(control_outcomes)
        
        return stats_dict
    
    def _test_interactions(self, treatment, outcome, features, feature_names):
        """Test specific treatment-covariate interactions"""
        
        from sklearn.linear_model import LinearRegression
        
        # Determine if outcome is binary
        unique_outcomes = np.unique(outcome)
        is_binary = (
            outcome.dtype == bool or 
            (len(unique_outcomes) == 2 and 
             set(unique_outcomes).issubset({0, 1, True, False}))
        )
        
        interaction_results = {}
        
        # Test each feature
        for j, fname in enumerate(feature_names):
            # Create interaction term
            X = np.column_stack([
                treatment,
                features[:, j],
                treatment * features[:, j]
            ])
            
            if is_binary:
                # Logistic regression for binary outcomes
                lr = LogisticRegression(penalty=None, max_iter=1000)
            else:
                # Linear regression for continuous outcomes
                lr = LinearRegression()
            
            lr.fit(X, outcome)
            
            # Get interaction coefficient
            interaction_coef = lr.coef_[0, 2] if is_binary else lr.coef_[2]
            
            # Approximate p-value using bootstrap
            n_boot = 200
            boot_coefs = []
            
            for _ in range(n_boot):
                idx = np.random.choice(len(X), len(X), replace=True)
                try:
                    if is_binary:
                        lr_boot = LogisticRegression(penalty=None, max_iter=100)
                    else:
                        lr_boot = LinearRegression()
                    lr_boot.fit(X[idx], outcome[idx])
                    boot_coef = lr_boot.coef_[0, 2] if is_binary else lr_boot.coef_[2]
                    boot_coefs.append(boot_coef)
                except:
                    continue
            
            if len(boot_coefs) > 100:
                se = np.std(boot_coefs)
                z_stat = interaction_coef / (se + 1e-10)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                p_value = 1.0
            
            interaction_results[fname] = {
                'coefficient': interaction_coef,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return interaction_results
    
    def _derive_treatment_rules(self, cate_estimates, features, feature_names):
        """Derive optimal treatment rules"""
        
        # Simple rule: treat if CATE > threshold
        thresholds = [0, np.percentile(cate_estimates, 25), 
                     np.percentile(cate_estimates, 50)]
        
        rules = {}
        
        for threshold in thresholds:
            treat_rule = cate_estimates > threshold
            
            # Characteristics of treated group
            treated_features = features[treat_rule].mean(axis=0)
            control_features = features[~treat_rule].mean(axis=0)
            
            # Find features with largest differences
            feature_diffs = treated_features - control_features
            top_features_idx = np.argsort(np.abs(feature_diffs))[-5:][::-1]
            
            rules[f'threshold_{threshold:.3f}'] = {
                'threshold': threshold,
                'n_treat': treat_rule.sum(),
                'pct_treat': treat_rule.mean() * 100,
                'mean_cate_treated': cate_estimates[treat_rule].mean() if treat_rule.sum() > 0 else 0,
                'mean_cate_control': cate_estimates[~treat_rule].mean() if (~treat_rule).sum() > 0 else 0,
                'top_discriminating_features': [
                    (feature_names[idx], feature_diffs[idx]) 
                    for idx in top_features_idx
                ]
            }
        
        # Tree-based rule
        from sklearn.tree import DecisionTreeClassifier
        
        # Create binary outcome: high benefit vs low benefit
        high_benefit = cate_estimates > np.percentile(cate_estimates, 75)
        
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(features, high_benefit)
        
        rules['tree_based'] = {
            'tree_model': tree,
            'feature_importances': dict(zip(feature_names, tree.feature_importances_))
        }
        
        return rules
    
    def _evaluate_policy(self, cate_estimates, treatment, outcome):
        """Evaluate value of personalized treatment policy"""
        
        # Current policy value (observed)
        current_value = outcome.mean()
        
        # Oracle policy (treat only if CATE > 0)
        oracle_treat = cate_estimates > 0
        
        # Calculate oracle value more carefully
        n = len(outcome)
        oracle_value_components = []
        
        # For those who should be treated under oracle policy
        for i in range(n):
            if oracle_treat[i]:
                # They get the treatment effect
                oracle_value_components.append(cate_estimates[i])
            else:
                # They get baseline outcome (no treatment)
                if treatment[i] == 0:
                    oracle_value_components.append(outcome[i])
                else:
                    # Estimated counterfactual for treated who shouldn't be
                    oracle_value_components.append(outcome[i] - cate_estimates[i])
        
        oracle_value = np.mean(oracle_value_components)
        
        # Random policy (50% treatment)
        random_value = cate_estimates.mean() * 0.5
        
        # Practical policies at different thresholds
        policy_values = {}
        
        for pct in [25, 50, 75]:
            threshold = np.percentile(cate_estimates, pct)
            treat_rule = cate_estimates > threshold
            
            # Expected value under this policy
            policy_value_components = []
            for i in range(n):
                if treat_rule[i]:
                    policy_value_components.append(cate_estimates[i])
                else:
                    if treatment[i] == 0:
                        policy_value_components.append(outcome[i])
                    else:
                        policy_value_components.append(outcome[i] - cate_estimates[i])
            
            policy_value = np.mean(policy_value_components)
            
            # Calculate efficiency safely
            if oracle_value != random_value:
                efficiency = (policy_value - random_value) / (oracle_value - random_value)
            else:
                efficiency = 0
            
            policy_values[f'top_{100-pct}_pct'] = {
                'threshold': threshold,
                'value': policy_value,
                'improvement_vs_current': policy_value - current_value,
                'efficiency': efficiency
            }
        
        return {
            'current_value': current_value,
            'oracle_value': oracle_value,
            'random_value': random_value,
            'policy_values': policy_values,
            'max_improvement': oracle_value - current_value
        }

# =============================================================================
# EXAMPLE ANALYSIS FUNCTION
# =============================================================================

def run_heterogeneity_analysis(treatment, outcome, features, 
                             feature_names=None, subgroups=None,
                             method='causal_forest'):
    """Run complete heterogeneity analysis"""
    
    print_header("HETEROGENEOUS EFFECTS ANALYSIS")
    
    # Initialize analyzer
    analyzer = HeterogeneousEffectsAnalyzer(method=method, n_estimators=500)
    
    # Run analysis
    analyzer.analyze_heterogeneity(
        treatment, outcome, features, 
        feature_names=feature_names,
        subgroups=subgroups,
        test_interactions=True
    )
    
    # Summary
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Mean CATE: {analyzer.results['cate_mean']:.4f}")
    print(f"CATE SD: {analyzer.results['cate_std']:.4f}")
    
    if analyzer.results['heterogeneity_test']['significant']:
        print("✓ Significant heterogeneity detected!")
        print(f"  Variance p-value: {analyzer.results['heterogeneity_test']['p_variance']:.4f}")
        print(f"  R² p-value: {analyzer.results['heterogeneity_test']['p_r_squared']:.4f}")
    else:
        print("✗ No significant heterogeneity detected")
    
    # Top effect modifiers
    print("\nTop 5 effect modifiers:")
    top_modifiers = analyzer.results['modifier_importance'].head()
    for _, row in top_modifiers.iterrows():
        print(f"  {row['feature']}: {row['combined_importance']:.3f}")
    
    # Prepare results for JSON serialization
    results_json = {
        'cate_mean': float(analyzer.results['cate_mean']),
        'cate_std': float(analyzer.results['cate_std']),
        'cate_quantiles': [float(q) for q in analyzer.results['cate_quantiles']],
        'heterogeneity_test': analyzer.results['heterogeneity_test'],
        'modifier_importance': analyzer.results['modifier_importance'].to_dict('records'),
        'subgroup_effects': analyzer.results['subgroup_effects'],
        'interaction_tests': analyzer.results['interaction_tests'],
        'policy_value': {}
    }
    
    # Handle policy_value nested structure
    if 'policy_value' in analyzer.results:
        pv = analyzer.results['policy_value']
        results_json['policy_value'] = {
            'current_value': float(pv['current_value']),
            'oracle_value': float(pv['oracle_value']),
            'random_value': float(pv['random_value']),
            'max_improvement': float(pv['max_improvement']),
            'policy_values': pv['policy_values']
        }
    
    # Handle treatment_rules separately (contains non-serializable tree model)
    if 'treatment_rules' in analyzer.results:
        treatment_rules_json = {}
        for key, rule in analyzer.results['treatment_rules'].items():
            if key == 'tree_based':
                # Skip the tree model itself, keep only feature importances
                treatment_rules_json[key] = {
                    'feature_importances': rule.get('feature_importances', {})
                }
            else:
                treatment_rules_json[key] = rule
        results_json['treatment_rules'] = treatment_rules_json
    
    # Save JSON-serializable results
    save_results(results_json, "heterogeneity_analysis_results.json")
    
    # Also save complete results with models using pickle
    import pickle
    results_pickle_path = os.path.join(Config.OUTPUT_PATH, 'results', 'heterogeneity_analysis_full.pkl')
    with open(results_pickle_path, 'wb') as f:
        pickle.dump(analyzer.results, f)
    print(f"\nFull results with models saved to: {results_pickle_path}")
    
    return analyzer.results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_header("HETEROGENEITY ANALYSIS MODULE")
    
    # Load processed data
    from phase3_01_data_processor import ComprehensiveDataProcessor
    
    processor = ComprehensiveDataProcessor()
    processed_data_path = os.path.join(Config.OUTPUT_PATH, 'results', 'processed_data.pkl')
    
    if os.path.exists(processed_data_path):
        print("\nLoading processed data...")
        analysis_data = processor.load_processed_data()
        
        # Define treatment (AD status)
        treatment = analysis_data['outcomes']['ad_case_primary'].values
        
        # Define outcome (e.g., metabolite change if temporal data exists)
        if analysis_data['temporal'] and 'delta_0_1' in analysis_data['temporal']:
            # Use metabolite change as outcome
            metabolite_changes = analysis_data['temporal']['delta_0_1']
            # Pick first metabolite for demonstration
            outcome = metabolite_changes[:, 0]
            
            print(f"\nAnalyzing heterogeneous effects of AD on metabolite changes")
        else:
            # Use diabetes as outcome
            outcome = analysis_data['outcomes']['has_diabetes_any'].values
            print(f"\nAnalyzing heterogeneous effects of AD on diabetes")
        
        # Features for effect modification
        features = analysis_data['features']['all']
        feature_names = analysis_data['features']['all_names']
        
        # Define subgroups based on demographics
        subgroups = {}
        if 'age_group' in analysis_data['demographics'].columns:
            for age_grp in analysis_data['demographics']['age_group'].unique():
                mask = analysis_data['demographics']['age_group'] == age_grp
                subgroups[f'age_{age_grp}'] = mask.values
        
        if 'sex' in analysis_data['demographics'].columns:
            for sex in analysis_data['demographics']['sex'].unique():
                mask = analysis_data['demographics']['sex'] == sex
                subgroups[f'sex_{sex}'] = mask.values
        
        print(f"\nData summary:")
        print(f"  Treatment (AD): {treatment.sum()} cases, {len(treatment)-treatment.sum()} controls")
        print(f"  Features: {features.shape[1]}")
        print(f"  Subgroups: {list(subgroups.keys())}")
        
        # Run analysis
        results = run_heterogeneity_analysis(
            treatment, outcome, features,
            feature_names=feature_names,
            subgroups=subgroups,
            method='causal_forest'
        )
        
        print(f"\nAnalysis complete! Results saved to: {Config.OUTPUT_PATH}/results/")
    else:
        print(f"\nERROR: Processed data not found. Run phase3_01_data_processor.py first")
