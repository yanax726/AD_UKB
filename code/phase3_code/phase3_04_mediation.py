#!/usr/bin/env python3
"""
phase3_04_mediation.py - High-Dimensional Mediation Analysis
"""

# Import configuration
try:
    from phase3_00_config import *
except ImportError:
    print("ERROR: Please run phase3_00_config.py first!")
    raise

# =============================================================================
# HIGH-DIMENSIONAL MEDIATION ANALYSIS
# =============================================================================

class HighDimensionalMediation:
    """Advanced mediation analysis for metabolomics with multiple testing correction"""
    
    def __init__(self, method='hdma', fdr_threshold=0.05, n_bootstrap=1000):
        self.method = method
        self.fdr_threshold = fdr_threshold
        self.n_bootstrap = n_bootstrap
        self.results = {}
        
    def fit(self, exposure, mediators, outcome, covariates=None, 
            mediator_names=None, pathway_info=None):
        """
        Perform high-dimensional mediation analysis
        
        Parameters:
        -----------
        exposure: array-like, shape (n_samples,)
        mediators: array-like, shape (n_samples, n_mediators)
        outcome: array-like, shape (n_samples,)
        covariates: array-like, shape (n_samples, n_covariates), optional
        mediator_names: list of mediator names, optional
        pathway_info: dict mapping mediators to pathways, optional
        """
        
        n_samples, n_mediators = mediators.shape
        print(f"\n  Testing {n_mediators} potential mediators...")
        
        # Standardize for stability
        scaler = StandardScaler()
        mediators_scaled = scaler.fit_transform(mediators)
        
        if mediator_names is None:
            mediator_names = [f'M{i+1}' for i in range(n_mediators)]
        
        # Step 1: Initial screening
        print("  Step 1: Initial screening...")
        screening_results = self._initial_screening(
            exposure, mediators_scaled, outcome, covariates
        )
        
        # Step 2: Joint significance testing
        print("  Step 2: Joint significance testing...")
        joint_results = self._joint_significance_test(
            exposure, mediators_scaled, outcome, covariates,
            screening_results
        )
        
        # Step 3: Estimate mediation effects for significant mediators
        print("  Step 3: Estimating mediation effects...")
        mediation_effects = []
        
        significant_idx = np.where(joint_results['significant'])[0]
        print(f"  Found {len(significant_idx)} significant mediators after FDR correction")
        
        for idx in tqdm(significant_idx, desc="  Calculating effects"):
            effect = self._estimate_single_mediation(
                exposure, mediators_scaled[:, idx], outcome,
                covariates, idx, mediator_names[idx]
            )
            mediation_effects.append(effect)
        
        # Step 4: Pathway enrichment if pathway info provided
        pathway_enrichment = None
        if pathway_info is not None and len(significant_idx) > 0:
            print("  Step 4: Pathway enrichment analysis...")
            pathway_enrichment = self._pathway_enrichment(
                significant_idx, mediator_names, pathway_info
            )
        
        # Step 5: Network analysis of mediators
        if len(significant_idx) > 5:
            print("  Step 5: Mediator network analysis...")
            network_results = self._mediator_network_analysis(
                mediators_scaled[:, significant_idx],
                [mediator_names[i] for i in significant_idx]
            )
        else:
            network_results = None
        
        # Compile results
        self.results = {
            'screening': screening_results,
            'joint_testing': joint_results,
            'mediation_effects': mediation_effects,
            'n_significant': len(significant_idx),
            'significant_mediators': [mediator_names[i] for i in significant_idx],
            'pathway_enrichment': pathway_enrichment,
            'network_analysis': network_results,
            'total_effect': self._calculate_total_effect(exposure, outcome, covariates),
            'proportion_mediated_total': sum(e['proportion_mediated'] for e in mediation_effects)
        }
        
        return self
    
    def _initial_screening(self, exposure, mediators, outcome, covariates):
        """Screen mediators using exposure-mediator associations"""
        
        n_mediators = mediators.shape[1]
        alpha_pvalues = np.zeros(n_mediators)
        alpha_effects = np.zeros(n_mediators)
        
        # Test exposure -> mediator associations
        for j in range(n_mediators):
            # Regression M ~ X + C
            X = self._prepare_design_matrix(exposure, covariates)
            y = mediators[:, j]
            
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha_effects[j] = beta[1]  # Exposure effect
            
            # Calculate p-value
            residuals = y - X @ beta
            se = np.sqrt(np.sum(residuals**2) / (len(y) - X.shape[1]))
            se_beta = se / np.sqrt(np.sum((X[:, 1] - X[:, 1].mean())**2))
            
            t_stat = alpha_effects[j] / (se_beta + 1e-10)
            alpha_pvalues[j] = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - X.shape[1]))
        
        return {
            'alpha_effects': alpha_effects,
            'alpha_pvalues': alpha_pvalues,
            'screened': alpha_pvalues < 0.1  # Liberal screening threshold
        }
    
    def _joint_significance_test(self, exposure, mediators, outcome, covariates, screening):
        """Joint significance test with FDR correction"""
        
        n_mediators = mediators.shape[1]
        beta_pvalues = np.zeros(n_mediators)
        beta_effects = np.zeros(n_mediators)
        
        # Only test screened mediators
        test_idx = np.where(screening['screened'])[0]
        
        # Test mediator -> outcome associations
        for j in test_idx:
            # For binary outcome, use logistic regression
            X = self._prepare_design_matrix(
                np.column_stack([mediators[:, j], exposure]), 
                covariates
            )
            
            # Logistic regression
            lr = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
            lr.fit(X[:, 1:], outcome)  # Exclude intercept for sklearn
            
            beta_effects[j] = lr.coef_[0, 0]  # Mediator effect
            
            # Approximate p-value
            # Calculate standard errors using inverse Fisher information
            probs = lr.predict_proba(X[:, 1:])[:, 1]
            W = np.diag(probs * (1 - probs))
            try:
                cov_matrix = np.linalg.inv(X.T @ W @ X)
                se = np.sqrt(np.diag(cov_matrix)[1])  # SE for mediator
                z_stat = beta_effects[j] / (se + 1e-10)
                beta_pvalues[j] = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            except:
                beta_pvalues[j] = 1.0
        
        # Joint significance: max of alpha and beta p-values
        joint_pvalues = np.maximum(screening['alpha_pvalues'], beta_pvalues)
        joint_pvalues[~screening['screened']] = 1.0  # Set non-screened to 1
        
        # FDR correction
        reject, pvals_adjusted, _, _ = multipletests(
            joint_pvalues[screening['screened']], 
            alpha=self.fdr_threshold, 
            method='fdr_bh'
        )
        
        # Update full arrays
        fdr_adjusted = np.ones(n_mediators)
        fdr_adjusted[screening['screened']] = pvals_adjusted
        
        significant = np.zeros(n_mediators, dtype=bool)
        significant[screening['screened']] = reject
        
        return {
            'beta_effects': beta_effects,
            'beta_pvalues': beta_pvalues,
            'joint_pvalues': joint_pvalues,
            'fdr_pvalues': fdr_adjusted,
            'significant': significant
        }
    
    def _estimate_single_mediation(self, exposure, mediator, outcome, covariates, 
                                 idx, mediator_name):
        """Estimate mediation effect for a single mediator with bootstrap CI"""
        
        # Point estimates
        X_alpha = self._prepare_design_matrix(exposure, covariates)
        X_beta = self._prepare_design_matrix(
            np.column_stack([mediator, exposure]), 
            covariates
        )
        X_total = self._prepare_design_matrix(exposure, covariates)
        
        # Alpha path (exposure -> mediator)
        lr_alpha = LinearRegression()
        lr_alpha.fit(X_alpha, mediator)
        alpha = lr_alpha.coef_[1]  # Exposure effect
        
        # Beta path (mediator -> outcome controlling for exposure)
        lr_beta = LogisticRegression(penalty=None, max_iter=1000)
        lr_beta.fit(X_beta[:, 1:], outcome)
        beta = lr_beta.coef_[0, 0]  # Mediator effect
        
        # Direct effect
        direct = lr_beta.coef_[0, 1]  # Exposure effect controlling for mediator
        
        # Total effect
        lr_total = LogisticRegression(penalty=None, max_iter=1000)
        lr_total.fit(X_total[:, 1:], outcome)
        total = lr_total.coef_[0, 0]
        
        # Indirect effect
        indirect = alpha * beta
        
        # Proportion mediated
        prop_mediated = indirect / (total + 1e-10) if abs(total) > 1e-10 else 0
        
        # Bootstrap confidence intervals
        if self.n_bootstrap > 0:
            indirect_boot = []
            prop_boot = []
            
            n_samples = len(exposure)
            
            for _ in range(self.n_bootstrap):
                idx_boot = np.random.choice(n_samples, n_samples, replace=True)
                
                try:
                    # Refit on bootstrap sample
                    lr_a = LinearRegression()
                    lr_a.fit(X_alpha[idx_boot], mediator[idx_boot])
                    alpha_b = lr_a.coef_[1]
                    
                    lr_b = LogisticRegression(penalty=None, max_iter=100)
                    lr_b.fit(X_beta[idx_boot, 1:], outcome[idx_boot])
                    beta_b = lr_b.coef_[0, 0]
                    
                    lr_t = LogisticRegression(penalty=None, max_iter=100)
                    lr_t.fit(X_total[idx_boot, 1:], outcome[idx_boot])
                    total_b = lr_t.coef_[0, 0]
                    
                    indirect_b = alpha_b * beta_b
                    prop_b = indirect_b / (total_b + 1e-10) if abs(total_b) > 1e-10 else 0
                    
                    indirect_boot.append(indirect_b)
                    prop_boot.append(prop_b)
                except:
                    continue
            
            if len(indirect_boot) > self.n_bootstrap * 0.8:
                ci_lower = np.percentile(indirect_boot, 2.5)
                ci_upper = np.percentile(indirect_boot, 97.5)
                se = np.std(indirect_boot)
                
                prop_ci_lower = np.percentile(prop_boot, 2.5)
                prop_ci_upper = np.percentile(prop_boot, 97.5)
            else:
                ci_lower = ci_upper = se = np.nan
                prop_ci_lower = prop_ci_upper = np.nan
        else:
            ci_lower = ci_upper = se = np.nan
            prop_ci_lower = prop_ci_upper = np.nan
        
        return {
            'mediator_idx': idx,
            'mediator_name': mediator_name,
            'alpha': alpha,
            'beta': beta,
            'indirect_effect': indirect,
            'direct_effect': direct,
            'total_effect': total,
            'proportion_mediated': prop_mediated,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'se': se,
            'prop_ci_lower': prop_ci_lower,
            'prop_ci_upper': prop_ci_upper,
            'significant': not (ci_lower <= 0 <= ci_upper) if not np.isnan(ci_lower) else False
        }
    
    def _prepare_design_matrix(self, exposure, covariates):
        """Prepare design matrix with intercept and covariates"""
        
        n_samples = len(exposure) if exposure.ndim == 1 else exposure.shape[0]
        
        # Start with intercept
        X = [np.ones(n_samples)]
        
        # Add exposure
        if exposure.ndim == 1:
            X.append(exposure)
        else:
            X.extend([exposure[:, i] for i in range(exposure.shape[1])])
        
        # Add covariates
        if covariates is not None:
            if covariates.ndim == 1:
                X.append(covariates)
            else:
                X.extend([covariates[:, i] for i in range(covariates.shape[1])])
        
        return np.column_stack(X)
    
    def _calculate_total_effect(self, exposure, outcome, covariates):
        """Calculate total effect of exposure on outcome"""
        
        X = self._prepare_design_matrix(exposure, covariates)
        
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(X[:, 1:], outcome)
        
        return lr.coef_[0, 0]
    
    def _pathway_enrichment(self, significant_idx, mediator_names, pathway_info):
        """Perform pathway enrichment analysis"""
        
        # Count significant mediators per pathway
        pathway_counts = defaultdict(int)
        pathway_totals = defaultdict(int)
        
        for idx, name in enumerate(mediator_names):
            if name in pathway_info:
                pathway = pathway_info[name]
                pathway_totals[pathway] += 1
                
                if idx in significant_idx:
                    pathway_counts[pathway] += 1
        
        # Hypergeometric test for each pathway
        enrichment_results = []
        
        n_total = len(mediator_names)
        n_sig_total = len(significant_idx)
        
        for pathway, count in pathway_counts.items():
            n_pathway = pathway_totals[pathway]
            
            # Hypergeometric test
            pval = stats.hypergeom.sf(
                count - 1, n_total, n_sig_total, n_pathway
            )
            
            # Enrichment ratio
            expected = n_sig_total * n_pathway / n_total
            enrichment = count / expected if expected > 0 else 0
            
            enrichment_results.append({
                'pathway': pathway,
                'n_significant': count,
                'n_total': n_pathway,
                'enrichment_ratio': enrichment,
                'p_value': pval
            })
        
        # Sort by p-value
        enrichment_results.sort(key=lambda x: x['p_value'])
        
        # FDR correction
        if enrichment_results:
            pvals = [r['p_value'] for r in enrichment_results]
            _, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')
            
            for i, result in enumerate(enrichment_results):
                result['fdr_p_value'] = pvals_adj[i]
                result['significant'] = pvals_adj[i] < self.fdr_threshold
        
        return enrichment_results
    
    def _mediator_network_analysis(self, mediators, mediator_names):
        """Analyze network structure of significant mediators"""
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(mediators.T)
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for name in mediator_names:
            G.add_node(name)
        
        # Add edges for strong correlations
        threshold = 0.5
        for i in range(len(mediator_names)):
            for j in range(i + 1, len(mediator_names)):
                if abs(corr_matrix[i, j]) > threshold:
                    G.add_edge(
                        mediator_names[i], 
                        mediator_names[j],
                        weight=corr_matrix[i, j]
                    )
        
        # Calculate network metrics
        metrics = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'n_components': nx.number_connected_components(G),
            'clustering_coefficient': nx.average_clustering(G),
            'correlation_matrix': corr_matrix,
            'graph': G
        }
        
        # Identify communities
        if G.number_of_edges() > 0:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G))
            metrics['n_communities'] = len(communities)
            metrics['communities'] = communities
        
        return metrics

# =============================================================================
# EXAMPLE ANALYSIS FUNCTION
# =============================================================================

def run_mediation_analysis(exposure, mediators, outcome, covariates=None,
                         mediator_names=None, method='hdma', fdr=0.05):
    """Run complete mediation analysis"""
    
    print_header("HIGH-DIMENSIONAL MEDIATION ANALYSIS")
    
    # Initialize model
    hdma = HighDimensionalMediation(method=method, fdr_threshold=fdr)
    
    # Fit model
    hdma.fit(exposure, mediators, outcome, covariates, mediator_names)
    
    # Summary
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Total mediators tested: {mediators.shape[1]}")
    print(f"Significant mediators: {hdma.results['n_significant']}")
    print(f"Total proportion mediated: {hdma.results['proportion_mediated_total']:.1%}")
    
    # Top mediators
    if hdma.results['mediation_effects']:
        print("\nTop 10 mediators:")
        sorted_effects = sorted(hdma.results['mediation_effects'], 
                              key=lambda x: abs(x['indirect_effect']), 
                              reverse=True)
        
        for i, effect in enumerate(sorted_effects[:10]):
            print(f"{i+1}. {effect['mediator_name']}: "
                  f"IE={effect['indirect_effect']:.4f} "
                  f"(PM={effect['proportion_mediated']:.1%})")
    
    # Save results
    save_results(hdma.results, "mediation_analysis_results.json")
    
    return hdma.results
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_header("MEDIATION ANALYSIS MODULE")
    
    # Load processed data
    from phase3_01_data_processor import ComprehensiveDataProcessor
    
    processor = ComprehensiveDataProcessor()
    processed_data_path = os.path.join(Config.OUTPUT_PATH, 'results', 'processed_data.pkl')
    
    if os.path.exists(processed_data_path):
        print("\nLoading processed data...")
        analysis_data = processor.load_processed_data()
        
        # Use real metabolomics data for mediation
        if 'metabolomics' in analysis_data['features']:
            print("\nUsing real metabolomics data for mediation analysis")
            
            # Exposure: AD status
            exposure = analysis_data['outcomes']['ad_case_primary'].values  # AD status
            mediators = analysis_data['features']['metabolomics']  # Metabolites as mediators
            outcome = analysis_data['outcomes']['has_diabetes_any'].values  # Disease outcome
            
            # Outcome: metabolic disease (e.g., diabetes)
            if 'has_diabetes_any' in analysis_data['outcomes'].columns:
                outcome = analysis_data['outcomes']['has_diabetes_any'].values
                
                # Covariates
                covariates = analysis_data['demographics'][['age_baseline']].values
                if 'sex' in analysis_data['demographics'].columns:
                    sex_encoded = pd.get_dummies(analysis_data['demographics']['sex'])
                    covariates = np.hstack([covariates, sex_encoded.values])
                
                print(f"\nAnalyzing mediation pathways:")
                print(f"  Exposure: AD (n={exposure.sum()} cases)")
                print(f"  Mediators: {mediators.shape[1]} metabolites")
                print(f"  Outcome: Diabetes (n={outcome.sum()} cases)")
                print(f"  Samples: {len(exposure)}")
                
                # Run mediation analysis
                results = run_mediation_analysis(
                    exposure, mediators, outcome, 
                    covariates=covariates,
                    mediator_names=mediator_names,
                    method='hdma',
                    fdr=0.05
                )
                
                print(f"\nAnalysis complete! Results saved to: {Config.OUTPUT_PATH}/results/")
            else:
                print("\nERROR: Diabetes outcome not found in data")
        else:
            print("\nERROR: Metabolomics data not found")
    else:
        print(f"\nERROR: Processed data not found. Run phase3_01_data_processor.py first")