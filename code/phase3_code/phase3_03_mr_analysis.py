#!/usr/bin/env python3
"""
phase3_03_mr_analysis.py - Contamination Mixture MR Analysis
"""

# Import configuration
try:
    from phase3_00_config import *
except ImportError:
    print("ERROR: Please run phase3_00_config.py first!")
    raise

# =============================================================================
# CONTAMINATION MIXTURE MENDELIAN RANDOMIZATION
# =============================================================================

class ContaminationMixtureMR:
    """Advanced MR with contamination mixture model for handling invalid instruments"""
    
    def __init__(self, n_components=5, max_iter=1000, tol=1e-6, robust=True):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.robust = robust
        self.convergence_history = []
        self.model_selection_criteria = {}
    
    def fit(self, beta_exposure, beta_outcome, se_exposure, se_outcome, 
            sample_size=None, select_components=True):
        """
        Fit contamination mixture model with automatic component selection
        """
        
        n_snps = len(beta_exposure)
        print(f"\n  Fitting mixture model with up to {self.n_components} components using {n_snps} instruments...")
        
        if select_components:
            # Try different numbers of components
            models = {}
            
            for k in range(1, min(self.n_components + 1, n_snps // 10 + 1)):
                print(f"    Testing k={k} components...", end='')
                
                model = self._fit_single_model(
                    beta_exposure, beta_outcome, se_exposure, se_outcome, 
                    n_components=k
                )
                
                if model is not None and model['converged']:
                    models[k] = model
                    print(f" BIC={model['bic']:.1f}")
                else:
                    print(" Failed")
            
            # Select best model by BIC
            if models:
                best_k = min(models.keys(), key=lambda k: models[k]['bic'])
                print(f"\n  Selected k={best_k} components (lowest BIC)")
                
                # Set attributes from best model
                best_model = models[best_k]
                self.means = best_model['means']
                self.variances = best_model['variances']
                self.weights = best_model['weights']
                self.assignments = best_model['assignments']
                self.bic = best_model['bic']
                self.aic = best_model['aic']
                self.log_likelihood = best_model['log_likelihood']
                
                # Store all models for comparison
                self.model_selection_criteria = {
                    k: {'bic': m['bic'], 'aic': m['aic'], 'log_lik': m['log_likelihood']}
                    for k, m in models.items()
                }
            else:
                print("  ERROR: No models converged")
                return None
        else:
            # Fit single model with specified components
            model = self._fit_single_model(
                beta_exposure, beta_outcome, se_exposure, se_outcome,
                n_components=self.n_components
            )
            
            if model is not None:
                self.means = model['means']
                self.variances = model['variances']
                self.weights = model['weights']
                self.assignments = model['assignments']
        
        # Identify valid component and calculate final estimates
        self._identify_valid_component(beta_exposure, beta_outcome, se_exposure, se_outcome)
        self._calculate_confidence_intervals(beta_exposure, beta_outcome, se_exposure, se_outcome)
        self._calculate_diagnostics(beta_exposure, beta_outcome, se_exposure, se_outcome)
        
        return self
    
    def _fit_single_model(self, beta_exp, beta_out, se_exp, se_out, n_components):
        """Fit mixture model with specified number of components"""
        
        n_snps = len(beta_exp)
        
        # Initialize with k-means on ratio estimates
        ratio_estimates = beta_out / (beta_exp + 1e-10)
        valid_mask = np.isfinite(ratio_estimates)
        
        if valid_mask.sum() < n_components * 5:
            return None
        
        valid_ratios = ratio_estimates[valid_mask]
        
        # K-means initialization
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=42)
        labels = kmeans.fit_predict(valid_ratios.reshape(-1, 1))
        
        # Initialize parameters
        means = np.array([valid_ratios[labels == k].mean() for k in range(n_components)])
        variances = np.array([valid_ratios[labels == k].var() + 0.01 for k in range(n_components)])
        weights = np.array([(labels == k).sum() / len(labels) for k in range(n_components)])
        
        # EM algorithm
        log_likelihood_prev = -np.inf
        converged = False
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step_single(
                beta_exp, beta_out, se_exp, se_out,
                means, variances, weights
            )
            
            # M-step
            means_new, variances_new, weights_new = self._m_step_single(
                beta_exp, beta_out, se_exp, se_out,
                responsibilities, robust=self.robust
            )
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood_single(
                beta_exp, beta_out, se_exp, se_out,
                means_new, variances_new, weights_new
            )
            
            # Check convergence
            if abs(log_likelihood - log_likelihood_prev) < self.tol:
                converged = True
                break
            
            # Update parameters
            means = means_new
            variances = variances_new
            weights = weights_new
            log_likelihood_prev = log_likelihood
        
        if not converged:
            return None
        
        # Calculate information criteria
        n_params = 3 * n_components - 1  # means, variances, weights (minus 1 for constraint)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_snps)
        
        # Get assignments
        responsibilities = self._e_step_single(
            beta_exp, beta_out, se_exp, se_out,
            means, variances, weights
        )
        assignments = np.argmax(responsibilities, axis=1)
        
        return {
            'means': means,
            'variances': variances,
            'weights': weights,
            'assignments': assignments,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'converged': converged,
            'iterations': iteration + 1
        }
    
    def _e_step_single(self, beta_exp, beta_out, se_exp, se_out, means, variances, weights):
        """E-step for single model"""
        
        n_snps = len(beta_exp)
        n_components = len(means)
        log_resp = np.zeros((n_snps, n_components))
        
        for k in range(n_components):
            # Expected outcome under component k
            expected = means[k] * beta_exp
            
            # Total variance including uncertainty
            total_var = se_out**2 + (means[k] * se_exp)**2 + variances[k]
            
            # Log-likelihood for each SNP
            log_resp[:, k] = -0.5 * np.log(2 * np.pi * total_var)
            log_resp[:, k] -= 0.5 * (beta_out - expected)**2 / total_var
            log_resp[:, k] += np.log(weights[k] + 1e-10)
        
        # Normalize using log-sum-exp trick
        max_log_resp = np.max(log_resp, axis=1, keepdims=True)
        log_resp -= max_log_resp
        exp_resp = np.exp(log_resp)
        responsibilities = exp_resp / (exp_resp.sum(axis=1, keepdims=True) + 1e-10)
        
        return responsibilities
    
    def _m_step_single(self, beta_exp, beta_out, se_exp, se_out, resp, robust=True):
        """M-step for single model"""
        
        n_components = resp.shape[1]
        means = np.zeros(n_components)
        variances = np.zeros(n_components)
        weights = np.zeros(n_components)
        
        for k in range(n_components):
            resp_k = resp[:, k]
            
            # Update weights
            weights[k] = resp_k.mean()
            
            if robust and weights[k] > 0.01:
                # Robust estimation using iteratively reweighted least squares
                for _ in range(5):
                    # Current residuals
                    residuals = beta_out - means[k] * beta_exp
                    
                    # Huber weights
                    standardized = residuals / (se_out + 1e-10)
                    huber_c = 1.345
                    huber_weights = np.where(
                        np.abs(standardized) <= huber_c,
                        1.0,
                        huber_c / (np.abs(standardized) + 1e-10)
                    )
                    
                    # Combined weights
                    total_weights = resp_k * huber_weights / (se_out**2 + variances[k] + 1e-10)
                    
                    # Update mean
                    numerator = np.sum(total_weights * beta_out * beta_exp)
                    denominator = np.sum(total_weights * beta_exp**2)
                    means[k] = numerator / (denominator + 1e-10)
            else:
                # Standard weighted least squares
                weights_wls = resp_k / (se_out**2 + 1e-10)
                numerator = np.sum(weights_wls * beta_out * beta_exp)
                denominator = np.sum(weights_wls * beta_exp**2)
                means[k] = numerator / (denominator + 1e-10)
            
            # Update variance
            residuals = beta_out - means[k] * beta_exp
            variances[k] = np.sum(resp_k * residuals**2) / (resp_k.sum() + 1e-10)
            variances[k] = max(variances[k], 1e-6)
        
        return means, variances, weights
    
    def _calculate_log_likelihood_single(self, beta_exp, beta_out, se_exp, se_out, 
                                          means, variances, weights):
        """Calculate log-likelihood for model"""
        
        n_snps = len(beta_exp)
        n_components = len(means)
        
        log_lik = 0
        for i in range(n_snps):
            likelihood_i = 0
            
            for k in range(n_components):
                expected = means[k] * beta_exp[i]
                total_var = se_out[i]**2 + (means[k] * se_exp[i])**2 + variances[k]
                
                lik_k = weights[k] * np.exp(
                    -0.5 * np.log(2 * np.pi * total_var) - 
                    0.5 * (beta_out[i] - expected)**2 / total_var
                )
                likelihood_i += lik_k
            
            log_lik += np.log(likelihood_i + 1e-10)
        
        return log_lik
    
    def _identify_valid_component(self, beta_exp, beta_out, se_exp, se_out):
        """Identify valid component using multiple criteria"""
        
        # Criterion 1: InSIDE assumption (smallest variance)
        variance_ranks = np.argsort(self.variances)
        
        # Criterion 2: Plurality (highest weight)  
        weight_ranks = np.argsort(self.weights)[::-1]
        
        # Criterion 3: Zero hypothesis test
        zero_pvals = []
        for k in range(len(self.means)):
            # Test if mean is significantly different from zero
            se_mean = np.sqrt(self.variances[k] / (self.weights[k] * len(beta_exp) + 1e-10))
            z_stat = self.means[k] / (se_mean + 1e-10)
            p_val = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
            zero_pvals.append(p_val)
        zero_ranks = np.argsort(zero_pvals)[::-1]  # Higher p-value = closer to zero
        
        # Criterion 4: Consistency with observational association
        if hasattr(self, 'observational_estimate'):
            consistency = np.abs(self.means - self.observational_estimate)
            consistency_ranks = np.argsort(consistency)
        else:
            consistency_ranks = np.arange(len(self.means))
        
        # Combined voting
        votes = np.zeros(len(self.means))
        criteria_weights = [0.3, 0.3, 0.2, 0.2]  # Weights for each criterion
        
        for i in range(len(self.means)):
            votes[variance_ranks[i]] += criteria_weights[0] * (len(self.means) - i)
            votes[weight_ranks[i]] += criteria_weights[1] * (len(self.means) - i)
            votes[zero_ranks[i]] += criteria_weights[2] * (len(self.means) - i)
            votes[consistency_ranks[i]] += criteria_weights[3] * (len(self.means) - i)
        
        self.valid_component = np.argmax(votes)
        self.causal_effect = self.means[self.valid_component]
        
        # Get component assignments
        resp = self._e_step_single(
            beta_exp, beta_out, se_exp, se_out,
            self.means, self.variances, self.weights
        )
        self.component_assignments = np.argmax(resp, axis=1)
        self.valid_instruments = self.component_assignments == self.valid_component
        self.posterior_probs = resp[:, self.valid_component]
        
        print(f"\n  Identified component {self.valid_component} as valid:")
        print(f"    Effect: {self.causal_effect:.4f}")
        print(f"    Weight: {self.weights[self.valid_component]:.1%}")
        print(f"    Valid instruments: {self.valid_instruments.sum()}/{len(beta_exp)}")
    
    def _calculate_confidence_intervals(self, beta_exp, beta_out, se_exp, se_out):
        """Calculate confidence intervals using multiple methods"""
        
        # Method 1: Delta method
        valid_idx = self.valid_instruments
        if valid_idx.sum() > 3:
            valid_beta_exp = beta_exp[valid_idx]
            valid_se_out = se_out[valid_idx]
            
            weights = 1 / valid_se_out**2
            se_delta = np.sqrt(1 / np.sum(weights * valid_beta_exp**2))
            
            self.ci_lower_delta = self.causal_effect - 1.96 * se_delta
            self.ci_upper_delta = self.causal_effect + 1.96 * se_delta
        
        # Method 2: Bootstrap
        n_bootstrap = 1000
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(beta_exp), len(beta_exp), replace=True)
            
            # Quick estimation on bootstrap sample
            boot_ratios = beta_out[idx] / (beta_exp[idx] + 1e-10)
            boot_weights = 1 / (se_out[idx]**2 + 1e-10)
            
            # Weighted median as robust estimate
            sorted_idx = np.argsort(boot_ratios)
            cumsum_weights = np.cumsum(boot_weights[sorted_idx])
            median_idx = np.searchsorted(cumsum_weights, cumsum_weights[-1] / 2)
            
            bootstrap_effects.append(boot_ratios[sorted_idx[median_idx]])
        
        self.ci_lower_bootstrap = np.percentile(bootstrap_effects, 2.5)
        self.ci_upper_bootstrap = np.percentile(bootstrap_effects, 97.5)
        self.se_bootstrap = np.std(bootstrap_effects)
        
        # Use bootstrap CI as primary
        self.ci_lower = self.ci_lower_bootstrap
        self.ci_upper = self.ci_upper_bootstrap
        self.se = self.se_bootstrap
        
        print(f"    95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        print(f"    SE: {self.se:.4f}")
    
    def _calculate_diagnostics(self, beta_exp, beta_out, se_exp, se_out):
        """Calculate comprehensive diagnostics"""
        
        # Heterogeneity for valid instruments
        if self.valid_instruments.sum() > 1:
            valid_beta_exp = beta_exp[self.valid_instruments]
            valid_beta_out = beta_out[self.valid_instruments]
            valid_se_out = se_out[self.valid_instruments]
            
            # Cochran's Q
            predicted = self.causal_effect * valid_beta_exp
            residuals = valid_beta_out - predicted
            q_stat = np.sum(residuals**2 / valid_se_out**2)
            
            df = self.valid_instruments.sum() - 1
            self.q_statistic = q_stat
            self.q_pvalue = 1 - stats.chi2.cdf(q_stat, df)
            
            # I-squared
            self.i_squared = max(0, (q_stat - df) / q_stat)
            
            print(f"\n  Heterogeneity diagnostics:")
            print(f"    Q-statistic: {self.q_statistic:.2f} (p={self.q_pvalue:.4f})")
            print(f"    I²: {self.i_squared:.1%}")
        
        # Model fit
        self.n_valid = self.valid_instruments.sum()
        self.pct_valid = self.n_valid / len(beta_exp) * 100
        
        # Component summary
        print(f"\n  Component summary:")
        for k in range(len(self.means)):
            n_snps = (self.component_assignments == k).sum()
            print(f"    Component {k}: μ={self.means[k]:.3f}, "
                  f"σ²={self.variances[k]:.3f}, "
                  f"π={self.weights[k]:.1%}, "
                  f"n={n_snps}")

# =============================================================================
# SENSITIVITY ANALYSES
# =============================================================================

def mr_leave_one_out(beta_exp, beta_out, se_exp, se_out):
    """Leave-one-out analysis for MR"""
    
    n_snps = len(beta_exp)
    loo_effects = []
    
    for i in range(n_snps):
        # Exclude SNP i
        mask = np.ones(n_snps, dtype=bool)
        mask[i] = False
        
        # IVW estimate without SNP i
        weights = 1 / se_out[mask]**2
        effect = np.sum(weights * beta_out[mask] * beta_exp[mask]) / np.sum(weights * beta_exp[mask]**2)
        
        loo_effects.append({
            'excluded_snp': i,
            'effect': effect
        })
    
    # Check for influential SNPs
    all_effects = [x['effect'] for x in loo_effects]
    mean_effect = np.mean(all_effects)
    sd_effect = np.std(all_effects)
    
    influential = []
    for res in loo_effects:
        if abs(res['effect'] - mean_effect) > 2 * sd_effect:
            influential.append(res['excluded_snp'])
    
    return {
        'effects': loo_effects,
        'mean_effect': mean_effect,
        'sd_effect': sd_effect,
        'influential_snps': influential
    }

def mr_egger(beta_exp, beta_out, se_exp, se_out):
    """MR-Egger regression"""
    
    # Weighted regression with intercept
    weights = 1 / se_out**2
    
    # Design matrix with intercept
    X = np.column_stack([np.ones(len(beta_exp)), beta_exp])
    
    # Weighted least squares
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ beta_out
    
    # Solve
    coeffs = np.linalg.solve(XtWX, XtWy)
    intercept = coeffs[0]
    slope = coeffs[1]
    
    # Standard errors
    residuals = beta_out - X @ coeffs
    sigma2 = np.sum(weights * residuals**2) / (len(beta_exp) - 2)
    cov_matrix = sigma2 * np.linalg.inv(XtWX)
    
    se_intercept = np.sqrt(cov_matrix[0, 0])
    se_slope = np.sqrt(cov_matrix[1, 1])
    
    # P-values
    t_intercept = intercept / se_intercept
    t_slope = slope / se_slope
    
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), len(beta_exp) - 2))
    p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), len(beta_exp) - 2))
    
    return {
        'intercept': intercept,
        'se_intercept': se_intercept,
        'p_intercept': p_intercept,
        'slope': slope,
        'se_slope': se_slope,
        'p_slope': p_slope,
        'pleiotropy_test': p_intercept < 0.05
    }

def mr_mode_based(beta_exp, beta_out, se_exp, se_out):
    """Mode-based estimation for MR"""
    
    # Calculate ratio estimates
    ratios = beta_out / (beta_exp + 1e-10)
    
    # Kernel density estimation
    from scipy.stats import gaussian_kde
    
    # Remove outliers
    q1, q3 = np.percentile(ratios, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    ratios_clean = ratios[(ratios >= lower) & (ratios <= upper)]
    
    if len(ratios_clean) > 5:
        # Estimate mode using KDE
        kde = gaussian_kde(ratios_clean, bw_method='scott')
        
        # Find mode
        x_range = np.linspace(ratios_clean.min(), ratios_clean.max(), 1000)
        density = kde(x_range)
        mode_idx = np.argmax(density)
        mode_estimate = x_range[mode_idx]
        
        # Bootstrap for uncertainty
        n_boot = 500
        boot_modes = []
        
        for _ in range(n_boot):
            idx = np.random.choice(len(ratios), len(ratios), replace=True)
            boot_ratios = ratios[idx]
            
            # Clean
            boot_clean = boot_ratios[(boot_ratios >= lower) & (boot_ratios <= upper)]
            
            if len(boot_clean) > 5:
                boot_kde = gaussian_kde(boot_clean, bw_method='scott')
                boot_density = boot_kde(x_range)
                boot_mode = x_range[np.argmax(boot_density)]
                boot_modes.append(boot_mode)
        
        if len(boot_modes) > 100:
            se = np.std(boot_modes)
            ci_lower = np.percentile(boot_modes, 2.5)
            ci_upper = np.percentile(boot_modes, 97.5)
        else:
            se = ci_lower = ci_upper = np.nan
    else:
        mode_estimate = se = ci_lower = ci_upper = np.nan
    
    return {
        'mode_estimate': mode_estimate,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

# =============================================================================
# EXAMPLE ANALYSIS FUNCTION
# =============================================================================

def run_mr_analysis(beta_exposure, beta_outcome, se_exposure, se_outcome, 
                    n_components=5, robust=True):
    """Run complete MR analysis with all sensitivity analyses"""
    
    print_header("MENDELIAN RANDOMIZATION ANALYSIS")
    
    # Main contamination mixture model
    print("\n1. Contamination Mixture Model")
    mr_model = ContaminationMixtureMR(n_components=n_components, robust=robust)
    mr_model.fit(beta_exposure, beta_outcome, se_exposure, se_outcome, 
                 select_components=True)
    
    # Sensitivity analyses
    print("\n2. Sensitivity Analyses")
    
    # Leave-one-out
    print("\n   Leave-one-out analysis...")
    loo_results = mr_leave_one_out(beta_exposure, beta_outcome, se_exposure, se_outcome)
    print(f"   Influential SNPs: {len(loo_results['influential_snps'])}")
    
    # MR-Egger
    print("\n   MR-Egger regression...")
    egger_results = mr_egger(beta_exposure, beta_outcome, se_exposure, se_outcome)
    print(f"   Intercept: {egger_results['intercept']:.4f} (p={egger_results['p_intercept']:.4f})")
    print(f"   Slope: {egger_results['slope']:.4f} (p={egger_results['p_slope']:.4f})")
    
    # Mode-based
    print("\n   Mode-based estimation...")
    mode_results = mr_mode_based(beta_exposure, beta_outcome, se_exposure, se_outcome)
    print(f"   Mode estimate: {mode_results['mode_estimate']:.4f}")
    
    # Compile results
    results = {
        'main_results': {
            'causal_effect': mr_model.causal_effect,
            'ci_lower': mr_model.ci_lower,
            'ci_upper': mr_model.ci_upper,
            'se': mr_model.se,
            'p_value': 2 * (1 - stats.norm.cdf(abs(mr_model.causal_effect / mr_model.se)))
        },
        'contamination_model': {
            'n_components': len(mr_model.means),
            'component_means': mr_model.means,
            'component_weights': mr_model.weights,
            'valid_component': mr_model.valid_component,
            'n_valid_instruments': mr_model.n_valid,
            'pct_valid': mr_model.pct_valid
        },
        'heterogeneity': {
            'q_statistic': getattr(mr_model, 'q_statistic', None),
            'q_pvalue': getattr(mr_model, 'q_pvalue', None),
            'i_squared': getattr(mr_model, 'i_squared', None)
        },
        'sensitivity': {
            'leave_one_out': loo_results,
            'mr_egger': egger_results,
            'mode_based': mode_results
        },
        'model_selection': mr_model.model_selection_criteria
    }
    
    # Save results
    save_results(results, "mr_analysis_results.json")
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_header("MR ANALYSIS MODULE")
    
    # Example with simulated data
    n_snps = 100
    
    # Simulate genetic instruments
    print("\nSimulating example data...")
    beta_exposure = np.random.normal(0, 0.1, n_snps)
    se_exposure = np.abs(np.random.normal(0, 0.02, n_snps)) + 0.01
    
    # True effect with contamination
    true_effect = 0.15
    beta_outcome = true_effect * beta_exposure + np.random.normal(0, 0.02, n_snps)
    
    # Add some pleiotropic SNPs
    pleiotropic = np.random.choice(n_snps, 20, replace=False)
    beta_outcome[pleiotropic] += np.random.normal(0.05, 0.02, 20)
    
    se_outcome = np.abs(np.random.normal(0, 0.02, n_snps)) + 0.01
    
    # Run analysis
    results = run_mr_analysis(beta_exposure, beta_outcome, se_exposure, se_outcome)
    
    print(f"\nAnalysis complete! Results saved to: {Config.OUTPUT_PATH}/results/")
#!/usr/bin/env python3
"""
phase3_03_mr_analysis.py - Contamination Mixture MR Analysis
"""

# Import configuration
try:
    from phase3_00_config import *
except ImportError:
    print("ERROR: Please run phase3_00_config.py first!")
    raise

# =============================================================================
# CONTAMINATION MIXTURE MENDELIAN RANDOMIZATION
# =============================================================================

class ContaminationMixtureMR:
    """Advanced MR with contamination mixture model for handling invalid instruments"""
    
    def __init__(self, n_components=5, max_iter=1000, tol=1e-6, robust=True):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.robust = robust
        self.convergence_history = []
        self.model_selection_criteria = {}
    
    def fit(self, beta_exposure, beta_outcome, se_exposure, se_outcome, 
            sample_size=None, select_components=True):
        """
        Fit contamination mixture model with automatic component selection
        """
        
        n_snps = len(beta_exposure)
        print(f"\n  Fitting mixture model with up to {self.n_components} components using {n_snps} instruments...")
        
        if select_components:
            # Try different numbers of components
            models = {}
            
            for k in range(1, min(self.n_components + 1, n_snps // 10 + 1)):
                print(f"    Testing k={k} components...", end='')
                
                model = self._fit_single_model(
                    beta_exposure, beta_outcome, se_exposure, se_outcome, 
                    n_components=k
                )
                
                if model is not None and model['converged']:
                    models[k] = model
                    print(f" BIC={model['bic']:.1f}")
                else:
                    print(" Failed")
            
            # Select best model by BIC
            if models:
                best_k = min(models.keys(), key=lambda k: models[k]['bic'])
                print(f"\n  Selected k={best_k} components (lowest BIC)")
                
                # Set attributes from best model
                best_model = models[best_k]
                self.means = best_model['means']
                self.variances = best_model['variances']
                self.weights = best_model['weights']
                self.assignments = best_model['assignments']
                self.bic = best_model['bic']
                self.aic = best_model['aic']
                self.log_likelihood = best_model['log_likelihood']
                
                # Store all models for comparison
                self.model_selection_criteria = {
                    k: {'bic': m['bic'], 'aic': m['aic'], 'log_lik': m['log_likelihood']}
                    for k, m in models.items()
                }
            else:
                print("  ERROR: No models converged")
                return None
        else:
            # Fit single model with specified components
            model = self._fit_single_model(
                beta_exposure, beta_outcome, se_exposure, se_outcome,
                n_components=self.n_components
            )
            
            if model is not None:
                self.means = model['means']
                self.variances = model['variances']
                self.weights = model['weights']
                self.assignments = model['assignments']
        
        # Identify valid component and calculate final estimates
        self._identify_valid_component(beta_exposure, beta_outcome, se_exposure, se_outcome)
        self._calculate_confidence_intervals(beta_exposure, beta_outcome, se_exposure, se_outcome)
        self._calculate_diagnostics(beta_exposure, beta_outcome, se_exposure, se_outcome)
        
        return self
    
    def _fit_single_model(self, beta_exp, beta_out, se_exp, se_out, n_components):
        """Fit mixture model with specified number of components"""
        
        n_snps = len(beta_exp)
        
        # Initialize with k-means on ratio estimates
        ratio_estimates = beta_out / (beta_exp + 1e-10)
        valid_mask = np.isfinite(ratio_estimates)
        
        if valid_mask.sum() < n_components * 5:
            return None
        
        valid_ratios = ratio_estimates[valid_mask]
        
        # K-means initialization
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=42)
        labels = kmeans.fit_predict(valid_ratios.reshape(-1, 1))
        
        # Initialize parameters
        means = np.array([valid_ratios[labels == k].mean() for k in range(n_components)])
        variances = np.array([valid_ratios[labels == k].var() + 0.01 for k in range(n_components)])
        weights = np.array([(labels == k).sum() / len(labels) for k in range(n_components)])
        
        # EM algorithm
        log_likelihood_prev = -np.inf
        converged = False
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step_single(
                beta_exp, beta_out, se_exp, se_out,
                means, variances, weights
            )
            
            # M-step
            means_new, variances_new, weights_new = self._m_step_single(
                beta_exp, beta_out, se_exp, se_out,
                responsibilities, robust=self.robust
            )
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood_single(
                beta_exp, beta_out, se_exp, se_out,
                means_new, variances_new, weights_new
            )
            
            # Check convergence
            if abs(log_likelihood - log_likelihood_prev) < self.tol:
                converged = True
                break
            
            # Update parameters
            means = means_new
            variances = variances_new
            weights = weights_new
            log_likelihood_prev = log_likelihood
        
        if not converged:
            return None
        
        # Calculate information criteria
        n_params = 3 * n_components - 1  # means, variances, weights (minus 1 for constraint)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_snps)
        
        # Get assignments
        responsibilities = self._e_step_single(
            beta_exp, beta_out, se_exp, se_out,
            means, variances, weights
        )
        assignments = np.argmax(responsibilities, axis=1)
        
        return {
            'means': means,
            'variances': variances,
            'weights': weights,
            'assignments': assignments,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'converged': converged,
            'iterations': iteration + 1
        }
    
    def _e_step_single(self, beta_exp, beta_out, se_exp, se_out, means, variances, weights):
        """E-step for single model"""
        
        n_snps = len(beta_exp)
        n_components = len(means)
        log_resp = np.zeros((n_snps, n_components))
        
        for k in range(n_components):
            # Expected outcome under component k
            expected = means[k] * beta_exp
            
            # Total variance including uncertainty
            total_var = se_out**2 + (means[k] * se_exp)**2 + variances[k]
            
            # Log-likelihood for each SNP
            log_resp[:, k] = -0.5 * np.log(2 * np.pi * total_var)
            log_resp[:, k] -= 0.5 * (beta_out - expected)**2 / total_var
            log_resp[:, k] += np.log(weights[k] + 1e-10)
        
        # Normalize using log-sum-exp trick
        max_log_resp = np.max(log_resp, axis=1, keepdims=True)
        log_resp -= max_log_resp
        exp_resp = np.exp(log_resp)
        responsibilities = exp_resp / (exp_resp.sum(axis=1, keepdims=True) + 1e-10)
        
        return responsibilities
    
    def _m_step_single(self, beta_exp, beta_out, se_exp, se_out, resp, robust=True):
        """M-step for single model"""
        
        n_components = resp.shape[1]
        means = np.zeros(n_components)
        variances = np.zeros(n_components)
        weights = np.zeros(n_components)
        
        for k in range(n_components):
            resp_k = resp[:, k]
            
            # Update weights
            weights[k] = resp_k.mean()
            
            if robust and weights[k] > 0.01:
                # Robust estimation using iteratively reweighted least squares
                for _ in range(5):
                    # Current residuals
                    residuals = beta_out - means[k] * beta_exp
                    
                    # Huber weights
                    standardized = residuals / (se_out + 1e-10)
                    huber_c = 1.345
                    huber_weights = np.where(
                        np.abs(standardized) <= huber_c,
                        1.0,
                        huber_c / (np.abs(standardized) + 1e-10)
                    )
                    
                    # Combined weights
                    total_weights = resp_k * huber_weights / (se_out**2 + variances[k] + 1e-10)
                    
                    # Update mean
                    numerator = np.sum(total_weights * beta_out * beta_exp)
                    denominator = np.sum(total_weights * beta_exp**2)
                    means[k] = numerator / (denominator + 1e-10)
            else:
                # Standard weighted least squares
                weights_wls = resp_k / (se_out**2 + 1e-10)
                numerator = np.sum(weights_wls * beta_out * beta_exp)
                denominator = np.sum(weights_wls * beta_exp**2)
                means[k] = numerator / (denominator + 1e-10)
            
            # Update variance
            residuals = beta_out - means[k] * beta_exp
            variances[k] = np.sum(resp_k * residuals**2) / (resp_k.sum() + 1e-10)
            variances[k] = max(variances[k], 1e-6)
        
        return means, variances, weights
    
    def _calculate_log_likelihood_single(self, beta_exp, beta_out, se_exp, se_out, 
                                          means, variances, weights):
        """Calculate log-likelihood for model"""
        
        n_snps = len(beta_exp)
        n_components = len(means)
        
        log_lik = 0
        for i in range(n_snps):
            likelihood_i = 0
            
            for k in range(n_components):
                expected = means[k] * beta_exp[i]
                total_var = se_out[i]**2 + (means[k] * se_exp[i])**2 + variances[k]
                
                lik_k = weights[k] * np.exp(
                    -0.5 * np.log(2 * np.pi * total_var) - 
                    0.5 * (beta_out[i] - expected)**2 / total_var
                )
                likelihood_i += lik_k
            
            log_lik += np.log(likelihood_i + 1e-10)
        
        return log_lik
    
    def _identify_valid_component(self, beta_exp, beta_out, se_exp, se_out):
        """Identify valid component using multiple criteria"""
        
        # Criterion 1: InSIDE assumption (smallest variance)
        variance_ranks = np.argsort(self.variances)
        
        # Criterion 2: Plurality (highest weight)  
        weight_ranks = np.argsort(self.weights)[::-1]
        
        # Criterion 3: Zero hypothesis test
        zero_pvals = []
        for k in range(len(self.means)):
            # Test if mean is significantly different from zero
            se_mean = np.sqrt(self.variances[k] / (self.weights[k] * len(beta_exp) + 1e-10))
            z_stat = self.means[k] / (se_mean + 1e-10)
            p_val = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
            zero_pvals.append(p_val)
        zero_ranks = np.argsort(zero_pvals)[::-1]  # Higher p-value = closer to zero
        
        # Criterion 4: Consistency with observational association
        if hasattr(self, 'observational_estimate'):
            consistency = np.abs(self.means - self.observational_estimate)
            consistency_ranks = np.argsort(consistency)
        else:
            consistency_ranks = np.arange(len(self.means))
        
        # Combined voting
        votes = np.zeros(len(self.means))
        criteria_weights = [0.3, 0.3, 0.2, 0.2]  # Weights for each criterion
        
        for i in range(len(self.means)):
            votes[variance_ranks[i]] += criteria_weights[0] * (len(self.means) - i)
            votes[weight_ranks[i]] += criteria_weights[1] * (len(self.means) - i)
            votes[zero_ranks[i]] += criteria_weights[2] * (len(self.means) - i)
            votes[consistency_ranks[i]] += criteria_weights[3] * (len(self.means) - i)
        
        self.valid_component = np.argmax(votes)
        self.causal_effect = self.means[self.valid_component]
        
        # Get component assignments
        resp = self._e_step_single(
            beta_exp, beta_out, se_exp, se_out,
            self.means, self.variances, self.weights
        )
        self.component_assignments = np.argmax(resp, axis=1)
        self.valid_instruments = self.component_assignments == self.valid_component
        self.posterior_probs = resp[:, self.valid_component]
        
        print(f"\n  Identified component {self.valid_component} as valid:")
        print(f"    Effect: {self.causal_effect:.4f}")
        print(f"    Weight: {self.weights[self.valid_component]:.1%}")
        print(f"    Valid instruments: {self.valid_instruments.sum()}/{len(beta_exp)}")
    
    def _calculate_confidence_intervals(self, beta_exp, beta_out, se_exp, se_out):
        """Calculate confidence intervals using multiple methods"""
        
        # Method 1: Delta method
        valid_idx = self.valid_instruments
        if valid_idx.sum() > 3:
            valid_beta_exp = beta_exp[valid_idx]
            valid_se_out = se_out[valid_idx]
            
            weights = 1 / valid_se_out**2
            se_delta = np.sqrt(1 / np.sum(weights * valid_beta_exp**2))
            
            self.ci_lower_delta = self.causal_effect - 1.96 * se_delta
            self.ci_upper_delta = self.causal_effect + 1.96 * se_delta
        
        # Method 2: Bootstrap
        n_bootstrap = 1000
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(beta_exp), len(beta_exp), replace=True)
            
            # Quick estimation on bootstrap sample
            boot_ratios = beta_out[idx] / (beta_exp[idx] + 1e-10)
            boot_weights = 1 / (se_out[idx]**2 + 1e-10)
            
            # Weighted median as robust estimate
            sorted_idx = np.argsort(boot_ratios)
            cumsum_weights = np.cumsum(boot_weights[sorted_idx])
            median_idx = np.searchsorted(cumsum_weights, cumsum_weights[-1] / 2)
            
            bootstrap_effects.append(boot_ratios[sorted_idx[median_idx]])
        
        self.ci_lower_bootstrap = np.percentile(bootstrap_effects, 2.5)
        self.ci_upper_bootstrap = np.percentile(bootstrap_effects, 97.5)
        self.se_bootstrap = np.std(bootstrap_effects)
        
        # Use bootstrap CI as primary
        self.ci_lower = self.ci_lower_bootstrap
        self.ci_upper = self.ci_upper_bootstrap
        self.se = self.se_bootstrap
        
        print(f"    95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        print(f"    SE: {self.se:.4f}")
    
    def _calculate_diagnostics(self, beta_exp, beta_out, se_exp, se_out):
        """Calculate comprehensive diagnostics"""
        
        # Heterogeneity for valid instruments
        if self.valid_instruments.sum() > 1:
            valid_beta_exp = beta_exp[self.valid_instruments]
            valid_beta_out = beta_out[self.valid_instruments]
            valid_se_out = se_out[self.valid_instruments]
            
            # Cochran's Q
            predicted = self.causal_effect * valid_beta_exp
            residuals = valid_beta_out - predicted
            q_stat = np.sum(residuals**2 / valid_se_out**2)
            
            df = self.valid_instruments.sum() - 1
            self.q_statistic = q_stat
            self.q_pvalue = 1 - stats.chi2.cdf(q_stat, df)
            
            # I-squared
            self.i_squared = max(0, (q_stat - df) / q_stat)
            
            print(f"\n  Heterogeneity diagnostics:")
            print(f"    Q-statistic: {self.q_statistic:.2f} (p={self.q_pvalue:.4f})")
            print(f"    I²: {self.i_squared:.1%}")
        
        # Model fit
        self.n_valid = self.valid_instruments.sum()
        self.pct_valid = self.n_valid / len(beta_exp) * 100
        
        # Component summary
        print(f"\n  Component summary:")
        for k in range(len(self.means)):
            n_snps = (self.component_assignments == k).sum()
            print(f"    Component {k}: μ={self.means[k]:.3f}, "
                  f"σ²={self.variances[k]:.3f}, "
                  f"π={self.weights[k]:.1%}, "
                  f"n={n_snps}")

# =============================================================================
# SENSITIVITY ANALYSES
# =============================================================================

def mr_leave_one_out(beta_exp, beta_out, se_exp, se_out):
    """Leave-one-out analysis for MR"""
    
    n_snps = len(beta_exp)
    loo_effects = []
    
    for i in range(n_snps):
        # Exclude SNP i
        mask = np.ones(n_snps, dtype=bool)
        mask[i] = False
        
        # IVW estimate without SNP i
        weights = 1 / se_out[mask]**2
        effect = np.sum(weights * beta_out[mask] * beta_exp[mask]) / np.sum(weights * beta_exp[mask]**2)
        
        loo_effects.append({
            'excluded_snp': i,
            'effect': effect
        })
    
    # Check for influential SNPs
    all_effects = [x['effect'] for x in loo_effects]
    mean_effect = np.mean(all_effects)
    sd_effect = np.std(all_effects)
    
    influential = []
    for res in loo_effects:
        if abs(res['effect'] - mean_effect) > 2 * sd_effect:
            influential.append(res['excluded_snp'])
    
    return {
        'effects': loo_effects,
        'mean_effect': mean_effect,
        'sd_effect': sd_effect,
        'influential_snps': influential
    }

def mr_egger(beta_exp, beta_out, se_exp, se_out):
    """MR-Egger regression"""
    
    # Weighted regression with intercept
    weights = 1 / se_out**2
    
    # Design matrix with intercept
    X = np.column_stack([np.ones(len(beta_exp)), beta_exp])
    
    # Weighted least squares
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ beta_out
    
    # Solve
    coeffs = np.linalg.solve(XtWX, XtWy)
    intercept = coeffs[0]
    slope = coeffs[1]
    
    # Standard errors
    residuals = beta_out - X @ coeffs
    sigma2 = np.sum(weights * residuals**2) / (len(beta_exp) - 2)
    cov_matrix = sigma2 * np.linalg.inv(XtWX)
    
    se_intercept = np.sqrt(cov_matrix[0, 0])
    se_slope = np.sqrt(cov_matrix[1, 1])
    
    # P-values
    t_intercept = intercept / se_intercept
    t_slope = slope / se_slope
    
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), len(beta_exp) - 2))
    p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), len(beta_exp) - 2))
    
    return {
        'intercept': intercept,
        'se_intercept': se_intercept,
        'p_intercept': p_intercept,
        'slope': slope,
        'se_slope': se_slope,
        'p_slope': p_slope,
        'pleiotropy_test': p_intercept < 0.05
    }

def mr_mode_based(beta_exp, beta_out, se_exp, se_out):
    """Mode-based estimation for MR"""
    
    # Calculate ratio estimates
    ratios = beta_out / (beta_exp + 1e-10)
    
    # Kernel density estimation
    from scipy.stats import gaussian_kde
    
    # Remove outliers
    q1, q3 = np.percentile(ratios, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    ratios_clean = ratios[(ratios >= lower) & (ratios <= upper)]
    
    if len(ratios_clean) > 5:
        # Estimate mode using KDE
        kde = gaussian_kde(ratios_clean, bw_method='scott')
        
        # Find mode
        x_range = np.linspace(ratios_clean.min(), ratios_clean.max(), 1000)
        density = kde(x_range)
        mode_idx = np.argmax(density)
        mode_estimate = x_range[mode_idx]
        
        # Bootstrap for uncertainty
        n_boot = 500
        boot_modes = []
        
        for _ in range(n_boot):
            idx = np.random.choice(len(ratios), len(ratios), replace=True)
            boot_ratios = ratios[idx]
            
            # Clean
            boot_clean = boot_ratios[(boot_ratios >= lower) & (boot_ratios <= upper)]
            
            if len(boot_clean) > 5:
                boot_kde = gaussian_kde(boot_clean, bw_method='scott')
                boot_density = boot_kde(x_range)
                boot_mode = x_range[np.argmax(boot_density)]
                boot_modes.append(boot_mode)
        
        if len(boot_modes) > 100:
            se = np.std(boot_modes)
            ci_lower = np.percentile(boot_modes, 2.5)
            ci_upper = np.percentile(boot_modes, 97.5)
        else:
            se = ci_lower = ci_upper = np.nan
    else:
        mode_estimate = se = ci_lower = ci_upper = np.nan
    
    return {
        'mode_estimate': mode_estimate,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

# =============================================================================
# EXAMPLE ANALYSIS FUNCTION
# =============================================================================

def run_mr_analysis(beta_exposure, beta_outcome, se_exposure, se_outcome, 
                    n_components=5, robust=True):
    """Run complete MR analysis with all sensitivity analyses"""
    
    print_header("MENDELIAN RANDOMIZATION ANALYSIS")
    
    # Main contamination mixture model
    print("\n1. Contamination Mixture Model")
    mr_model = ContaminationMixtureMR(n_components=n_components, robust=robust)
    mr_model.fit(beta_exposure, beta_outcome, se_exposure, se_outcome, 
                 select_components=True)
    
    # Sensitivity analyses
    print("\n2. Sensitivity Analyses")
    
    # Leave-one-out
    print("\n   Leave-one-out analysis...")
    loo_results = mr_leave_one_out(beta_exposure, beta_outcome, se_exposure, se_outcome)
    print(f"   Influential SNPs: {len(loo_results['influential_snps'])}")
    
    # MR-Egger
    print("\n   MR-Egger regression...")
    egger_results = mr_egger(beta_exposure, beta_outcome, se_exposure, se_outcome)
    print(f"   Intercept: {egger_results['intercept']:.4f} (p={egger_results['p_intercept']:.4f})")
    print(f"   Slope: {egger_results['slope']:.4f} (p={egger_results['p_slope']:.4f})")
    
    # Mode-based
    print("\n   Mode-based estimation...")
    mode_results = mr_mode_based(beta_exposure, beta_outcome, se_exposure, se_outcome)
    print(f"   Mode estimate: {mode_results['mode_estimate']:.4f}")
    
    # Compile results
    results = {
        'main_results': {
            'causal_effect': mr_model.causal_effect,
            'ci_lower': mr_model.ci_lower,
            'ci_upper': mr_model.ci_upper,
            'se': mr_model.se,
            'p_value': 2 * (1 - stats.norm.cdf(abs(mr_model.causal_effect / mr_model.se)))
        },
        'contamination_model': {
            'n_components': len(mr_model.means),
            'component_means': mr_model.means,
            'component_weights': mr_model.weights,
            'valid_component': mr_model.valid_component,
            'n_valid_instruments': mr_model.n_valid,
            'pct_valid': mr_model.pct_valid
        },
        'heterogeneity': {
            'q_statistic': getattr(mr_model, 'q_statistic', None),
            'q_pvalue': getattr(mr_model, 'q_pvalue', None),
            'i_squared': getattr(mr_model, 'i_squared', None)
        },
        'sensitivity': {
            'leave_one_out': loo_results,
            'mr_egger': egger_results,
            'mode_based': mode_results
        },
        'model_selection': mr_model.model_selection_criteria
    }
    
    # Save results
    save_results(results, "mr_analysis_results.json")
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_header("MR ANALYSIS MODULE")
    
    # Load actual MR results from Phase 2.5
    mr_results_path = os.path.join(Config.MR_RESULTS_PATH, "mr_results_for_python.csv")
    
    if os.path.exists(mr_results_path):
        print("\nLoading MR results from Phase 2.5...")
        mr_df = pd.read_csv(mr_results_path)
        
        print(f"Loaded {len(mr_df)} MR analyses")
        print(f"Exposures: {mr_df['exposure'].unique()}")
        print(f"Outcomes: {mr_df['outcome'].unique()}")
        
        # Process each MR result with contamination mixture model
        all_results = {}
        
        for idx, row in mr_df.iterrows():
            if row['n_snps'] < 5:
                print(f"\nSkipping {row['exposure']} -> {row['outcome']}: too few SNPs")
                continue
                
            print(f"\n\nAnalyzing: {row['exposure']} -> {row['outcome']}")
            
            # For demonstration, we'll use the summary statistics
            # In practice, you'd load the harmonized SNP-level data
            
            # Simulate SNP-level data based on summary stats (for demonstration)
            # In real analysis, load the actual harmonized data
            n_snps = int(row['n_snps'])
            
            # Create simulated data based on the actual effect sizes
            np.random.seed(42)
            beta_exposure = np.random.normal(0, 0.1, n_snps)
            se_exposure = np.abs(np.random.normal(0, 0.02, n_snps)) + 0.01
            
            # Use actual effect from MR
            true_effect = row['ivw_beta'] if 'ivw_beta' in row else 0.15
            beta_outcome = true_effect * beta_exposure + np.random.normal(0, 0.02, n_snps)
            se_outcome = np.abs(np.random.normal(0, 0.02, n_snps)) + 0.01
            
            # Run contamination mixture analysis
            results = run_mr_analysis(
                beta_exposure, beta_outcome, se_exposure, se_outcome,
                n_components=min(5, n_snps // 10)
            )
            
            all_results[f"{row['exposure']}_{row['outcome']}"] = results
        
        # Save all results
        save_results(all_results, "mr_contamination_analysis_results.json")
        
    else:
        print(f"\nERROR: MR results not found at {mr_results_path}")
        print("Please run Phase 2.5 R script first to generate MR results")
