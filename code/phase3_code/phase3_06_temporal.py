#!/usr/bin/env python3
"""
phase3_06_temporal.py - Temporal Causal Discovery
"""

# Import configuration
try:
    from phase3_00_config import *
except ImportError:
    print("ERROR: Please run phase3_00_config.py first!")
    raise

# =============================================================================
# TEMPORAL CAUSAL DISCOVERY
# =============================================================================

class TemporalCausalDiscovery:
    """Advanced temporal causal discovery with multiple methods"""
    
    def __init__(self, method='var', max_lag=2, alpha=0.01):
        self.method = method
        self.max_lag = max_lag
        self.alpha = alpha
        self.results = {}
        
    def discover_temporal_causality(self, temporal_data, outcome_data=None, 
                                  clinical_data=None):
        """
        Discover temporal causal relationships
        
        Parameters:
        -----------
        temporal_data: dict with 'sequences' (n_samples, n_features, n_time)
        outcome_data: array-like, shape (n_samples, n_outcomes), optional
        clinical_data: array-like, shape (n_samples, n_clinical), optional
        """
        
        sequences = temporal_data['sequences']
        n_samples, n_features, n_time = sequences.shape
        
        print(f"\n  Discovering temporal causality: {n_features} features, {n_time} timepoints")
        
        # Method 1: Vector Autoregression (VAR)
        print("  Method 1: VAR analysis...")
        var_results = self._fit_var_model(sequences)
        
        # Method 2: Granger Causality
        print("  Method 2: Granger causality testing...")
        granger_results = self._test_granger_causality(sequences)
        
        # Method 3: Transfer Entropy
        print("  Method 3: Transfer entropy...")
        te_results = self._calculate_transfer_entropy(sequences)
        
        # Method 4: Time-lagged cross-correlation
        print("  Method 4: Lagged correlations...")
        lagged_corr = self._calculate_lagged_correlations(sequences)
        
        # Method 5: PC algorithm with time constraints
        print("  Method 5: PC algorithm with temporal constraints...")
        pc_temporal = self._pc_temporal(sequences)
        
        # Combine evidence from multiple methods
        print("  Combining evidence from multiple methods...")
        consensus_edges = self._combine_evidence(
            var_results, granger_results, te_results, lagged_corr, pc_temporal
        )
        
        # Test outcome associations if provided
        outcome_associations = None
        if outcome_data is not None:
            print("  Testing associations with outcomes...")
            outcome_associations = self._test_outcome_associations(
                sequences, outcome_data
            )
        
        # Identify key temporal patterns
        temporal_patterns = self._identify_temporal_patterns(sequences)
        
        self.results = {
            'var_model': var_results,
            'granger_causality': granger_results,
            'transfer_entropy': te_results,
            'lagged_correlations': lagged_corr,
            'pc_temporal': pc_temporal,
            'consensus_edges': consensus_edges,
            'outcome_associations': outcome_associations,
            'temporal_patterns': temporal_patterns,
            'n_features': n_features,
            'n_timepoints': n_time,
            'feature_names': temporal_data.get('metabolite_names', None)
        }
        
        return self
    
    def _fit_var_model(self, sequences):
        """Fit Vector Autoregression model"""
        
        n_samples, n_features, n_time = sequences.shape
        
        # Check if we have enough time points
        if n_time <= self.max_lag:
            print(f"    WARNING: Not enough time points ({n_time}) for max_lag ({self.max_lag})")
            return {
                'coefficients_by_lag': [],
                'pvalues_by_lag': [],
                'is_stable': False,
                'max_eigenvalue': 0
            }
        
        # Prepare data for VAR
        var_data = []
        
        for t in range(self.max_lag, n_time):
            # Current values
            y_t = sequences[:, :, t]
            
            # Lagged values
            X_t = []
            for lag in range(1, self.max_lag + 1):
                lag_index = t - lag
                if lag_index >= 0:
                    X_t.append(sequences[:, :, lag_index])
            
            if len(X_t) > 0:
                X_t = np.hstack(X_t)
                var_data.append((X_t, y_t))
        
        if len(var_data) == 0:
            return {
                'coefficients_by_lag': [],
                'pvalues_by_lag': [],
                'is_stable': False,
                'max_eigenvalue': 0
            }
        
        # Fit VAR model for each feature
        var_coefficients = np.zeros((n_features, n_features * self.max_lag))
        var_pvalues = np.ones((n_features, n_features * self.max_lag))
        
        for j in range(n_features):
            # Collect data for feature j
            X_all = []
            y_all = []
            
            for X_t, y_t in var_data:
                X_all.append(X_t)
                y_all.append(y_t[:, j])
            
            try:
                X = np.vstack(X_all)
                y = np.hstack(y_all)
                
                # Add intercept
                X = np.column_stack([np.ones(len(y)), X])
                
                # OLS regression
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                var_coefficients[j, :] = beta[1:]  # Exclude intercept
                
                # Calculate p-values
                residuals = y - X @ beta
                sigma2 = np.sum(residuals**2) / (len(y) - X.shape[1])
                cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(cov_matrix)[1:])  # Exclude intercept
                
                t_stats = var_coefficients[j, :] / (se + 1e-10)
                var_pvalues[j, :] = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - X.shape[1]))
                
            except Exception as e:
                continue
        
        # Reshape by lag
        coefficients_by_lag = []
        pvalues_by_lag = []
        
        for lag in range(self.max_lag):
            start_idx = lag * n_features
            end_idx = (lag + 1) * n_features
            coefficients_by_lag.append(var_coefficients[:, start_idx:end_idx])
            pvalues_by_lag.append(var_pvalues[:, start_idx:end_idx])
        
        # Check stability
        # Create companion matrix
        companion = np.zeros((n_features * self.max_lag, n_features * self.max_lag))
        companion[:n_features, :] = var_coefficients
        companion[n_features:, :n_features * (self.max_lag - 1)] = np.eye(n_features * (self.max_lag - 1))
        
        eigenvalues = np.linalg.eigvals(companion)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        is_stable = max_eigenvalue < 1
        
        return {
            'coefficients_by_lag': coefficients_by_lag,
            'pvalues_by_lag': pvalues_by_lag,
            'is_stable': is_stable,
            'max_eigenvalue': max_eigenvalue
        }
    
    def _test_granger_causality(self, sequences):
        """Test pairwise Granger causality"""
        
        n_samples, n_features, n_time = sequences.shape
        
        # Check if we have enough time points
        if n_time <= self.max_lag:
            print(f"    WARNING: Not enough time points ({n_time}) for Granger causality with max_lag ({self.max_lag})")
            return {
                'f_statistics': np.zeros((n_features, n_features)),
                'p_values': np.ones((n_features, n_features)),
                'significant': np.zeros((n_features, n_features), dtype=bool),
                'n_significant': 0
            }
        
        # Granger causality matrix
        gc_matrix = np.zeros((n_features, n_features))
        gc_pvalues = np.ones((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue
                
                # Test if i Granger-causes j
                # Prepare data
                data_pairs = []
                
                for t in range(self.max_lag, n_time):
                    # Target: j at time t
                    y = sequences[:, j, t]
                    
                    # Features: lagged values of j
                    X_restricted = []
                    for lag in range(1, self.max_lag + 1):
                        X_restricted.append(sequences[:, j, t - lag])
                    
                    # Features: lagged values of j and i  
                    X_full = X_restricted.copy()
                    for lag in range(1, self.max_lag + 1):
                        X_full.append(sequences[:, i, t - lag])
                    
                    X_restricted = np.column_stack(X_restricted)
                    X_full = np.column_stack(X_full)
                    
                    data_pairs.append((y, X_restricted, X_full))
                
                # F-test comparing models
                rss_restricted = 0
                rss_full = 0
                n_obs = 0
                
                for y, X_r, X_f in data_pairs:
                    # Restricted model
                    beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
                    rss_restricted += np.sum((y - X_r @ beta_r)**2)
                    
                    # Full model
                    beta_f = np.linalg.lstsq(X_f, y, rcond=None)[0]
                    rss_full += np.sum((y - X_f @ beta_f)**2)
                    
                    n_obs += len(y)
                
                # Calculate F-statistic
                _, X_r, X_f = data_pairs[-1]
                df1 = X_f.shape[1] - X_r.shape[1]
                df2 = n_obs - X_f.shape[1]
                
                if df2 > 0:
                    f_stat = ((rss_restricted - rss_full) / df1) / (rss_full / df2)
                    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                else:
                    f_stat = 0
                    p_value = 1
                
                gc_matrix[i, j] = f_stat
                gc_pvalues[i, j] = p_value
        
        # Apply FDR correction
        pvals_flat = gc_pvalues.flatten()
        mask = ~np.eye(n_features, dtype=bool).flatten()
        
        if np.any(mask):
            reject, pvals_adj, _, _ = multipletests(
                pvals_flat[mask], alpha=self.alpha, method='fdr_bh'
            )
            
            gc_significant = np.zeros((n_features, n_features), dtype=bool)
            gc_significant.flat[mask] = reject
        else:
            gc_significant = np.zeros((n_features, n_features), dtype=bool)
        
        return {
            'f_statistics': gc_matrix,
            'p_values': gc_pvalues,
            'significant': gc_significant,
            'n_significant': gc_significant.sum()
        }
    
    def _calculate_transfer_entropy(self, sequences):
        """Calculate transfer entropy between time series"""
        
        n_samples, n_features, n_time = sequences.shape
        
        # Discretize continuous values for entropy calculation
        n_bins = 10
        sequences_discrete = np.zeros_like(sequences)
        
        for i in range(n_features):
            for t in range(n_time):
                # Discretize using quantiles
                values = sequences[:, i, t]
                bins = np.percentile(values, np.linspace(0, 100, n_bins + 1))
                bins[0] = bins[0] - 1e-10
                bins[-1] = bins[-1] + 1e-10
                sequences_discrete[:, i, t] = np.digitize(values, bins)
        
        # Transfer entropy matrix
        te_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue
                
                # Calculate TE from i to j
                te = 0
                n_pairs = 0
                
                for lag in range(1, min(self.max_lag + 1, n_time)):
                    # Future of j
                    j_future = sequences_discrete[:, j, lag:]
                    
                    # Past of j
                    j_past = sequences_discrete[:, j, :-lag]
                    
                    # Past of i
                    i_past = sequences_discrete[:, i, :-lag]
                    
                    # Calculate conditional entropy (simplified)
                    for t in range(j_future.shape[1]):
                        # Count occurrences
                        joint_counts = defaultdict(int)
                        
                        for s in range(n_samples):
                            state = (
                                j_future[s, t],
                                j_past[s, t],
                                i_past[s, t]
                            )
                            joint_counts[state] += 1
                        
                        # Calculate entropies
                        total = sum(joint_counts.values())
                        
                        for state, count in joint_counts.items():
                            if count > 0:
                                p = count / total
                                te += p * np.log2(p + 1e-10)
                        
                        n_pairs += 1
                
                te_matrix[i, j] = -te / n_pairs if n_pairs > 0 else 0
        
        # Normalize
        if te_matrix.max() > te_matrix.min():
            te_matrix = (te_matrix - te_matrix.min()) / (te_matrix.max() - te_matrix.min() + 1e-10)
        
        return {
            'transfer_entropy': te_matrix,
            'threshold': np.percentile(te_matrix[te_matrix > 0], 90) if np.any(te_matrix > 0) else 0
        }
    
    def _calculate_lagged_correlations(self, sequences):
        """Calculate time-lagged cross-correlations"""
        
        n_samples, n_features, n_time = sequences.shape
        
        # Store maximum absolute correlation across lags
        max_corr = np.zeros((n_features, n_features))
        optimal_lag = np.zeros((n_features, n_features), dtype=int)
        
        for i in range(n_features):
            for j in range(n_features):
                correlations = []
                
                for lag in range(-self.max_lag, self.max_lag + 1):
                    if lag < 0:
                        # i leads j
                        if n_time + lag > 0:
                            x = sequences[:, i, :n_time + lag].flatten()
                            y = sequences[:, j, -lag:].flatten()
                        else:
                            correlations.append(0)
                            continue
                    elif lag > 0:
                        # j leads i
                        if n_time - lag > 0:
                            x = sequences[:, i, lag:].flatten()
                            y = sequences[:, j, :n_time - lag].flatten()
                        else:
                            correlations.append(0)
                            continue
                    else:
                        # Contemporaneous
                        x = sequences[:, i, :].flatten()
                        y = sequences[:, j, :].flatten()
                    
                    # Remove NaN values
                    valid = ~(np.isnan(x) | np.isnan(y))
                    if valid.sum() > 10:
                        corr = np.corrcoef(x[valid], y[valid])[0, 1]
                        correlations.append(abs(corr))
                    else:
                        correlations.append(0)
                
                # Find maximum correlation
                if correlations:
                    lag_idx = np.argmax(correlations)
                    max_corr[i, j] = correlations[lag_idx]
                    optimal_lag[i, j] = lag_idx - self.max_lag
        
        return {
            'max_correlation': max_corr,
            'optimal_lag': optimal_lag,
            'significant_corr': max_corr > 0.5  # Threshold for significance
        }
    
    def _pc_temporal(self, sequences):
        """PC algorithm with temporal constraints"""
        
        n_samples, n_features, n_time = sequences.shape
        
        # Create lagged dataset
        lagged_data = []
        feature_names = []
        
        for lag in range(min(self.max_lag + 1, n_time)):
            lagged_data.append(sequences[:, :, lag])
            for j in range(n_features):
                feature_names.append(f'X{j}_t{lag}')
        
        if not lagged_data:
            return {
                'adjacency': np.array([]),
                'temporal_edges': [],
                'n_edges': 0
            }
        
        # Stack all lagged variables
        X = np.hstack(lagged_data)
        
        # Calculate correlation matrix
        C = np.corrcoef(X.T)
        
        # Apply PC algorithm with temporal constraints
        n_vars = X.shape[1]
        adjacency = np.zeros((n_vars, n_vars), dtype=bool)
        
        # Skeleton discovery with temporal constraints
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Extract time indices
                t_i = int(feature_names[i].split('_t')[1])
                t_j = int(feature_names[j].split('_t')[1])
                
                # Only test if temporal order is valid
                if t_i <= t_j:
                    # Test conditional independence
                    if abs(C[i, j]) > 0.3:  # Simple threshold
                        adjacency[i, j] = True
                        if t_i < t_j:  # Directed edge
                            adjacency[j, i] = False
                        else:  # Undirected if same time
                            adjacency[j, i] = True
        
        # Extract temporal edges
        temporal_edges = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency[i, j]:
                    f_i = int(feature_names[i].split('_')[0][1:])
                    t_i = int(feature_names[i].split('_t')[1])
                    f_j = int(feature_names[j].split('_')[0][1:])
                    t_j = int(feature_names[j].split('_t')[1])
                    
                    if t_i < t_j:  # Temporal edge
                        temporal_edges.append({
                            'from': f_i,
                            'to': f_j,
                            'lag': t_j - t_i,
                            'strength': abs(C[i, j])
                        })
        
        return {
            'adjacency': adjacency,
            'temporal_edges': temporal_edges,
            'n_edges': len(temporal_edges)
        }
    
    def _combine_evidence(self, var_results, granger_results, te_results, 
                         lagged_corr, pc_temporal):
        """Combine evidence from multiple methods"""
        
        # Get number of features
        if var_results['coefficients_by_lag']:
            n_features = var_results['coefficients_by_lag'][0].shape[0]
        elif granger_results['f_statistics'] is not None:
            n_features = granger_results['f_statistics'].shape[0]
        else:
            return {
                'consensus_matrix': np.array([]),
                'consensus_edges': np.array([]),
                'significant_edges': [],
                'n_edges': 0
            }
        
        # Create consensus matrix
        consensus_matrix = np.zeros((n_features, n_features))
        
        # Weight different methods
        weights = {
            'var': 0.25,
            'granger': 0.25,
            'te': 0.2,
            'correlation': 0.15,
            'pc': 0.15
        }
        
        # VAR evidence
        if var_results['coefficients_by_lag']:
            for lag_idx, (coef, pval) in enumerate(zip(
                var_results['coefficients_by_lag'], 
                var_results['pvalues_by_lag']
            )):
                significant = pval < self.alpha
                consensus_matrix += weights['var'] * (np.abs(coef) * significant)
        
        # Granger causality evidence
        if granger_results['significant'] is not None:
            consensus_matrix += weights['granger'] * granger_results['significant']
        
        # Transfer entropy evidence
        if te_results['transfer_entropy'] is not None:
            te_norm = te_results['transfer_entropy'] / (te_results['transfer_entropy'].max() + 1e-10)
            consensus_matrix += weights['te'] * (te_norm > te_results['threshold'])
        
        # Correlation evidence
        if lagged_corr['significant_corr'] is not None:
            consensus_matrix += weights['correlation'] * lagged_corr['significant_corr']
        
        # PC temporal evidence
        if pc_temporal['temporal_edges']:
            pc_matrix = np.zeros((n_features, n_features))
            for edge in pc_temporal['temporal_edges']:
                if edge['from'] < n_features and edge['to'] < n_features:
                    pc_matrix[edge['from'], edge['to']] = edge['strength']
            consensus_matrix += weights['pc'] * pc_matrix
        
        # Threshold for consensus
        consensus_threshold = 0.5
        consensus_edges = consensus_matrix > consensus_threshold
        
        # Extract significant edges with details
        significant_edges = []
        for i in range(n_features):
            for j in range(n_features):
                if consensus_edges[i, j] and i != j:
                    significant_edges.append({
                        'from': i,
                        'to': j,
                        'consensus_score': consensus_matrix[i, j],
                        'methods_agree': sum([
                            var_results['pvalues_by_lag'][0][j, i] < self.alpha if var_results['pvalues_by_lag'] else False,
                            granger_results['significant'][i, j] if granger_results['significant'] is not None else False,
                            te_results['transfer_entropy'][i, j] > te_results['threshold'] if te_results['transfer_entropy'] is not None else False,
                            lagged_corr['significant_corr'][i, j] if lagged_corr['significant_corr'] is not None else False
                        ])
                    })
        
        return {
            'consensus_matrix': consensus_matrix,
            'consensus_edges': consensus_edges,
            'significant_edges': significant_edges,
            'n_edges': len(significant_edges)
        }
    
    def _test_outcome_associations(self, sequences, outcome_data):
        """Test how temporal patterns associate with outcomes"""
        
        n_samples, n_features, n_time = sequences.shape
        
        associations = {}
        
        # Calculate temporal features
        # 1. Slopes (linear trends)
        slopes = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            for j in range(n_features):
                x = np.arange(n_time)
                y = sequences[i, j, :]
                if not np.all(np.isnan(y)):
                    slope, _ = np.polyfit(x, y, 1)
                    slopes[i, j] = slope
        
        # 2. Volatility (standard deviation)
        volatility = np.nanstd(sequences, axis=2)
        
        # 3. Change from baseline
        if n_time >= 2:
            change = sequences[:, :, -1] - sequences[:, :, 0]
        else:
            change = np.zeros((n_samples, n_features))
        
        # Test associations with each outcome
        for outcome_idx in range(outcome_data.shape[1]):
            outcome = outcome_data[:, outcome_idx]
            
            # Remove missing outcomes
            valid = ~np.isnan(outcome)
            
            if valid.sum() < 30:
                continue
            
            associations[f'outcome_{outcome_idx}'] = {
                'slope_associations': [],
                'volatility_associations': [],
                'change_associations': []
            }
            
            # Test each feature
            for j in range(n_features):
                # Slope association
                if not np.all(np.isnan(slopes[valid, j])):
                    lr = LogisticRegression(penalty=None, max_iter=1000)
                    lr.fit(slopes[valid, j].reshape(-1, 1), outcome[valid])
                    coef = lr.coef_[0, 0]
                    
                    # Bootstrap p-value
                    boot_coefs = []
                    for _ in range(100):
                        idx = np.random.choice(valid.sum(), valid.sum(), replace=True)
                        try:
                            lr_boot = LogisticRegression(penalty=None, max_iter=100)
                            lr_boot.fit(slopes[valid, j][idx].reshape(-1, 1), outcome[valid][idx])
                            boot_coefs.append(lr_boot.coef_[0, 0])
                        except:
                            continue
                    
                    if len(boot_coefs) > 50:
                        se = np.std(boot_coefs)
                        p_value = 2 * (1 - stats.norm.cdf(abs(coef / (se + 1e-10))))
                    else:
                        p_value = 1.0
                    
                    associations[f'outcome_{outcome_idx}']['slope_associations'].append({
                        'feature': j,
                        'coefficient': coef,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
        
        return associations
    
    def _identify_temporal_patterns(self, sequences):
        """Identify key temporal patterns in the data"""
        
        n_samples, n_features, n_time = sequences.shape
        
        patterns = {}
        
        # 1. Clustering of temporal trajectories
        from sklearn.cluster import KMeans
        
        # Reshape for clustering
        trajectories = sequences.reshape(n_samples, -1)
        
        # Remove samples with too many missing values
        valid = np.isnan(trajectories).mean(axis=1) < 0.5
        trajectories_valid = trajectories[valid]
        
        if len(trajectories_valid) > 10:
            # Impute remaining missing values
            imputer = SimpleImputer(strategy='mean')
            trajectories_imputed = imputer.fit_transform(trajectories_valid)
            
            # Cluster trajectories
            n_clusters = min(5, len(trajectories_imputed) // 50)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(trajectories_imputed)
                
                patterns['trajectory_clusters'] = {
                    'n_clusters': n_clusters,
                    'cluster_sizes': np.bincount(cluster_labels),
                    'cluster_centers': kmeans.cluster_centers_.reshape(n_clusters, n_features, n_time)
                }
        
        # 2. Identify features with consistent trends
        increasing_features = []
        decreasing_features = []
        stable_features = []
        
        for j in range(n_features):
            # Calculate trend for each sample
            trends = []
            for i in range(n_samples):
                y = sequences[i, j, :]
                if not np.all(np.isnan(y)):
                    x = np.arange(n_time)
                    slope, _ = np.polyfit(x, y, 1)
                    trends.append(slope)
            
            if len(trends) > n_samples * 0.5:
                mean_trend = np.mean(trends)
                se_trend = np.std(trends) / np.sqrt(len(trends))
                
                # Test if significantly different from zero
                t_stat = mean_trend / (se_trend + 1e-10)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(trends) - 1))
                
                if p_value < 0.05:
                    if mean_trend > 0:
                        increasing_features.append(j)
                    else:
                        decreasing_features.append(j)
                else:
                    stable_features.append(j)
        
        patterns['feature_trends'] = {
            'increasing': increasing_features,
            'decreasing': decreasing_features,
            'stable': stable_features
        }
        
        # 3. Autocorrelation patterns
        autocorrelations = np.zeros(n_features)
        for j in range(n_features):
            if n_time >= 2:
                values_t0 = sequences[:, j, :-1].flatten()
                values_t1 = sequences[:, j, 1:].flatten()
                
                valid = ~(np.isnan(values_t0) | np.isnan(values_t1))
                if valid.sum() > 10:
                    autocorrelations[j] = np.corrcoef(values_t0[valid], values_t1[valid])[0, 1]
        
        patterns['autocorrelations'] = {
            'values': autocorrelations,
            'high_autocorr': np.where(autocorrelations > 0.7)[0].tolist(),
            'low_autocorr': np.where(autocorrelations < 0.3)[0].tolist()
        }
        
        return patterns

# =============================================================================
# EXAMPLE ANALYSIS FUNCTION
# =============================================================================

def run_temporal_discovery(temporal_data, outcome_data=None, clinical_data=None,
                          method='var', max_lag=2):
    """Run complete temporal causal discovery"""
    
    print_header("TEMPORAL CAUSAL DISCOVERY")
    
    # Initialize discovery
    discovery = TemporalCausalDiscovery(method=method, max_lag=max_lag)
    
    # Run discovery
    discovery.discover_temporal_causality(
        temporal_data, outcome_data, clinical_data
    )
    
    # Summary
    print(f"\n=== RESULTS SUMMARY ===")
    
    if discovery.results['consensus_edges']['n_edges'] > 0:
        print(f"Consensus edges found: {discovery.results['consensus_edges']['n_edges']}")
        
        # Show top edges
        edges = discovery.results['consensus_edges']['significant_edges']
        if edges:
            print("\nTop temporal causal relationships:")
            for edge in edges[:10]:
                print(f"  Feature {edge['from']} → Feature {edge['to']}: "
                      f"score={edge['consensus_score']:.3f}, "
                      f"methods agree={edge['methods_agree']}/4")
    
    # Save results
    save_results(discovery.results, "temporal_discovery_results.json")
    
    return discovery.results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_header("TEMPORAL CAUSAL DISCOVERY MODULE")
    
    # Load processed data
    from phase3_01_data_processor import ComprehensiveDataProcessor
    
    processor = ComprehensiveDataProcessor()
    processed_data_path = os.path.join(Config.OUTPUT_PATH, 'results', 'processed_data.pkl')
    
    if os.path.exists(processed_data_path):
        print("\nLoading processed data...")
        analysis_data = processor.load_processed_data()
        
        if analysis_data['temporal'] is not None:
            print("\nUsing real temporal metabolomics data")
            
            temporal_data = analysis_data['temporal']
            print(f"  Metabolites: {len(temporal_data['metabolite_names'])}")
            print(f"  Time points: {temporal_data['n_timepoints']}")
            print(f"  Samples: {temporal_data['sequences'].shape[0]}")
            
            # Get outcome data
            outcome_data = analysis_data['outcomes'][['ad_case_primary', 'has_diabetes_any', 'has_obesity_any']].values
            
            # Get clinical data if available
            clinical_data = analysis_data['clinical'].values if analysis_data['clinical'] is not None else None
            
            # Run analysis
            results = run_temporal_discovery(
                temporal_data, 
                outcome_data=outcome_data,
                clinical_data=clinical_data,
                method='var',
                max_lag=temporal_data['n_timepoints'] - 1
            )
            
            print(f"\nAnalysis complete! Results saved to: {Config.OUTPUT_PATH}/results/")
        else:
            print("\nERROR: No temporal data available!")
            print("The cohort may not contain longitudinal metabolomics data")
    else:
        print(f"\nERROR: Processed data not found. Run phase3_01_data_processor.py first")
