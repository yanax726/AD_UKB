#!/usr/bin/env python3
"""
phase3_validation_final.py - Complete Validation Pipeline with Progress Tracking
Run this AFTER CausalFormer to validate all results for publication
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Progress bar handling
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")
    
    # Fallback progress tracker
    class SimpleProgress:
        def __init__(self, total, desc=""):
            self.total = total
            self.current = 0
            self.desc = desc
            print(f"\n{desc}")
            
        def update(self, n=1):
            self.current += n
            pct = int(100 * self.current / self.total)
            print(f"  Progress: {pct}% ({self.current}/{self.total})", end='\r')
            
        def close(self):
            print()  # New line after completion

# ================================================================================
# CONFIGURATION
# ================================================================================

class ValidationConfig:
    """Configuration for validation"""
    CAUSALFORMER_RESULTS = "./causalformer_results"
    PHASE3_RESULTS = "./phase3_comprehensive_publication/results"
    OUTPUT_DIR = "./validation_report"
    
    # Thresholds for publication quality
    MIN_AUC = 0.65
    MIN_EFFECT_SIZE = 0.01
    MIN_SIGNIFICANT_PATHS = 5
    MIN_POWER = 0.80
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================================
# MAIN VALIDATOR
# ================================================================================

class ComprehensiveValidator:
    """Complete validation with progress tracking"""
    
    def __init__(self):
        self.config = ValidationConfig()
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'issues': [],
            'recommendations': []
        }
        self.progress = None
        
    def run_validation(self):
        """Run complete validation pipeline"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION FOR PUBLICATION")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Total validation steps
        total_steps = 8
        
        if TQDM_AVAILABLE:
            self.progress = tqdm(total=total_steps, desc="Overall Validation", 
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            self.progress = SimpleProgress(total_steps, "Overall Validation")
        
        try:
            # Step 1: Check CausalFormer Results
            self._validate_step("CausalFormer Results", self._check_causalformer)
            
            # Step 2: Check Data Quality
            self._validate_step("Data Quality", self._check_data_quality)
            
            # Step 3: Check Model Performance
            self._validate_step("Model Performance", self._check_model_performance)
            
            # Step 4: Check Statistical Significance
            self._validate_step("Statistical Significance", self._check_significance)
            
            # Step 5: Check Effect Sizes
            self._validate_step("Effect Sizes", self._check_effect_sizes)
            
            # Step 6: Check Biological Plausibility
            self._validate_step("Biological Plausibility", self._check_biological_plausibility)
            
            # Step 7: Check Statistical Power
            self._validate_step("Statistical Power", self._check_power)
            
            # Step 8: Generate Report
            self._validate_step("Report Generation", self._generate_report)
            
        finally:
            if self.progress:
                self.progress.close()
        
        # Summary
        self._print_summary()
        
        return len(self.report['issues']) == 0
    
    def _validate_step(self, step_name, validation_func):
        """Execute validation step with progress tracking"""
        
        print(f"\n{'='*60}")
        print(f"VALIDATING: {step_name}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            result = validation_func()
            elapsed = time.time() - start_time
            
            status = "✅ PASSED" if result else "⚠️ ISSUES FOUND"
            print(f"\n{status} (took {elapsed:.2f}s)")
            
            self.report['checks'][step_name] = {
                'passed': result,
                'duration': elapsed
            }
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            self.report['checks'][step_name] = {
                'passed': False,
                'error': str(e)
            }
            result = False
        
        if self.progress:
            self.progress.update(1)
        
        return result
    
    def _check_causalformer(self):
        """Validate CausalFormer results"""
        
        results_file = os.path.join(self.config.CAUSALFORMER_RESULTS, 
                                   "causal_analysis_results.json")
        
        if not os.path.exists(results_file):
            self.report['issues'].append("CausalFormer results not found")
            print("  ❌ Results file not found")
            return False
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("\n📊 CausalFormer Metrics:")
        print("-" * 40)
        
        # Check AUCs
        aucs = results.get('disease_aucs', {})
        low_auc_diseases = []
        
        for disease, auc in aucs.items():
            status = "✓" if auc >= self.config.MIN_AUC else "✗"
            print(f"  {status} {disease:15s} AUC: {auc:.3f}")
            
            if auc < self.config.MIN_AUC:
                low_auc_diseases.append(disease)
        
        if low_auc_diseases:
            self.report['issues'].append(f"Low AUC for: {', '.join(low_auc_diseases)}")
        
        # Check causal effects
        effects = results.get('causal_effects', {})
        n_significant = 0
        
        print("\n📈 Causal Effects:")
        print("-" * 40)
        
        for disease, effect_data in effects.items():
            ci = effect_data.get('ci_95', [0, 0])
            mean_effect = effect_data.get('mean', 0)
            
            # Check if CI excludes zero
            is_sig = (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)
            
            if is_sig:
                n_significant += 1
            
            status = "✓" if is_sig else "✗"
            print(f"  {status} {disease:15s} β={mean_effect:+.4f} "
                  f"CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
        
        # Check pathway weights diversity
        mediators = results.get('top_mediators', [])
        unique_mediators = len(set(mediators))
        
        if unique_mediators < len(mediators) * 0.9:
            self.report['issues'].append("Low diversity in pathway weights")
            print(f"\n  ⚠️ Low mediator diversity: {unique_mediators}/{len(mediators)} unique")
        
        return n_significant >= 2 and len(low_auc_diseases) == 0
    
    def _check_data_quality(self):
        """Check data quality metrics"""
        
        print("\n📊 Data Quality Checks:")
        print("-" * 40)
        
        # Check if processed data exists
        data_file = os.path.join(self.config.PHASE3_RESULTS, "processed_data.pkl")
        
        if os.path.exists(data_file):
            import joblib
            data = joblib.load(data_file)
            
            # Check sample size
            n_samples = data['metadata'].get('n_samples', 0)
            print(f"  Sample size: {n_samples:,}")
            
            if n_samples < 10000:
                self.report['issues'].append(f"Small sample size: {n_samples}")
            
            # Check feature completeness
            n_features = data['metadata'].get('n_features', 0)
            print(f"  Total features: {n_features}")
            
            # Check temporal data
            has_temporal = data['metadata'].get('has_temporal', False)
            print(f"  Temporal data: {'Yes ✓' if has_temporal else 'No ✗'}")
            
            if not has_temporal:
                self.report['recommendations'].append(
                    "Consider using temporal data for stronger causal inference"
                )
            
            return n_samples >= 10000
        else:
            print("  ⚠️ Processed data not found")
            return False
    
    def _check_model_performance(self):
        """Check model training performance"""
        
        print("\n📊 Model Performance:")
        print("-" * 40)
        
        checkpoint_file = os.path.join(self.config.CAUSALFORMER_RESULTS, "best_model.pt")
        
        if os.path.exists(checkpoint_file):
            import torch
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            
            # Check training history
            history = checkpoint.get('history', {})
            
            if history:
                # Check for overfitting
                train_losses = history.get('train_loss', [])
                val_losses = history.get('val_loss', [])
                
                if train_losses and val_losses:
                    final_train = train_losses[-1]
                    final_val = val_losses[-1]
                    
                    gap = final_val - final_train
                    print(f"  Final train loss: {final_train:.4f}")
                    print(f"  Final val loss: {final_val:.4f}")
                    print(f"  Generalization gap: {gap:.4f}")
                    
                    if gap > 0.5:
                        self.report['issues'].append(f"Possible overfitting: gap={gap:.4f}")
                
                # Check convergence
                if len(val_losses) > 5:
                    recent_improvement = val_losses[-5] - val_losses[-1]
                    print(f"  Recent improvement: {recent_improvement:.4f}")
                    
                    if recent_improvement < 0.001:
                        self.report['recommendations'].append(
                            "Model may benefit from longer training or different hyperparameters"
                        )
            
            return True
        else:
            print("  ⚠️ Model checkpoint not found")
            return False
    
    def _check_significance(self):
        """Check statistical significance of results"""
        
        print("\n📊 Statistical Significance:")
        print("-" * 40)
        
        # Load results
        results_file = os.path.join(self.config.CAUSALFORMER_RESULTS, 
                                   "causal_analysis_results.json")
        
        if not os.path.exists(results_file):
            return False
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Multiple testing correction
        effects = results.get('causal_effects', {})
        p_values = []
        
        for disease, effect_data in effects.items():
            # Calculate p-value from CI
            mean_effect = effect_data.get('mean', 0)
            ci = effect_data.get('ci_95', [0, 0])
            
            # Approximate SE from CI
            se = (ci[1] - ci[0]) / (2 * 1.96) if ci[1] != ci[0] else 1
            z_score = abs(mean_effect) / (se + 1e-10)
            p_val = 2 * (1 - stats.norm.cdf(z_score))
            p_values.append(p_val)
        
        # Apply FDR correction
        from statsmodels.stats.multitest import multipletests
        if p_values:
            reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            
            n_sig_fdr = sum(reject)
            print(f"  Significant after FDR: {n_sig_fdr}/{len(p_values)}")
            
            for i, (disease, p_val, p_adj, sig) in enumerate(
                zip(effects.keys(), p_values, pvals_corrected, reject)
            ):
                status = "✓" if sig else "✗"
                print(f"    {status} {disease}: p={p_val:.3f}, q={p_adj:.3f}")
            
            if n_sig_fdr < 1:
                self.report['issues'].append("No significant effects after FDR correction")
            
            return n_sig_fdr >= 1
        
        return False
    
    def _check_effect_sizes(self):
        """Check if effect sizes are meaningful"""
        
        print("\n📊 Effect Size Analysis:")
        print("-" * 40)
        
        results_file = os.path.join(self.config.CAUSALFORMER_RESULTS,
                                   "causal_analysis_results.json")
        
        if not os.path.exists(results_file):
            return False
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        effects = results.get('causal_effects', {})
        
        # Calculate standardized effect sizes
        meaningful_effects = 0
        
        for disease, effect_data in effects.items():
            mean_effect = abs(effect_data.get('mean', 0))
            rr = effect_data.get('relative_risk', 1)
            
            # Check if effect is meaningful
            is_meaningful = mean_effect >= self.config.MIN_EFFECT_SIZE
            
            if is_meaningful:
                meaningful_effects += 1
            
            status = "✓" if is_meaningful else "✗"
            print(f"  {status} {disease:15s} |β|={mean_effect:.4f}, RR={rr:.2f}")
        
        print(f"\nMeaningful effects: {meaningful_effects}/{len(effects)}")
        
        if meaningful_effects < len(effects) * 0.5:
            self.report['recommendations'].append(
                "Consider focusing on diseases with larger effect sizes"
            )
        
        return meaningful_effects >= 2
    
    def _check_biological_plausibility(self):
        """Check biological plausibility of findings"""
        
        print("\n📊 Biological Plausibility:")
        print("-" * 40)
        
        # Known associations from literature
        expected = {
            'Diabetes': {'direction': 'positive', 'strength': 'moderate'},
            'Hypertension': {'direction': 'positive', 'strength': 'weak'},
            'Obesity': {'direction': 'positive', 'strength': 'moderate'},
            'Hyperlipidemia': {'direction': 'positive', 'strength': 'moderate'}
        }
        
        results_file = os.path.join(self.config.CAUSALFORMER_RESULTS,
                                   "causal_analysis_results.json")
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            effects = results.get('causal_effects', {})
            
            consistent = 0
            for disease, effect_data in effects.items():
                mean_effect = effect_data.get('mean', 0)
                
                if disease in expected:
                    expected_dir = expected[disease]['direction']
                    
                    is_consistent = (
                        (expected_dir == 'positive' and mean_effect > 0) or
                        (expected_dir == 'negative' and mean_effect < 0)
                    )
                    
                    if is_consistent:
                        consistent += 1
                    
                    status = "✓" if is_consistent else "✗"
                    print(f"  {status} {disease}: "
                          f"Expected={expected_dir}, Observed={'positive' if mean_effect > 0 else 'negative'}")
            
            consistency_rate = consistent / len(expected) if expected else 0
            print(f"\nConsistency with literature: {consistency_rate:.0%}")
            
            if consistency_rate < 0.5:
                self.report['issues'].append(
                    "Results inconsistent with established literature"
                )
            
            return consistency_rate >= 0.5
        
        return False
    
    def _check_power(self):
        """Check statistical power"""
        
        print("\n📊 Statistical Power Analysis:")
        print("-" * 40)
        
        # Estimate from sample size
        n_samples = 226382  # From your data
        
        # Post-hoc power for observed effects
        from statsmodels.stats.power import tt_solve_power
        
        # Small effect size (Cohen's d = 0.2)
        power_small = tt_solve_power(
            effect_size=0.2,
            nobs=n_samples,
            alpha=0.05,
            power=None,
            alternative='two-sided'
        )
        
        # Medium effect size (Cohen's d = 0.5)
        power_medium = tt_solve_power(
            effect_size=0.5,
            nobs=n_samples,
            alpha=0.05,
            power=None,
            alternative='two-sided'
        )
        
        print(f"  Sample size: {n_samples:,}")
        print(f"  Power for d=0.2: {power_small:.0%}")
        print(f"  Power for d=0.5: {power_medium:.0%}")
        
        adequate = power_small >= self.config.MIN_POWER
        
        if not adequate:
            self.report['recommendations'].append(
                "Consider increasing sample size or focusing on larger effects"
            )
        
        return adequate
    
    def _generate_report(self):
        """Generate comprehensive validation report"""
        
        print("\n📝 Generating Validation Report...")
        
        # Calculate overall score
        n_passed = sum(1 for check in self.report['checks'].values() 
                      if check.get('passed', False))
        n_total = len(self.report['checks'])
        
        self.report['summary'] = {
            'checks_passed': n_passed,
            'checks_total': n_total,
            'pass_rate': n_passed / n_total if n_total > 0 else 0,
            'n_issues': len(self.report['issues']),
            'n_recommendations': len(self.report['recommendations']),
            'publication_ready': n_passed == n_total and len(self.report['issues']) == 0
        }
        
        # Save report
        report_file = os.path.join(self.config.OUTPUT_DIR, 'validation_report.json')
        with open(report_file, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        
        print(f"  ✅ Report saved to: {report_file}")
        
        # Create summary figure
        self._create_summary_figure()
        
        return True
    
    def _create_summary_figure(self):
        """Create visual summary of validation"""
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Validation checks pie chart
        checks = self.report['checks']
        passed = sum(1 for c in checks.values() if c.get('passed', False))
        failed = len(checks) - passed
        
        ax1.pie([passed, failed], labels=['Passed', 'Failed'],
                colors=['#90EE90', '#FFB6C1'], autopct='%1.0f%%',
                startangle=90)
        ax1.set_title('Validation Checks', fontweight='bold')
        
        # Issues and recommendations bar chart
        categories = ['Issues', 'Recommendations']
        counts = [len(self.report['issues']), len(self.report['recommendations'])]
        
        bars = ax2.bar(categories, counts, color=['#FF6B6B', '#4ECDC4'])
        ax2.set_ylabel('Count')
        ax2.set_title('Problems Identified', fontweight='bold')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        plt.suptitle('Validation Summary Report', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fig_file = os.path.join(self.config.OUTPUT_DIR, 'validation_summary.png')
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Figure saved to: {fig_file}")
    
    def _print_summary(self):
        """Print final summary"""
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        summary = self.report['summary']
        
        print(f"\n✅ Checks Passed: {summary['checks_passed']}/{summary['checks_total']} "
              f"({summary['pass_rate']:.0%})")
        
        if self.report['issues']:
            print(f"\n⚠️ Issues Found ({len(self.report['issues'])}):")
            for i, issue in enumerate(self.report['issues'], 1):
                print(f"  {i}. {issue}")
        
        if self.report['recommendations']:
            print(f"\n💡 Recommendations ({len(self.report['recommendations'])}):")
            for i, rec in enumerate(self.report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        
        if summary['publication_ready']:
            print("🎉 PUBLICATION READY!")
            print("Your results meet publication quality standards.")
        else:
            print("📋 ADDITIONAL WORK NEEDED")
            print("Please address the issues above before publication.")
        
        print("="*80)
        print(f"\nFull report saved to: {self.config.OUTPUT_DIR}/validation_report.json")

# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Run validation pipeline"""
    
    validator = ComprehensiveValidator()
    success = validator.run_validation()
    
    if success:
        print("\n✅ Validation completed successfully!")
    else:
        print("\n⚠️ Validation completed with issues - see report")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
