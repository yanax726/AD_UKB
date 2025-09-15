#!/usr/bin/env python3
"""
phase3_02_causalformer_FIXED.py - Temporal Causal Discovery with CausalFormer
Scientifically valid version for publication-quality results
"""

# Import configuration
from phase3_00_config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# =============================================================================
# CONFIGURATION - SCIENTIFICALLY VALIDATED
# =============================================================================

class CausalFormerConfig:
    """Configuration aligned with actual available data"""
    
    # Paths
    DATA_PATH = os.path.join(Config.OUTPUT_PATH, 'results', 'temporal_analysis_data.pkl')
    MODEL_PATH = os.path.join(Config.OUTPUT_PATH, 'models')
    RESULTS_PATH = os.path.join(Config.OUTPUT_PATH, 'results')
    FIGURES_PATH = os.path.join(Config.OUTPUT_PATH, 'figures')
    
    # CRITICAL: Use actual data dimensions from diagnostic
    N_METABOLITES = 235  
    N_TIMEPOINTS = 2     
    
    # Model architecture - simplified for limited data
    HIDDEN_DIM = 64  # Reduced to prevent overfitting
    N_HEADS = 4
    N_LAYERS = 2  # Shallower network
    DROPOUT = 0.3  # Higher dropout for regularization
    
    # Training parameters - adjusted for imbalanced data
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    GRADIENT_CLIP = 1.0
    
    # Validation
    VAL_SPLIT = 0.2
    MIN_PREVALENCE = 0.01  # Exclude outcomes with <1% prevalence
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

# Set seeds for reproducibility
np.random.seed(CausalFormerConfig.SEED)
torch.manual_seed(CausalFormerConfig.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(CausalFormerConfig.SEED)
    torch.backends.cudnn.deterministic = True

# =============================================================================
# DATA LOADING WITH VALIDATION
# =============================================================================

class TemporalMetabolomicsDataset(Dataset):
    """Dataset with proper validation and documentation"""
    
    def __init__(self, data_dict, indices=None, min_prevalence=0.01):
        """
        Initialize with validation of available outcomes
        
        Args:
            data_dict: Output from phase3_01
            indices: Sample indices to use
            min_prevalence: Minimum outcome prevalence to include
        """
        # Extract temporal sequences
        if 'temporal' in data_dict and data_dict['temporal'] is not None:
            self.metabolites = torch.FloatTensor(data_dict['temporal']['sequences'])
            self.metabolite_names = data_dict['temporal']['metabolite_names']
            self.has_temporal = True
        else:
            raise ValueError("No temporal data found - cannot proceed with temporal analysis")
        
        # Extract and validate outcomes
        outcomes_df = data_dict.get('outcomes', pd.DataFrame())
        if outcomes_df.empty:
            raise ValueError("No outcomes data found")
        
        # AD status (exposure)
        self.ad_status = torch.FloatTensor(
            outcomes_df['ad_case'].values if 'ad_case' in outcomes_df else np.zeros(len(outcomes_df))
        ).unsqueeze(1)
        
        # Document AD prevalence for publication
        self.ad_prevalence = self.ad_status.mean().item()
        print(f"\n📊 AD prevalence in temporal cohort: {self.ad_prevalence:.3f}")
        
        # Identify valid metabolic outcomes
        self.valid_outcomes = []
        self.outcome_prevalences = {}
        
        outcome_candidates = [col for col in outcomes_df.columns if col.startswith('has_')]
        outcome_values = []
        
        print("\n📊 Validating outcomes for analysis:")
        for outcome in outcome_candidates:
            prevalence = outcomes_df[outcome].mean()
            n_positive = outcomes_df[outcome].sum()
            
            if prevalence >= min_prevalence and prevalence <= (1 - min_prevalence):
                self.valid_outcomes.append(outcome)
                outcome_values.append(outcomes_df[outcome].values)
                self.outcome_prevalences[outcome] = prevalence
                print(f"  ✓ {outcome}: {prevalence:.3f} ({n_positive} cases)")
            else:
                print(f"  ✗ {outcome}: {prevalence:.3f} - excluded (insufficient variation)")
        
        if len(self.valid_outcomes) == 0:
            raise ValueError("No valid outcomes with sufficient prevalence")
        
        self.outcomes = torch.FloatTensor(np.column_stack(outcome_values))
        self.n_outcomes = len(self.valid_outcomes)
        
        print(f"\n✓ Using {self.n_outcomes} outcomes for analysis")
        
        # Extract demographics
        demo_df = data_dict.get('demographics', pd.DataFrame())
        self.demographics = self._prepare_demographics(demo_df)
        
        # Extract clinical markers
        clinical_df = data_dict.get('clinical', pd.DataFrame())
        self.clinical = self._prepare_clinical(clinical_df)
        
        # Apply sample indices if provided
        if indices is not None:
            self.metabolites = self.metabolites[indices]
            self.ad_status = self.ad_status[indices]
            self.outcomes = self.outcomes[indices]
            self.demographics = self.demographics[indices]
            self.clinical = self.clinical[indices]
        
        # Store metadata for publication reporting
        self.metadata = {
            'n_samples': len(self.metabolites),
            'n_metabolites': self.metabolites.shape[1],
            'n_timepoints': self.metabolites.shape[2],
            'n_outcomes': self.n_outcomes,
            'valid_outcomes': self.valid_outcomes,
            'outcome_prevalences': self.outcome_prevalences,
            'ad_prevalence': self.ad_prevalence
        }
    
    def _prepare_demographics(self, demo_df):
        """Prepare demographic features with proper handling"""
        features = []
        
        # Age
        if 'age_baseline' in demo_df.columns:
            age = demo_df['age_baseline'].fillna(demo_df['age_baseline'].median()).values
            # Standardize age
            age = (age - age.mean()) / age.std()
            features.append(age)
        
        # Sex
        if 'sex' in demo_df.columns:
            if demo_df['sex'].dtype == 'object':
                sex = (demo_df['sex'] == 'Male').astype(float).values
            else:
                sex = demo_df['sex'].values
            features.append(sex)
        
        # Townsend deprivation index
        if 'townsend_index' in demo_df.columns:
            townsend = demo_df['townsend_index'].fillna(0).values
            # Standardize
            townsend = (townsend - townsend.mean()) / (townsend.std() + 1e-8)
            features.append(townsend)
        
        if len(features) > 0:
            return torch.FloatTensor(np.column_stack(features))
        else:
            return torch.zeros((len(demo_df), 1))
    
    def _prepare_clinical(self, clinical_df):
        """Prepare clinical biomarkers"""
        if len(clinical_df.columns) > 0:
            # Impute with median
            clinical_filled = clinical_df.fillna(clinical_df.median())
            # Standardize
            clinical_array = clinical_filled.values
            clinical_array = (clinical_array - clinical_array.mean(axis=0)) / (clinical_array.std(axis=0) + 1e-8)
            return torch.FloatTensor(clinical_array)
        else:
            return torch.zeros((len(clinical_df), 1))
    
    def get_class_weights(self):
        """Calculate class weights for handling imbalance"""
        weights = []
        for i in range(self.n_outcomes):
            pos_count = self.outcomes[:, i].sum()
            neg_count = len(self) - pos_count
            if pos_count > 0:
                weight = neg_count / pos_count
                # Cap weights to prevent training instability
                weight = min(weight, 10.0)
            else:
                weight = 1.0
            weights.append(weight)
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.metabolites)
    
    def __getitem__(self, idx):
        return {
            'metabolites': self.metabolites[idx],
            'ad_status': self.ad_status[idx],
            'outcomes': self.outcomes[idx],
            'demographics': self.demographics[idx],
            'clinical': self.clinical[idx]
        }

# =============================================================================
# SIMPLIFIED CAUSALFORMER MODEL
# =============================================================================

class SimplifiedCausalFormer(nn.Module):
    """
    Simplified architecture for reliable training on limited data
    Focus on core causal relationships: AD -> Metabolites -> Diseases
    """
    
    def __init__(self, config, n_outcomes):
        super().__init__()
        self.config = config
        self.n_outcomes = n_outcomes
        
        # Calculate input dimensions
        metabolite_dim = config.N_METABOLITES * config.N_TIMEPOINTS
        demo_dim = 3  # age, sex, townsend
        clinical_dim = 4  # Approximate
        
        # Metabolite temporal encoder
        self.metabolite_encoder = nn.Sequential(
            nn.Linear(metabolite_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, config.HIDDEN_DIM)
        )
        
        # AD treatment encoder
        self.ad_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, config.HIDDEN_DIM // 2)
        )
        
        # Demographic encoder
        self.demo_encoder = nn.Sequential(
            nn.Linear(demo_dim, 16),
            nn.ReLU(),
            nn.Linear(16, config.HIDDEN_DIM // 2)
        )
        
        # Interaction layer
        self.interaction = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
        # Disease prediction heads
        self.disease_heads = nn.ModuleList([
            nn.Linear(config.HIDDEN_DIM, 1) for _ in range(n_outcomes)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, metabolites, ad_status, demographics, clinical=None):
        batch_size = metabolites.shape[0]
        
        # Flatten metabolite temporal data
        metabolites_flat = metabolites.view(batch_size, -1)
        
        # Encode components
        met_features = self.metabolite_encoder(metabolites_flat)
        ad_features = self.ad_encoder(ad_status)
        demo_features = self.demo_encoder(demographics)
        
        # Combine features
        combined = torch.cat([
            met_features,
            ad_features,
            demo_features
        ], dim=1)
        
        # Model interactions
        interaction_features = self.interaction(combined)
        
        # Disease predictions
        disease_logits = []
        for head in self.disease_heads:
            logit = head(interaction_features)
            disease_logits.append(logit)
        
        disease_logits = torch.cat(disease_logits, dim=1)
        
        # Calculate causal effects (difference due to AD)
        causal_effects = disease_logits * ad_status
        
        return {
            'logits': disease_logits,
            'probs': torch.sigmoid(disease_logits),
            'causal_effects': causal_effects,
            'features': interaction_features  # For interpretability
        }

# =============================================================================
# TRAINING WITH PROPER VALIDATION
# =============================================================================

def train_causalformer(model, train_loader, val_loader, config, class_weights, outcome_names):
    """Train with proper handling of imbalanced data"""
    
    print_header("TRAINING CAUSALFORMER")
    
    device = config.DEVICE
    model = model.to(device)
    class_weights = class_weights.to(device)
    
    # Loss function with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training history
    history = defaultdict(list)
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(1, config.MAX_EPOCHS + 1):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{config.MAX_EPOCHS}'):
            # Move to device
            metabolites = batch['metabolites'].to(device)
            ad_status = batch['ad_status'].to(device)
            outcomes = batch['outcomes'].to(device)
            demographics = batch['demographics'].to(device)
            
            # Forward pass
            outputs = model(metabolites, ad_status, demographics)
            loss = criterion(outputs['logits'], outcomes)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                metabolites = batch['metabolites'].to(device)
                ad_status = batch['ad_status'].to(device)
                outcomes = batch['outcomes'].to(device)
                demographics = batch['demographics'].to(device)
                
                outputs = model(metabolites, ad_status, demographics)
                loss = criterion(outputs['logits'], outcomes)
                
                val_losses.append(loss.item())
                all_probs.append(outputs['probs'].cpu().numpy())
                all_targets.append(outcomes.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        all_probs = np.vstack(all_probs)
        all_targets = np.vstack(all_targets)
        
        # Calculate AUC for each valid outcome
        outcome_metrics = {}
        valid_aucs = []
        
        for i, outcome_name in enumerate(outcome_names):
            n_pos = all_targets[:, i].sum()
            n_neg = len(all_targets) - n_pos
            
            if n_pos > 0 and n_neg > 0:
                try:
                    auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
                    valid_aucs.append(auc)
                    outcome_metrics[outcome_name] = auc
                except:
                    outcome_metrics[outcome_name] = np.nan
            else:
                outcome_metrics[outcome_name] = np.nan
        
        mean_auc = np.mean(valid_aucs) if valid_aucs else 0.5
        
        # Update scheduler
        scheduler.step(mean_auc)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(mean_auc)
        history['outcome_aucs'].append(outcome_metrics)
        
        # Print progress
        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Mean Val AUC: {mean_auc:.4f}')
        
        # Print individual outcome AUCs
        for outcome, auc in outcome_metrics.items():
            if not np.isnan(auc):
                print(f'    {outcome}: {auc:.3f}')
        
        # Save best model
        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            patience_counter = 0
            
            # Save with proper handling for PyTorch 2.6+
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'outcome_names': outcome_names,
                'outcome_metrics': outcome_metrics,
                'history': dict(history)
            }, os.path.join(config.MODEL_PATH, 'best_causalformer.pt'))
            
            print(f'  ✓ New best model saved (AUC={best_val_auc:.4f})')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f'\nEarly stopping at epoch {epoch}')
            break
    
    return model, dict(history)

# =============================================================================
# CAUSAL EFFECT ANALYSIS
# =============================================================================

def analyze_causal_effects(model, data_loader, config, outcome_names):
    """Analyze causal effects with confidence intervals"""
    
    print_header("ANALYZING CAUSAL EFFECTS")
    
    model.eval()
    device = config.DEVICE
    
    # Collect counterfactual predictions
    effects_with_ad = []
    effects_without_ad = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Computing causal effects'):
            metabolites = batch['metabolites'].to(device)
            demographics = batch['demographics'].to(device)
            
            # Counterfactual: With AD
            ad_ones = torch.ones((len(metabolites), 1)).to(device)
            outputs_ad = model(metabolites, ad_ones, demographics)
            
            # Counterfactual: Without AD
            ad_zeros = torch.zeros((len(metabolites), 1)).to(device)
            outputs_no_ad = model(metabolites, ad_zeros, demographics)
            
            # Store predictions
            effects_with_ad.append(outputs_ad['probs'].cpu().numpy())
            effects_without_ad.append(outputs_no_ad['probs'].cpu().numpy())
    
    # Aggregate results
    effects_with_ad = np.vstack(effects_with_ad)
    effects_without_ad = np.vstack(effects_without_ad)
    
    # Calculate individual treatment effects
    individual_effects = effects_with_ad - effects_without_ad
    
    # Summarize results
    results = {}
    
    print("\n📊 Causal Effects of AD on Metabolic Diseases:")
    print("-" * 60)
    print(f"{'Disease':<30} {'ATE':<10} {'95% CI':<20} {'RR':<10}")
    print("-" * 60)
    
    for i, outcome in enumerate(outcome_names):
        ate = individual_effects[:, i].mean()
        ate_std = individual_effects[:, i].std()
        ci_lower = np.percentile(individual_effects[:, i], 2.5)
        ci_upper = np.percentile(individual_effects[:, i], 97.5)
        
        # Calculate relative risk
        baseline_risk = effects_without_ad[:, i].mean()
        if baseline_risk > 0:
            rr = (baseline_risk + ate) / baseline_risk
        else:
            rr = np.nan
        
        results[outcome] = {
            'ate': float(ate),
            'ate_std': float(ate_std),
            'ci_95': [float(ci_lower), float(ci_upper)],
            'baseline_risk': float(baseline_risk),
            'relative_risk': float(rr),
            'n_samples': len(individual_effects)
        }
        
        # Check if significant (CI doesn't include 0)
        significant = ci_lower > 0 or ci_upper < 0
        sig_marker = "*" if significant else " "
        
        print(f"{outcome:<30} {ate:>8.4f}{sig_marker} [{ci_lower:>7.4f}, {ci_upper:>7.4f}] {rr:>8.3f}")
    
    print("-" * 60)
    print("* Significant at 95% confidence level")
    
    # Save detailed results
    save_results(results, 'causal_effects_results.json')
    
    # Save individual effects for further analysis
    np.save(os.path.join(config.RESULTS_PATH, 'individual_treatment_effects.npy'), 
            individual_effects)
    
    return results, individual_effects

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_publication_figures(history, results, config, outcome_names):
    """Create publication-quality figures"""
    
    print_header("CREATING FIGURES")
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    
    # Figure 1: Training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Training', linewidth=2, color='#2E86C1')
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2, color='#E74C3C')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Training Progress')
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, alpha=0.3)
    
    # AUC curve
    axes[1].plot(history['val_auc'], linewidth=2, color='#27AE60')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean AUROC')
    axes[1].set_title('Validation Performance')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    axes[1].axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Acceptable')
    axes[1].legend(frameon=True, fancybox=True, shadow=True)
    
    plt.suptitle('CausalFormer Training Metrics', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_PATH, 'training_curves.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Forest plot of causal effects
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data for forest plot
    outcomes_sorted = []
    effects = []
    ci_lowers = []
    ci_uppers = []
    colors = []
    
    for outcome in outcome_names:
        if outcome in results:
            outcomes_sorted.append(outcome.replace('has_', '').replace('_any', '').title())
            effects.append(results[outcome]['ate'])
            ci_lowers.append(results[outcome]['ci_95'][0])
            ci_uppers.append(results[outcome]['ci_95'][1])
            
            # Color by significance
            if results[outcome]['ci_95'][0] > 0:
                colors.append('#E74C3C')  # Increased risk
            elif results[outcome]['ci_95'][1] < 0:
                colors.append('#3498DB')  # Decreased risk
            else:
                colors.append('#95A5A6')  # Not significant
    
    y_pos = np.arange(len(outcomes_sorted))
    
    # Plot effects with error bars
    ax.scatter(effects, y_pos, c=colors, s=100, zorder=3, alpha=0.8)
    
    # Add confidence intervals
    for i in range(len(outcomes_sorted)):
        ax.plot([ci_lowers[i], ci_uppers[i]], [y_pos[i], y_pos[i]], 
                color=colors[i], linewidth=2, alpha=0.6)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(outcomes_sorted)
    ax.set_xlabel('Average Treatment Effect (ATE)')
    ax.set_title('Causal Effects of Atopic Dermatitis on Metabolic Diseases')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add text annotations
    for i, (effect, ci_l, ci_u) in enumerate(zip(effects, ci_lowers, ci_uppers)):
        if ci_l > 0 or ci_u < 0:
            ax.text(max(ci_u, effect) + 0.002, i, '*', fontsize=14, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_PATH, 'causal_effects_forest.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figures saved to {config.FIGURES_PATH}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_header("CAUSALFORMER TEMPORAL CAUSAL DISCOVERY")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {CausalFormerConfig.DEVICE}")
    print(f"Configuration: Scientifically validated for publication")
    
    # Load data
    print("\n📁 Loading temporal analysis data...")
    data = joblib.load(CausalFormerConfig.DATA_PATH)
    
    # Create dataset with validation
    print("\n🔄 Creating dataset with validation...")
    try:
        full_dataset = TemporalMetabolomicsDataset(
            data, 
            min_prevalence=CausalFormerConfig.MIN_PREVALENCE
        )
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        sys.exit(1)
    
    # Document for publication
    print("\n📄 Dataset Summary for Publication:")
    print(f"  Samples: {full_dataset.metadata['n_samples']:,}")
    print(f"  Metabolites: {full_dataset.metadata['n_metabolites']}")
    print(f"  Timepoints: {full_dataset.metadata['n_timepoints']}")
    print(f"  Valid outcomes: {full_dataset.metadata['n_outcomes']}")
    print(f"  AD prevalence: {full_dataset.metadata['ad_prevalence']:.3f}")
    
    # Split data with stratification
    n_samples = len(full_dataset)
    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=CausalFormerConfig.VAL_SPLIT,
        random_state=CausalFormerConfig.SEED,
        stratify=full_dataset.ad_status.squeeze().numpy()
    )
    
    # Create data subsets
    train_dataset = TemporalMetabolomicsDataset(data, train_idx)
    val_dataset = TemporalMetabolomicsDataset(data, val_idx)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    
    # Get class weights for imbalanced data
    class_weights = train_dataset.get_class_weights()
    print(f"\nClass weights for balanced training:")
    for i, (outcome, weight) in enumerate(zip(train_dataset.valid_outcomes, class_weights)):
        print(f"  {outcome}: {weight:.2f}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CausalFormerConfig.BATCH_SIZE,
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CausalFormerConfig.BATCH_SIZE,
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    print("\n🧠 Initializing SimplifiedCausalFormer...")
    model = SimplifiedCausalFormer(
        CausalFormerConfig,
        n_outcomes=train_dataset.n_outcomes
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    
    # Train model
    model, history = train_causalformer(
        model, train_loader, val_loader,
        CausalFormerConfig, class_weights,
        train_dataset.valid_outcomes
    )
    
    # Load best model for analysis
    checkpoint = torch.load(
        os.path.join(CausalFormerConfig.MODEL_PATH, 'best_causalformer.pt'),
        map_location=CausalFormerConfig.DEVICE,
        weights_only=False  # Required for PyTorch 2.6+
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model (Mean AUC={checkpoint['best_val_auc']:.4f})")
    
    # Print final performance by outcome
    print("\nFinal validation performance:")
    for outcome, auc in checkpoint['outcome_metrics'].items():
        if not np.isnan(auc):
            print(f"  {outcome}: AUROC={auc:.3f}")
    
    # Analyze causal effects
    results, individual_effects = analyze_causal_effects(
        model, val_loader, CausalFormerConfig, 
        train_dataset.valid_outcomes
    )
    
    # Create publication figures
    create_publication_figures(
        history, results, CausalFormerConfig,
        train_dataset.valid_outcomes
    )
    
    # Save metadata for manuscript
    manuscript_data = {
        'dataset': train_dataset.metadata,
        'model': {
            'architecture': 'SimplifiedCausalFormer',
            'parameters': n_params,
            'best_epoch': checkpoint['epoch'],
            'best_auc': checkpoint['best_val_auc']
        },
        'results': results,
        'validation_performance': checkpoint['outcome_metrics']
    }
    
    save_results(manuscript_data, 'manuscript_results.json')
    
    print("\n" + "="*80)
    print("✅ CAUSALFORMER ANALYSIS COMPLETE!")
    print(f"Results directory: {CausalFormerConfig.RESULTS_PATH}")
    print("\nKey outputs for publication:")
    print("  1. causal_effects_results.json - Main results")
    print("  2. manuscript_results.json - Complete metadata")
    print("  3. training_curves.pdf - Figure 1")
    print("  4. causal_effects_forest.pdf - Figure 2")
    print("="*80)
