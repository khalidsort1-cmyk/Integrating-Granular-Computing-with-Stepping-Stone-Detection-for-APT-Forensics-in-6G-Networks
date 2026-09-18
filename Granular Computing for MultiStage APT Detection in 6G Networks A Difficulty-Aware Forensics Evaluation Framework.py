import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from scipy.stats import entropy
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

# ---------------- Optional dependencies ----------------
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️ SMOTE not available. Install with: pip install imbalanced-learn")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Install with: pip install optuna")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install with: pip install torch")

SKLEARN_BASELINES_AVAILABLE = True


# ============================================================================
# 0. Data source selection
# ============================================================================
def choose_data_source():
    print("\n" + "=" * 70)
    print("📡 CHOOSE DATA SOURCE")
    print("=" * 70)
    print("  [1] Load real CSV dataset(s) from folder")
    print("  [2] Run full difficulty series (VERY HARD → HARD → EASY)")
    print("=" * 70)
    while True:
        choice = input("Enter choice [1/2]: ").strip()
        if choice in ('1', '2'):
            return choice
        print("⚠️  Please enter 1 or 2.")


# ============================================================================
# 0b. Synthetic 6G APT dataset generator (parameterised difficulty)
# ============================================================================
DIFFICULTY_PRESETS = {
    'very_hard': dict(mimicry_rate=0.30, spread_scale=1.00, separation_scale=1.00),
    'hard':      dict(mimicry_rate=0.15, spread_scale=0.75, separation_scale=1.15),
    'easy':      dict(mimicry_rate=0.05, spread_scale=0.50, separation_scale=1.40),
}

DIFFICULTY_LABELS = {'very_hard': 'VERY HARD', 'hard': 'HARD', 'easy': 'EASY'}


def generate_synthetic_6g_apt(n_samples=50000, random_state=42,
                              difficulty='very_hard'):
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"difficulty must be one of {list(DIFFICULTY_PRESETS)}")
    cfg = DIFFICULTY_PRESETS[difficulty]
    mimicry_rate     = cfg['mimicry_rate']
    spread_scale     = cfg['spread_scale']
    separation_scale = cfg['separation_scale']

    rng = np.random.RandomState(random_state)
    n_benign = n_samples // 2
    n_attack = n_samples - n_benign
    feature_names = [
        'flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts',
        'totlen_fwd_pkts', 'totlen_bwd_pkts',
        'fwd_pkt_len_mean', 'bwd_pkt_len_mean',
        'flow_iat_mean', 'flow_iat_std',
        'fwd_iat_mean', 'bwd_iat_mean',
        'rtt_proxy', 'jitter', 'pkt_loss_rate',
        'syn_flag_cnt', 'ack_flag_cnt', 'psh_flag_cnt', 'urg_flag_cnt',
        'down_up_ratio', 'active_mean', 'idle_mean',
        'subflow_fwd_bytes', 'subflow_bwd_bytes',
    ]

    def logn(mu, sigma, size): return rng.lognormal(mu, sigma * spread_scale, size)
    def norm(mu, sigma, size): return rng.normal(mu, sigma * spread_scale, size)
    def pois(lam, size):       return rng.poisson(lam, size).astype(float)
    def beta(a, b, size):      return rng.beta(a, b, size)
    def gamma(sh, sc, size):   return rng.gamma(sh, sc * spread_scale, size)

    benign = {
        'flow_duration':      logn(8.0, 1.2, n_benign),
        'tot_fwd_pkts':       pois(25, n_benign),
        'tot_bwd_pkts':       pois(22, n_benign),
        'totlen_fwd_pkts':    logn(7.5, 1.0, n_benign),
        'totlen_bwd_pkts':    logn(7.3, 1.0, n_benign),
        'fwd_pkt_len_mean':   norm(300, 120, n_benign),
        'bwd_pkt_len_mean':   norm(350, 140, n_benign),
        'flow_iat_mean':      logn(4.0, 1.5, n_benign),
        'flow_iat_std':       logn(3.5, 1.2, n_benign),
        'fwd_iat_mean':       logn(4.2, 1.5, n_benign),
        'bwd_iat_mean':       logn(3.8, 1.5, n_benign),
        'rtt_proxy':          logn(2.5, 0.8, n_benign),
        'jitter':             gamma(2.0, 2.0, n_benign),
        'pkt_loss_rate':      beta(0.5, 20, n_benign),
        'syn_flag_cnt':       pois(1, n_benign),
        'ack_flag_cnt':       pois(20, n_benign),
        'psh_flag_cnt':       pois(5, n_benign),
        'urg_flag_cnt':       pois(0.05, n_benign),
        'down_up_ratio':      beta(2, 2, n_benign),
        'active_mean':        logn(5.0, 1.2, n_benign),
        'idle_mean':          logn(6.0, 1.5, n_benign),
        'subflow_fwd_bytes':  logn(8.0, 1.0, n_benign),
        'subflow_bwd_bytes':  logn(8.0, 1.0, n_benign),
    }

    stages = ['Recon', 'InitialAccess', 'LateralMovement', 'C2Beacon', 'Exfiltration']
    stage_sizes = np.array([0.15, 0.20, 0.25, 0.25, 0.15])
    stage_counts = (stage_sizes * n_attack).astype(int)
    stage_counts[-1] = n_attack - stage_counts[:-1].sum()

    attack = {k: np.zeros(n_attack) for k in feature_names}
    offset = 0
    S = separation_scale

    for stage, cnt in zip(stages, stage_counts):
        sl = slice(offset, offset + cnt)
        if stage == 'Recon':
            attack['flow_duration'][sl]     = logn(6.5 - 0.3 * S, 1.5, cnt)
            attack['tot_fwd_pkts'][sl]      = pois(5, cnt)
            attack['tot_bwd_pkts'][sl]      = pois(3, cnt)
            attack['totlen_fwd_pkts'][sl]   = logn(5.5, 1.0, cnt)
            attack['totlen_bwd_pkts'][sl]   = logn(5.0, 1.0, cnt)
            attack['fwd_pkt_len_mean'][sl]  = norm(80, 40, cnt)
            attack['bwd_pkt_len_mean'][sl]  = norm(60, 40, cnt)
            attack['flow_iat_mean'][sl]     = logn(5.5 + 0.4 * S, 1.2, cnt)
            attack['flow_iat_std'][sl]      = logn(4.5, 1.0, cnt)
            attack['fwd_iat_mean'][sl]      = logn(5.5 + 0.4 * S, 1.2, cnt)
            attack['bwd_iat_mean'][sl]      = logn(5.0 + 0.3 * S, 1.2, cnt)
            attack['rtt_proxy'][sl]         = logn(2.8, 0.9, cnt)
            attack['jitter'][sl]            = gamma(2.5, 2.0, cnt)
            attack['pkt_loss_rate'][sl]     = beta(1.0, 15, cnt)
            attack['syn_flag_cnt'][sl]      = pois(8, cnt)
            attack['ack_flag_cnt'][sl]      = pois(2, cnt)
            attack['psh_flag_cnt'][sl]      = pois(0.5, cnt)
            attack['urg_flag_cnt'][sl]      = pois(0.05, cnt)
            attack['down_up_ratio'][sl]     = beta(0.8, 3, cnt)
            attack['active_mean'][sl]       = logn(3.5, 1.0, cnt)
            attack['idle_mean'][sl]         = logn(7.5 + 0.3 * S, 1.2, cnt)
            attack['subflow_fwd_bytes'][sl] = logn(5.5, 1.0, cnt)
            attack['subflow_bwd_bytes'][sl] = logn(5.0, 1.0, cnt)

        elif stage == 'InitialAccess':
            attack['flow_duration'][sl]     = logn(7.5 - 0.5 * S, 1.3, cnt)
            attack['tot_fwd_pkts'][sl]      = pois(35, cnt)
            attack['tot_bwd_pkts'][sl]      = pois(18, cnt)
            attack['totlen_fwd_pkts'][sl]   = logn(8.5 + 0.3 * S, 1.0, cnt)
            attack['totlen_bwd_pkts'][sl]   = logn(7.0 - 0.3 * S, 1.0, cnt)
            attack['fwd_pkt_len_mean'][sl]  = norm(500, 200, cnt)
            attack['bwd_pkt_len_mean'][sl]  = norm(250, 120, cnt)
            attack['flow_iat_mean'][sl]     = logn(3.5 - 0.4 * S, 1.2, cnt)
            attack['flow_iat_std'][sl]      = logn(3.8, 1.0, cnt)
            attack['fwd_iat_mean'][sl]      = logn(3.5 - 0.4 * S, 1.2, cnt)
            attack['bwd_iat_mean'][sl]      = logn(3.5 - 0.3 * S, 1.2, cnt)
            attack['rtt_proxy'][sl]         = logn(2.2, 0.7, cnt)
            attack['jitter'][sl]            = gamma(1.8, 2.5, cnt)
            attack['pkt_loss_rate'][sl]     = beta(1.5, 10, cnt)
            attack['syn_flag_cnt'][sl]      = pois(3, cnt)
            attack['ack_flag_cnt'][sl]      = pois(30, cnt)
            attack['psh_flag_cnt'][sl]      = pois(12, cnt)
            attack['urg_flag_cnt'][sl]      = pois(1.5, cnt)
            attack['down_up_ratio'][sl]     = beta(1.5, 2.5, cnt)
            attack['active_mean'][sl]       = logn(4.5, 1.0, cnt)
            attack['idle_mean'][sl]         = logn(5.5 - 0.3 * S, 1.3, cnt)
            attack['subflow_fwd_bytes'][sl] = logn(8.5 + 0.3 * S, 1.0, cnt)
            attack['subflow_bwd_bytes'][sl] = logn(7.0 - 0.3 * S, 1.0, cnt)

        elif stage == 'LateralMovement':
            attack['flow_duration'][sl]     = logn(7.0 - 0.3 * S, 1.4, cnt)
            attack['tot_fwd_pkts'][sl]      = pois(18, cnt)
            attack['tot_bwd_pkts'][sl]      = pois(15, cnt)
            attack['totlen_fwd_pkts'][sl]   = logn(7.0, 1.1, cnt)
            attack['totlen_bwd_pkts'][sl]   = logn(7.0, 1.1, cnt)
            attack['fwd_pkt_len_mean'][sl]  = norm(280, 130, cnt)
            attack['bwd_pkt_len_mean'][sl]  = norm(320, 150, cnt)
            attack['flow_iat_mean'][sl]     = logn(4.2, 1.4, cnt)
            attack['flow_iat_std'][sl]      = logn(3.7, 1.2, cnt)
            attack['fwd_iat_mean'][sl]      = logn(4.3, 1.4, cnt)
            attack['bwd_iat_mean'][sl]      = logn(4.0, 1.4, cnt)
            attack['rtt_proxy'][sl]         = logn(2.4, 0.8, cnt)
            attack['jitter'][sl]            = gamma(2.0, 2.2, cnt)
            attack['pkt_loss_rate'][sl]     = beta(0.8, 18, cnt)
            attack['syn_flag_cnt'][sl]      = pois(2, cnt)
            attack['ack_flag_cnt'][sl]      = pois(18, cnt)
            attack['psh_flag_cnt'][sl]      = pois(4, cnt)
            attack['urg_flag_cnt'][sl]      = pois(0.3, cnt)
            attack['down_up_ratio'][sl]     = beta(2, 2, cnt)
            attack['active_mean'][sl]       = logn(4.8, 1.1, cnt)
            attack['idle_mean'][sl]         = logn(6.2, 1.4, cnt)
            attack['subflow_fwd_bytes'][sl] = logn(7.0, 1.1, cnt)
            attack['subflow_bwd_bytes'][sl] = logn(7.0, 1.1, cnt)

        elif stage == 'C2Beacon':
            attack['flow_duration'][sl]     = logn(6.0, 1.0, cnt)
            attack['tot_fwd_pkts'][sl]      = pois(6, cnt)
            attack['tot_bwd_pkts'][sl]      = pois(6, cnt)
            attack['totlen_fwd_pkts'][sl]   = logn(5.5, 0.7, cnt)
            attack['totlen_bwd_pkts'][sl]   = logn(5.5, 0.7, cnt)
            attack['fwd_pkt_len_mean'][sl]  = norm(120, 30, cnt)
            attack['bwd_pkt_len_mean'][sl]  = norm(120, 30, cnt)
            attack['flow_iat_mean'][sl]     = logn(6.5 + 0.5 * S, 0.5, cnt)
            attack['flow_iat_std'][sl]      = logn(2.0 - 0.4 * S, 0.5, cnt)
            attack['fwd_iat_mean'][sl]      = logn(6.5 + 0.5 * S, 0.5, cnt)
            attack['bwd_iat_mean'][sl]      = logn(6.5 + 0.5 * S, 0.5, cnt)
            attack['rtt_proxy'][sl]         = logn(2.5, 0.6, cnt)
            attack['jitter'][sl]            = gamma(0.8, 1.0, cnt)
            attack['pkt_loss_rate'][sl]     = beta(0.3, 30, cnt)
            attack['syn_flag_cnt'][sl]      = pois(0.5, cnt)
            attack['ack_flag_cnt'][sl]      = pois(6, cnt)
            attack['psh_flag_cnt'][sl]      = pois(1.5, cnt)
            attack['urg_flag_cnt'][sl]      = pois(0.02, cnt)
            attack['down_up_ratio'][sl]     = beta(1, 1, cnt)
            attack['active_mean'][sl]       = logn(2.5, 0.5, cnt)
            attack['idle_mean'][sl]         = logn(7.5 + 0.3 * S, 0.5, cnt)
            attack['subflow_fwd_bytes'][sl] = logn(5.5, 0.7, cnt)
            attack['subflow_bwd_bytes'][sl] = logn(5.5, 0.7, cnt)

        else:  # Exfiltration
            attack['flow_duration'][sl]     = logn(9.0 + 0.4 * S, 1.2, cnt)
            attack['tot_fwd_pkts'][sl]      = pois(60, cnt)
            attack['tot_bwd_pkts'][sl]      = pois(8, cnt)
            attack['totlen_fwd_pkts'][sl]   = logn(10.0 + 0.4 * S, 1.0, cnt)
            attack['totlen_bwd_pkts'][sl]   = logn(6.0 - 0.4 * S, 1.0, cnt)
            attack['fwd_pkt_len_mean'][sl]  = norm(900, 250, cnt)
            attack['bwd_pkt_len_mean'][sl]  = norm(150, 80, cnt)
            attack['flow_iat_mean'][sl]     = logn(4.5, 1.5, cnt)
            attack['flow_iat_std'][sl]      = logn(4.0, 1.3, cnt)
            attack['fwd_iat_mean'][sl]      = logn(4.5, 1.5, cnt)
            attack['bwd_iat_mean'][sl]      = logn(4.0, 1.5, cnt)
            attack['rtt_proxy'][sl]         = logn(2.6, 0.9, cnt)
            attack['jitter'][sl]            = gamma(2.2, 2.0, cnt)
            attack['pkt_loss_rate'][sl]     = beta(1.0, 15, cnt)
            attack['syn_flag_cnt'][sl]      = pois(1, cnt)
            attack['ack_flag_cnt'][sl]      = pois(40, cnt)
            attack['psh_flag_cnt'][sl]      = pois(20, cnt)
            attack['urg_flag_cnt'][sl]      = pois(0.5, cnt)
            attack['down_up_ratio'][sl]     = beta(max(0.1, 0.3 / S), 5, cnt)
            attack['active_mean'][sl]       = logn(6.0, 1.0, cnt)
            attack['idle_mean'][sl]         = logn(5.5 - 0.3 * S, 1.3, cnt)
            attack['subflow_fwd_bytes'][sl] = logn(10.0 + 0.4 * S, 1.0, cnt)
            attack['subflow_bwd_bytes'][sl] = logn(6.0 - 0.4 * S, 1.0, cnt)

        offset += cnt

    mimic_mask = rng.rand(n_attack) < mimicry_rate
    n_mimic = int(mimic_mask.sum())
    for k in feature_names:
        benign_src = benign[k] if k in benign else attack[k][:n_benign]
        if n_mimic > 0:
            attack[k][mimic_mask] = rng.choice(benign_src, size=n_mimic, replace=True)

    data = {k: np.concatenate([benign[k], attack[k]]) for k in feature_names}
    y = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_attack, dtype=int)])
    idx = rng.permutation(n_samples)
    X_df = pd.DataFrame({k: v[idx] for k, v in data.items()}).clip(lower=0)
    y_s = pd.Series(y[idx], name='label')

    label = DIFFICULTY_LABELS[difficulty]
    print(f"\n✅ Synthetic 6G APT dataset generated ({label})")
    print(f"   Samples : {X_df.shape[0]}   Features: {X_df.shape[1]}")
    print(f"   Attack  : {int(y_s.sum())}  Benign: {int((1-y_s).sum())}  "
          f"({y_s.mean():.2%} attack)")
    print(f"   Difficulty knobs: mimicry={mimicry_rate:.0%}, "
          f"spread×{spread_scale:.2f}, separation×{separation_scale:.2f}")
    return X_df, y_s


# ============================================================================
# 1. Dataset loader
# ============================================================================
def load_dataset(path=None, allow_multiple=True):
    if path is None:
        path = input("Enter folder or CSV file path: ").strip()

    if path.endswith('.csv'):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        file_path = path
        df = None
    else:
        base_dir = path
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Folder not found: {base_dir}")
        csv_files = [f for f in os.listdir(base_dir) if f.lower().endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {base_dir}")
        print("Available CSV files:")
        for i, f in enumerate(csv_files):
            print(f"   [{i+1}] {f}")

        if len(csv_files) == 1:
            file_path = os.path.join(base_dir, csv_files[0])
            df = None
            print(f"Using: {csv_files[0]}")
        else:
            print("\nLoad options:")
            print("   [a] Auto-select test file (default)")
            print("   [m] Load ALL CSVs and concatenate")
            for i, f in enumerate(csv_files):
                print(f"   [{i+1}] {f}")
            choice = input("Choice [a/m/number, default=a]: ").strip().lower()

            if choice == 'm' and allow_multiple:
                frames = []
                for f in csv_files:
                    fp = os.path.join(base_dir, f)
                    d = pd.read_csv(fp)
                    d.columns = [str(c).strip().strip("'\"").strip()
                                 for c in d.columns]
                    frames.append(d)
                    print(f"   + {f}: {d.shape[0]} rows")
                df = pd.concat(frames, ignore_index=True)
                file_path = f"CONCATENATED ({len(csv_files)} files)"
            elif choice.isdigit() and 1 <= int(choice) <= len(csv_files):
                file_path = os.path.join(base_dir, csv_files[int(choice) - 1])
                df = None
            else:
                auto = [f for f in csv_files if 'test' in f.lower()]
                file_path = os.path.join(base_dir,
                                         auto[0] if auto else csv_files[0])
                df = None

    if df is None:
        print(f"📂 Loading {file_path} ...")
        df = pd.read_csv(file_path)
    else:
        print(f"📂 Loaded {file_path}")

    df.columns = [str(c).strip().strip("'\"").strip() for c in df.columns]
    print(f"✅ {df.shape[0]} rows, {df.shape[1]} columns.")

    label_candidates = ['Label', 'label', 'labels', 'Labels',
                        'Attack_Type', 'attack', 'Attack',
                        'Class', 'class', 'target', 'Target', 'y']
    label_col = None
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[-1]
        print(f"⚠️ Using last column as label: '{label_col}'")
    print(f"Label column: '{label_col}'")

    y_raw = df[label_col]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = (y_raw > 0).astype(int)
    else:
        y_str = y_raw.astype(str).str.strip()
        benign = ['benign', 'Benign', 'BENIGN', 'normal', 'Normal', '0', 'no']
        is_benign = y_str.str.lower().isin([b.lower() for b in benign])
        y = (~is_benign).astype(int)
    print(f"Attacks: {y.sum()}, Benign: {len(y) - y.sum()}")

    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    X = X.loc[:, (X != X.iloc[0]).any()]
    print(f"Numeric features: {X.shape[1]}")
    return X, y


# ============================================================================
# 2. Discretization utilities
# ============================================================================
def entropy_of_partition(y):
    counts = Counter(y)
    probs = [c / len(y) for c in counts.values()]
    return entropy(probs, base=2)


def entropy_discretize_feature(X_col, y, max_bins=5, min_samples=5, depth=0,
                               max_candidates=25, min_gain=0.01):
    if depth >= max_bins - 1:
        return []
    if hasattr(X_col, 'values'):
        X_col = X_col.values
    if hasattr(y, 'values'):
        y = y.values
    n = len(y)
    if n < min_samples * 2:
        return []
    unique_vals = np.unique(X_col)
    if len(unique_vals) < 2:
        return []
    sorted_idx = np.argsort(X_col)
    X_sorted = X_col[sorted_idx]
    y_sorted = y[sorted_idx]
    total_ent = entropy_of_partition(y_sorted)
    candidates = (unique_vals[:-1] + unique_vals[1:]) / 2.0
    if len(candidates) > max_candidates:
        candidates = np.linspace(unique_vals[0], unique_vals[-1], max_candidates)
    best_gain = -1.0
    best_cut = None
    best_left_mask = best_right_mask = best_left_y = best_right_y = None
    for cut in candidates:
        left_mask = X_sorted <= cut
        right_mask = ~left_mask
        n_left = np.sum(left_mask)
        n_right = n - n_left
        if n_left < min_samples or n_right < min_samples:
            continue
        left_ent = entropy_of_partition(y_sorted[left_mask])
        right_ent = entropy_of_partition(y_sorted[right_mask])
        gain = total_ent - (n_left / n * left_ent + n_right / n * right_ent)
        if gain > best_gain:
            best_gain = gain
            best_cut = cut
            best_left_mask = left_mask
            best_right_mask = right_mask
            best_left_y = y_sorted[left_mask]
            best_right_y = y_sorted[right_mask]
    if best_cut is None or best_gain < min_gain:
        return []
    cuts_left = entropy_discretize_feature(
        X_sorted[best_left_mask], best_left_y, max_bins, min_samples,
        depth + 1, max_candidates, min_gain)
    cuts_right = entropy_discretize_feature(
        X_sorted[best_right_mask], best_right_y, max_bins, min_samples,
        depth + 1, max_candidates, min_gain)
    return sorted(np.unique([best_cut] + cuts_left + cuts_right))


# ============================================================================
# 3a. Granular Entropy Classifier
# ============================================================================
class GranularEntropyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins=5, min_samples_per_bin=5, min_confidence=0.35,
                 feature_subset=None):
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.min_confidence = min_confidence
        self.feature_subset = feature_subset

    def get_params(self, deep=True):
        return {'n_bins': self.n_bins,
                'min_samples_per_bin': self.min_samples_per_bin,
                'min_confidence': self.min_confidence,
                'feature_subset': self.feature_subset}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _discretize_feature(self, X_col, y):
        return np.array(entropy_discretize_feature(
            X_col, y, max_bins=self.n_bins,
            min_samples=self.min_samples_per_bin))

    def discretize(self, X, y):
        self.bins = {}
        for col in X.columns:
            self.bins[col] = self._discretize_feature(X[col], y)

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_gran = X.copy()
        for col in X.columns:
            if col in self.bins and len(self.bins[col]) > 0:
                X_gran[col] = np.digitize(X[col], self.bins[col], right=False)
            else:
                X_gran[col] = 0
        return X_gran

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.feature_subset is not None:
            X = X[self.feature_subset]
        self.feature_names_in_ = X.columns.tolist()
        self.majority_class = y.value_counts().idxmax()
        self.discretize(X, y)
        X_gran = self.transform(X)
        self.rules = []
        for col in self.feature_names_in_:
            for val in np.unique(X_gran[col]):
                mask = X_gran[col] == val
                if np.sum(mask) < self.min_samples_per_bin:
                    continue
                class_counts = y[mask].value_counts()
                if len(class_counts) == 0:
                    continue
                maj_class = class_counts.idxmax()
                conf = class_counts[maj_class] / np.sum(mask)
                if conf > self.min_confidence:
                    self.rules.append((col, val, maj_class, conf))
        self.rules.sort(key=lambda x: x[3], reverse=True)
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        if self.feature_subset is not None:
            X = X[self.feature_subset]
        X_gran = self.transform(X)
        preds = []
        for _, row in X_gran.iterrows():
            assigned = False
            for (col, val, cls, _) in self.rules:
                if row[col] == val:
                    preds.append(cls)
                    assigned = True
                    break
            if not assigned:
                preds.append(self.majority_class)
        return np.array(preds)


# ============================================================================
# 3b. Granular Information Granulation
# ============================================================================
class GranularInformationGranulation(BaseEstimator, ClassifierMixin):
    def __init__(self, similarity_quantile=0.5, min_granule_size=3,
                 weight_by_similarity=True):
        self.similarity_quantile = similarity_quantile
        self.min_granule_size = min_granule_size
        self.weight_by_similarity = weight_by_similarity

    def get_params(self, deep=True):
        return {'similarity_quantile': self.similarity_quantile,
                'min_granule_size': self.min_granule_size,
                'weight_by_similarity': self.weight_by_similarity}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.X_train_ = Xs
        self.y_train_ = y
        self.classes_, counts = np.unique(y, return_counts=True)
        self.majority_ = self.classes_[np.argmax(counts)]
        k = max(2, min(self.min_granule_size, len(Xs)))
        nn = NearestNeighbors(n_neighbors=k).fit(Xs)
        dists, _ = nn.kneighbors(Xs)
        r = float(np.quantile(dists[:, -1], self.similarity_quantile))
        self.radius_ = max(r, 1e-3)
        self.nn_ = NearestNeighbors(radius=self.radius_).fit(Xs)
        self.fallback_ = NearestNeighbors(n_neighbors=1).fit(Xs)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X)
        dist_lists, ind_lists = self.nn_.radius_neighbors(Xs)
        preds = np.empty(len(Xs), dtype=self.y_train_.dtype)
        empty = np.array([len(idx) < self.min_granule_size for idx in ind_lists])
        if empty.any():
            _, j = self.fallback_.kneighbors(Xs[empty])
            preds[empty] = self.y_train_[j[:, 0]]
        for i in np.where(~empty)[0]:
            labels = self.y_train_[ind_lists[i]]
            if self.weight_by_similarity:
                w = 1.0 / (dist_lists[i] + 1e-6)
                score = {}
                for l, wt in zip(labels, w):
                    score[l] = score.get(l, 0.0) + wt
                preds[i] = max(score.items(), key=lambda kv: kv[1])[0]
            else:
                vals, cnts = np.unique(labels, return_counts=True)
                preds[i] = vals[np.argmax(cnts)]
        return preds


# ============================================================================
# 3c. Granular Rough Set
# ============================================================================
class GranularRoughSetTheory(BaseEstimator, ClassifierMixin):
    def __init__(self, neighborhood_size=5, use_lower_approx=True):
        self.neighborhood_size = neighborhood_size
        self.use_lower_approx = use_lower_approx

    def get_params(self, deep=True):
        return {'neighborhood_size': self.neighborhood_size,
                'use_lower_approx': self.use_lower_approx}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.X_train_ = Xs
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        self.y_idx_ = np.array([self.class_to_idx_[v] for v in y])
        k = max(2, min(self.neighborhood_size, len(Xs)))
        self.k_ = k
        self.nn_ = NearestNeighbors(n_neighbors=k).fit(Xs)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X)
        dist, ind = self.nn_.kneighbors(Xs)
        neigh_y = self.y_idx_[ind]
        w = 1.0 / (dist + 1e-6)
        n_cls = len(self.classes_)
        scores = np.zeros((len(Xs), n_cls))
        for j in range(n_cls):
            scores[:, j] = (w * (neigh_y == j)).sum(axis=1)
        preds_idx = scores.argmax(axis=1)
        if self.use_lower_approx:
            pure = (neigh_y == neigh_y[:, [0]]).all(axis=1)
            preds_idx[pure] = neigh_y[pure, 0]
        return self.classes_[preds_idx]


# ============================================================================
# 3d. Granular Fuzzy Set (log-domain)
# ============================================================================
class GranularFuzzySetTheory(BaseEstimator, ClassifierMixin):
    def __init__(self, combine='product', use_prior=True):
        self.combine = combine
        self.use_prior = use_prior

    def get_params(self, deep=True):
        return {'combine': self.combine, 'use_prior': self.use_prior}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.stats_ = {}
        self.prior_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.stats_[c] = (Xc.mean(axis=0), Xc.std(axis=0) + 1e-6)
            self.prior_[c] = (y == c).mean() if self.use_prior else 1.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        M = np.zeros((len(X), len(self.classes_)))
        for j, c in enumerate(self.classes_):
            mu, sigma = self.stats_[c]
            logZ = -0.5 * ((X - mu) / sigma) ** 2
            if self.combine == 'product':
                m = logZ.sum(axis=1)
            elif self.combine == 'min':
                m = logZ.min(axis=1)
            else:
                m = logZ.mean(axis=1)
            M[:, j] = m + np.log(self.prior_[c] + 1e-12)
        return self.classes_[M.argmax(axis=1)]


# ============================================================================
# 3e. Granular Neighborhood Systems
# ============================================================================
class GranularNeighborhoodSystems(BaseEstimator, ClassifierMixin):
    def __init__(self, k_neighbors=5, radius_scale=1.5):
        self.k_neighbors = k_neighbors
        self.radius_scale = radius_scale

    def get_params(self, deep=True):
        return {'k_neighbors': self.k_neighbors, 'radius_scale': self.radius_scale}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.X_train_ = Xs
        self.y_train_ = y
        self.classes_, cnts = np.unique(y, return_counts=True)
        self.majority_ = self.classes_[np.argmax(cnts)]
        k = max(2, min(self.k_neighbors, len(Xs)))
        self.k_ = k
        self._nn = NearestNeighbors().fit(Xs)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X)
        n_train = len(self.X_train_)
        kq = min(self.k_ * 3, n_train)
        dist, ind = self._nn.kneighbors(Xs, n_neighbors=kq)
        preds = np.empty(len(Xs), dtype=self.y_train_.dtype)
        for i in range(len(Xs)):
            kk = min(self.k_, len(dist[i])) - 1
            r = dist[i, kk] * self.radius_scale + 1e-6
            mask = dist[i] <= r
            if not mask.any():
                mask[0] = True
            labels = self.y_train_[ind[i][mask]]
            w = 1.0 / (dist[i][mask] + 1e-6)
            score = {}
            for l, wt in zip(labels, w):
                score[l] = score.get(l, 0.0) + wt
            preds[i] = max(score.items(), key=lambda kv: kv[1])[0]
        return preds


# ============================================================================
# 3f. Granular Multi-Granular Clustering
# ============================================================================
class GranularMultiGranularClustering(BaseEstimator, ClassifierMixin):
    def __init__(self, k_values=(5, 10, 20, 40), random_state=42):
        self.k_values = k_values
        self.random_state = random_state

    def get_params(self, deep=True):
        return {'k_values': self.k_values, 'random_state': self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.classes_, cnts = np.unique(y, return_counts=True)
        self.majority_ = self.classes_[np.argmax(cnts)]
        self.models_ = []
        for k in self.k_values:
            kk = int(min(k, len(Xs)))
            if kk < 2:
                continue
            km = KMeans(n_clusters=kk, random_state=self.random_state, n_init=5)
            labels = km.fit_predict(Xs)
            cluster_class = {}
            for c in np.unique(labels):
                m = labels == c
                if not m.any():
                    continue
                vals, cs = np.unique(y[m], return_counts=True)
                cluster_class[int(c)] = vals[np.argmax(cs)]
            self.models_.append((km, cluster_class))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X)
        if not self.models_:
            return np.full(len(Xs), self.majority_, dtype=self.majority_.dtype)
        votes = np.empty((len(self.models_), len(Xs)), dtype=object)
        for mi, (km, cluster_class) in enumerate(self.models_):
            labs = km.predict(Xs)
            votes[mi] = np.array([cluster_class[int(l)] for l in labs])
        preds = np.empty(len(Xs), dtype=self.majority_.dtype)
        for i in range(len(Xs)):
            vals, cs = np.unique(votes[:, i], return_counts=True)
            preds[i] = vals[np.argmax(cs)]
        return preds


# ============================================================================
# 3g. Granular Formal Concept Analysis
# ============================================================================
class GranularFormalConceptAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, n_thresholds=2, min_support=0.05, top_k_pairs=20,
                 max_features=40):
        self.n_thresholds = n_thresholds
        self.min_support = min_support
        self.top_k_pairs = top_k_pairs
        self.max_features = max_features

    def get_params(self, deep=True):
        return {'n_thresholds': self.n_thresholds,
                'min_support': self.min_support,
                'top_k_pairs': self.top_k_pairs,
                'max_features': self.max_features}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _select_features(self, X):
        variances = X.var(axis=0)
        if X.shape[1] <= self.max_features:
            return np.arange(X.shape[1])
        return np.argsort(-variances)[:self.max_features]

    def _binarize(self, X):
        cols = []
        for j in range(X.shape[1]):
            for t in self.thresholds_[j]:
                cols.append((X[:, j] >= t).astype(np.int8))
        return np.column_stack(cols) if cols else np.zeros((len(X), 0), dtype=np.int8)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.feature_idx_ = self._select_features(X)
        Xsel = X[:, self.feature_idx_]
        n = len(Xsel)
        self.thresholds_ = []
        for j in range(Xsel.shape[1]):
            qs = np.linspace(0.25, 0.75, self.n_thresholds)
            self.thresholds_.append(np.quantile(Xsel[:, j], qs))
        B = self._binarize(Xsel)
        self.n_binattrs_ = B.shape[1]
        self.classes_, cnts = np.unique(y, return_counts=True)
        self.majority_ = self.classes_[np.argmax(cnts)]
        min_sup = max(int(self.min_support * n), 2)
        self.class_concepts_ = {}
        for c in self.classes_:
            mask = y == c
            Bc = B[mask]
            singles = []
            for j in range(self.n_binattrs_):
                s = int(Bc[:, j].sum())
                if s >= min_sup:
                    singles.append((j, s / max(len(Bc), 1)))
            singles.sort(key=lambda t: -t[1])
            concepts = [(frozenset([j]), s) for j, s in singles]
            top = [j for j, _ in singles[:self.top_k_pairs]]
            for i1 in range(len(top)):
                for i2 in range(i1 + 1, len(top)):
                    j1, j2 = top[i1], top[i2]
                    s = int((Bc[:, j1] & Bc[:, j2]).sum())
                    if s >= min_sup:
                        concepts.append((frozenset([j1, j2]), s / max(len(Bc), 1)))
            self.class_concepts_[c] = concepts
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        B = self._binarize(X[:, self.feature_idx_])
        preds = np.empty(len(X), dtype=self.majority_.dtype)
        for i in range(len(X)):
            row = B[i]
            best_c = self.majority_
            best_score = -1.0
            for c, concepts in self.class_concepts_.items():
                sc = 0.0
                for attrs, sup in concepts:
                    if all(row[a] for a in attrs):
                        sc += sup
                if sc > best_score:
                    best_score = sc
                    best_c = c
            preds[i] = best_c
        return preds


# ============================================================================
# 3h. Deep Learning Baseline: 1D CNN (PyTorch)
# ============================================================================
class CNN1DBaseline(BaseEstimator, ClassifierMixin):
    """1D CNN treating the feature vector as a length-d 1-channel sequence."""
    def __init__(self, epochs=20, batch_size=256, lr=1e-3,
                 conv_channels=(32, 64), fc_hidden=32, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.conv_channels = conv_channels
        self.fc_hidden = fc_hidden
        self.random_state = random_state

    def get_params(self, deep=True):
        return {'epochs': self.epochs, 'batch_size': self.batch_size,
                'lr': self.lr, 'conv_channels': self.conv_channels,
                'fc_hidden': self.fc_hidden, 'random_state': self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _build_model(self, n_features):
        c1, c2 = self.conv_channels
        return nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c2, self.fc_hidden),
            nn.ReLU(),
            nn.Linear(self.fc_hidden, 2),
        )

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CNN1DBaseline.")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.classes_ = np.unique(y)
        self.class_map_ = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self.class_map_[v] for v in y])

        Xt = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(y_idx, dtype=torch.long)

        self.model_ = self._build_model(Xs.shape[1])
        opt = optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                out = self.model_(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
        self.model_.eval()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            logits = self.model_(Xt)
            preds_idx = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[preds_idx]


# ============================================================================
# 4. SSD Alone (balance-constrained F1 threshold selection)
# ============================================================================
class SSDAlone(BaseEstimator, ClassifierMixin):
    """Threshold-based SSD classifier with balance-constrained F1 search."""
    def __init__(self, threshold_rtt=0.5, threshold_delay=0.5,
                 delay_col=None, rtt_col=None,
                 min_pos_rate=0.20, max_pos_rate=0.80):
        self.threshold_rtt = threshold_rtt
        self.threshold_delay = threshold_delay
        self.delay_col = delay_col
        self.rtt_col = rtt_col
        self.min_pos_rate = min_pos_rate
        self.max_pos_rate = max_pos_rate

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        delay_candidates = ['avg_inter_hop_delay', 'delay', 'duration',
                            'Flow_Duration', 'flow_duration']
        rtt_candidates = ['rtt_ratio', 'rtt', 'jitter', 'RTT', 'Avg_RTT']

        if self.delay_col is None:
            for col in X.columns:
                if any(d in col.lower() for d in delay_candidates):
                    self.delay_col = col
                    break
        if self.rtt_col is None:
            for col in X.columns:
                if any(r in col.lower() for r in rtt_candidates):
                    self.rtt_col = col
                    break

        if self.delay_col is None or self.rtt_col is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                self.delay_col = numeric_cols[0]
                self.rtt_col = numeric_cols[1]
                print(f"   ⚠️ SSD Alone fallback: {self.delay_col}, {self.rtt_col}")
            else:
                raise ValueError("Not enough numeric features.")

        X_delay = X[self.delay_col].values
        X_rtt   = X[self.rtt_col].values
        y_vals  = y.values

        if np.ptp(X_delay) > 0:
            delays = np.quantile(X_delay, np.linspace(0.05, 0.95, 25))
        else:
            delays = np.array([X_delay[0]])
        if np.ptp(X_rtt) > 0:
            rtts = np.quantile(X_rtt, np.linspace(0.05, 0.95, 25))
        else:
            rtts = np.array([X_rtt[0]])

        best_score = -np.inf
        best_th = (None, None)
        for td in delays:
            for tr in rtts:
                preds = ((X_delay > td) | (X_rtt > tr)).astype(int)
                pos_rate = preds.mean()
                if pos_rate < self.min_pos_rate or pos_rate > self.max_pos_rate:
                    continue
                score = f1_score(y_vals, preds, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_th = (td, tr)

        if best_th[0] is None:
            best_th = (float(np.median(X_delay)),
                       float(np.median(X_rtt)))
            print("   ⚠️ SSD Alone: balance constraint rejected all candidates, "
                  "using median thresholds")

        self.threshold_delay, self.threshold_rtt = best_th
        print(f"  SSD Alone: thresholds – {self.delay_col}="
              f"{self.threshold_delay:.3f}, {self.rtt_col}="
              f"{self.threshold_rtt:.3f}  (F1={best_score:.3f})")
        return self

    def predict(self, X):
        if self.delay_col is None or self.rtt_col is None:
            raise ValueError("Model not fitted.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_delay = X[self.delay_col].values
        X_rtt   = X[self.rtt_col].values
        return ((X_delay > self.threshold_delay) |
                (X_rtt > self.threshold_rtt)).astype(int)

    def get_params(self, deep=True):
        return {'threshold_rtt': self.threshold_rtt,
                'threshold_delay': self.threshold_delay,
                'delay_col': self.delay_col,
                'rtt_col': self.rtt_col,
                'min_pos_rate': self.min_pos_rate,
                'max_pos_rate': self.max_pos_rate}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ============================================================================
# 5. Label-leakage diagnostic
# ============================================================================
def diagnose_label_leakage(X, y, f1_threshold=0.90, cv_folds=3, verbose=True):
    report = {
        'suspicious': [],
        'single_feature_f1': {},
        'n_duplicates': int(X.duplicated().sum()),
        'n_samples': len(X),
        'f1_threshold': f1_threshold,
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    for col in X.columns:
        try:
            scores = cross_val_score(
                DecisionTreeClassifier(max_depth=1, random_state=42),
                X[[col]].values, y.values, cv=cv, scoring='f1',
                error_score=0.0)
            f1 = float(np.mean(scores))
            report['single_feature_f1'][col] = f1
            if f1 >= f1_threshold:
                report['suspicious'].append(col)
        except Exception:
            continue

    if verbose:
        print("\n" + "=" * 72)
        print("🔬 LABEL-LEAKAGE DIAGNOSTIC")
        print("=" * 72)
        print(f"  Single-feature F1 threshold : {f1_threshold}")
        print(f"  Features tested             : {X.shape[1]}")
        print(f"  Duplicate rows              : {report['n_duplicates']} "
              f"({report['n_duplicates'] / len(X):.2%})")
        top = sorted(report['single_feature_f1'].items(),
                     key=lambda kv: -kv[1])[:10]
        print("\n  Top-10 most predictive single features:")
        for col, f1 in top:
            flag = "⚠️ LEAK" if f1 >= f1_threshold else "     "
            print(f"    {flag} {col:42s} F1 = {f1:.4f}")
        if report['suspicious']:
            print(f"\n  ⚠️ {len(report['suspicious'])} suspicious feature(s):")
            for col in report['suspicious']:
                print(f"       • {col}  (F1 = "
                      f"{report['single_feature_f1'][col]:.4f})")
        else:
            print("\n  ✅ No single feature exceeds the leakage threshold.")
        print("=" * 72)
    return report['suspicious'], report


def drop_leaking_features(X, y, f1_threshold=0.90, verbose=True):
    suspicious, report = diagnose_label_leakage(
        X, y, f1_threshold=f1_threshold, verbose=verbose)
    if suspicious:
        print(f"\n🗑️  Dropping {len(suspicious)} leaking feature(s):")
        for c in suspicious:
            print(f"     − {c}")
        X = X.drop(columns=suspicious)
        print(f"   Features remaining: {X.shape[1]}")
    else:
        print("\n✅ No features dropped.")
    return X, report


# ============================================================================
# 6. Evaluation
# ============================================================================
def evaluate_classifier(clf, X, y, cv):
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
               'fit_time': [], 'predict_time': [], 'predict_per_sample': []}
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        start = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = clf.predict(X_test)
        pred_time = time.perf_counter() - start
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        metrics['fit_time'].append(fit_time)
        metrics['predict_time'].append(pred_time)
        metrics['predict_per_sample'].append(pred_time / len(y_test))
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


# ============================================================================
# 7. Bayesian Optimization
# ============================================================================
def objective_optuna(trial, X, y):
    n_bins = trial.suggest_int('n_bins', 3, 10)
    min_samples = trial.suggest_int('min_samples_per_bin', 3, 15)
    conf = trial.suggest_float('min_confidence', 0.2, 0.7)
    clf = GranularEntropyClassifier(
        n_bins=n_bins, min_samples_per_bin=min_samples,
        min_confidence=conf, feature_subset=None)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if SMOTE_AVAILABLE:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
    return np.mean(f1_scores)


def run_bayesian_optimization(X, y, n_trials=30):
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna not installed. Cannot run.")
        return None, None
    print(f"\n🔬 Bayesian Optimization (Optuna) – {n_trials} trials")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(lambda t: objective_optuna(t, X, y), n_trials=n_trials,
                   show_progress_bar=True)
    best = study.best_trial
    print(f"\n🏆 Best F1: {best.value:.4f}")
    return best.params, best.value


# ============================================================================
# 8. Plotting utilities
# ============================================================================
def _category_of(name):
    if name.startswith('ML Baseline'):
        return 'ML'
    if name.startswith('DL Baseline'):
        return 'DL'
    if name == 'SSD Alone':
        return 'SSD'
    return 'GrC'


def _pareto_frontier(xs, ys):
    order = np.argsort(xs)
    frontier = []
    best_y = -np.inf
    for i in order:
        if ys[i] > best_y:
            frontier.append(i)
            best_y = ys[i]
    return np.array(frontier, dtype=int)


def plot_bars(results, save_path="granular_comparison.png"):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed – skipping bar chart.")
        return
    sns.set_style("whitegrid")
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        means = [results[n][metric][0] for n in results.keys()]
        stds  = [results[n][metric][1] for n in results.keys()]
        names = list(results.keys())
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2., m + 0.02,
                    f"{m:.3f}", ha='center', va='bottom', fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"📈 Bar chart saved → {save_path}")


def plot_pareto(results, save_path="granular_pareto.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping Pareto plot.")
        return

    names     = list(results.keys())
    f1        = np.array([results[n]['f1'][0] for n in names])
    f1_std    = np.array([results[n]['f1'][1] for n in names])
    train     = np.array([results[n]['fit_time'][0] for n in names])
    train_std = np.array([results[n]['fit_time'][1] for n in names])
    infer     = np.array([results[n]['predict_per_sample'][0] * 1000 for n in names])
    infer_std = np.array([results[n]['predict_per_sample'][1] * 1000 for n in names])
    cats      = np.array([_category_of(n) for n in names])

    style = {
        'SSD': dict(color='#d62728', marker='X', size=180, label='SSD only'),
        'GrC': dict(color='#1f77b4', marker='o', size=120, label='Granular (GrC)'),
        'ML':  dict(color='#2ca02c', marker='s', size=150, label='ML baseline'),
        'DL':  dict(color='#ff7f0e', marker='D', size=150, label='DL baseline'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, x, xerr, xlabel, title in [
        (axes[0], train, train_std, 'Training time (s, log scale)', 'F1 vs Training Time'),
        (axes[1], infer, infer_std, 'Inference time (ms/sample, log scale)', 'F1 vs Inference Time'),
    ]:
        for cat, st in style.items():
            mask = cats == cat
            if not mask.any():
                continue
            ax.errorbar(
                x[mask], f1[mask], xerr=xerr[mask], yerr=f1_std[mask],
                fmt=st['marker'], color=st['color'], ecolor=st['color'],
                markersize=np.sqrt(st['size']) * 1.4,
                markeredgecolor='black', markeredgewidth=0.6,
                capsize=3, alpha=0.85, linestyle='none', label=st['label'])
        front_idx = _pareto_frontier(x, f1)
        order = np.argsort(x[front_idx])
        ax.plot(x[front_idx][order], f1[front_idx][order],
                '--', color='gray', linewidth=1.6, alpha=0.8,
                label='Pareto frontier')
        for i in front_idx:
            short = names[i].replace('SSD + Granular ', 'GrC·') \
                            .replace('ML Baseline: ', 'ML·') \
                            .replace('DL Baseline: ', 'DL·') \
                            .replace('SSD Alone', 'SSD')
            ax.annotate(short, (x[i], f1[i]), textcoords='offset points',
                        xytext=(6, 5), fontsize=8, alpha=0.9)
        ax.set_xscale('log')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('F1 score (mean ± std)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(max(0, f1.min() - 0.08), min(1.02, f1.max() + 0.05))
        ax.grid(True, which='both', linestyle=':', alpha=0.4)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    plt.suptitle('Pareto Analysis — Granular, ML and DL Baselines',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()
    print(f"📈 Pareto plot saved → {save_path}")


def plot_efficiency_score(results, save_path="granular_efficiency.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping efficiency plot.")
        return

    names = list(results.keys())
    f1s   = np.array([results[n]['f1'][0] for n in names])
    ttrain = np.array([results[n]['fit_time'][0] for n in names])
    cats  = np.array([_category_of(n) for n in names])

    eps = 1e-4
    efficiency = f1s / np.maximum(ttrain, eps)
    order = np.argsort(-efficiency)
    names_s  = [names[i] for i in order]
    eff_s    = efficiency[order]
    f1_s     = f1s[order]
    train_s  = ttrain[order]
    cats_s   = cats[order]

    color_map = {
        'SSD': '#d62728',
        'GrC': '#1f77b4',
        'ML':  '#2ca02c',
        'DL':  '#ff7f0e',
    }
    colors = [color_map[c] for c in cats_s]

    fig, ax = plt.subplots(figsize=(16, 8))
    xpos = np.arange(len(names_s))
    bars = ax.bar(xpos, eff_s, color=colors, edgecolor='black',
                  linewidth=0.7, alpha=0.85)

    ax.set_yscale('log')
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace('SSD + Granular ', 'GrC·')
                         .replace('ML Baseline: ', 'ML·')
                         .replace('DL Baseline: ', 'DL·')
                         .replace('SSD Alone', 'SSD')
                        for n in names_s],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Efficiency score   F1 / train_time (log scale)', fontsize=12)
    ax.set_title('Efficiency Score — F1 per Second of Training',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', which='both', linestyle=':', alpha=0.4)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.text(len(names_s) - 0.4, 1.15, 'F1 = train time (break-even)',
            fontsize=8, color='gray', ha='right')

    for bar, eff, f1v, tv in zip(bars, eff_s, f1_s, train_s):
        h = bar.get_height()
        label = f"{eff:.1f}\n(F1={f1v:.3f}, t={tv:.2f}s)"
        ax.text(bar.get_x() + bar.get_width() / 2.,
                h * 1.15, label,
                ha='center', va='bottom', fontsize=6.5, alpha=0.9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[c],
                             edgecolor='black', linewidth=0.7, label=lbl)
               for c, lbl in [('SSD', 'SSD only'),
                              ('GrC', 'Granular (GrC)'),
                              ('ML',  'ML baseline'),
                              ('DL',  'DL baseline')]]
    ax.legend(handles=handles, loc='upper right', fontsize=10,
              framealpha=0.9, title='Category')
    ax.set_ylim(1e-3, max(eff_s) * 20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()
    print(f"📈 Efficiency score saved → {save_path}")


def plot_difficulty_curve(series_results, save_path="granular_difficulty.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping difficulty curve.")
        return

    difficulties = ['very_hard', 'hard', 'easy']
    x_ticks = np.arange(len(difficulties))
    x_labels = ['VERY HARD', 'HARD', 'EASY']
    method_names = list(series_results[difficulties[0]].keys())

    style = {
        'SSD': dict(color='#d62728', marker='X', label='SSD only'),
        'GrC': dict(color='#1f77b4', marker='o', label='Granular (GrC)'),
        'ML':  dict(color='#2ca02c', marker='s', label='ML baseline'),
        'DL':  dict(color='#ff7f0e', marker='D', label='DL baseline'),
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    for name in method_names:
        cat = _category_of(name)
        st = style[cat]
        means = np.array([series_results[d][name]['f1'][0] for d in difficulties])
        stds  = np.array([series_results[d][name]['f1'][1] for d in difficulties])
        ax.plot(x_ticks, means, marker=st['marker'], color=st['color'],
                markersize=8, markeredgecolor='black', markeredgewidth=0.6,
                linewidth=1.8, alpha=0.85,
                label=name.replace('SSD + Granular ', 'GrC·')
                          .replace('ML Baseline: ', 'ML·')
                          .replace('DL Baseline: ', 'DL·'))
        ax.errorbar(x_ticks, means, yerr=stds, fmt='none',
                    ecolor=st['color'], capsize=3, alpha=0.6)
        ax.annotate(name.replace('SSD + Granular ', 'GrC·')
                       .replace('ML Baseline: ', 'ML·')
                       .replace('DL Baseline: ', 'DL·')
                       .replace('SSD Alone', 'SSD'),
                    (x_ticks[-1], means[-1]),
                    textcoords='offset points', xytext=(8, 0),
                    fontsize=7.5, color=st['color'], verticalalignment='center')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel('F1 score (mean ± std)', fontsize=12)
    ax.set_xlabel('Synthetic 6G APT difficulty level', fontsize=12)
    ax.set_title('F1 vs Difficulty — Granular, ML and DL Baselines',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.axvspan(-0.5, 0.5, alpha=0.05, color='red')
    ax.axvspan(0.5, 1.5, alpha=0.05, color='orange')
    ax.axvspan(1.5, 2.5, alpha=0.05, color='green')

    cat_handles = [plt.Line2D([0], [0], color=st['color'],
                              marker=st['marker'], linestyle='none',
                              markersize=9, markeredgecolor='black',
                              label=st['label'])
                   for st in style.values()]
    ax.legend(handles=cat_handles, loc='lower right',
              fontsize=10, framealpha=0.9, title='Category')

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()
    print(f"📈 Difficulty curve saved → {save_path}")


# ============================================================================
# 9. Difficulty series runner
# ============================================================================
def run_difficulty_series(n_samples=50000, random_state=42,
                          classifiers_factory=None, verbose=True):
    difficulties = ['very_hard', 'hard', 'easy']
    display = {'very_hard': 'VERY HARD', 'hard': 'HARD', 'easy': 'EASY'}
    series_results = {}

    for diff in difficulties:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"🔁 DIFFICULTY SERIES — {display[diff]}")
            print(f"{'=' * 70}")
        X, y = generate_synthetic_6g_apt(
            n_samples=n_samples, random_state=random_state, difficulty=diff)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        classifiers = classifiers_factory()
        results = {}
        for name, clf in classifiers.items():
            if verbose:
                print(f"   ⏳ {name} ...", end=' ', flush=True)
            try:
                scores = evaluate_classifier(clf, X, y, cv)
                results[name] = scores
                if verbose:
                    print(f"F1 = {scores['f1'][0]:.3f}")
            except Exception as e:
                if verbose:
                    print(f"❌ {e}")
        series_results[diff] = results
    return series_results


# ============================================================================
# 10. Classifier factory (13 methods: 8 GrC + 3 ML + 2 DL)
# ============================================================================
def build_classifier_dict():
    classifiers = {
        # ==================== SSD + Granular (8) ====================
        'SSD Alone': SSDAlone(),
        'SSD + Granular Entropy': GranularEntropyClassifier(
            n_bins=5, min_samples_per_bin=5, min_confidence=0.35),
        'SSD + Granular Info Granulation': GranularInformationGranulation(
            similarity_quantile=0.5, min_granule_size=3,
            weight_by_similarity=True),
        'SSD + Granular Rough Set': GranularRoughSetTheory(
            neighborhood_size=5, use_lower_approx=True),
        'SSD + Granular Fuzzy Set': GranularFuzzySetTheory(
            combine='product', use_prior=True),
        'SSD + Granular Neighborhood': GranularNeighborhoodSystems(
            k_neighbors=5, radius_scale=1.5),
        'SSD + Granular Multi-Clustering': GranularMultiGranularClustering(
            k_values=(5, 10, 20, 40), random_state=42),
        'SSD + Granular FCA': GranularFormalConceptAnalysis(
            n_thresholds=2, min_support=0.05, top_k_pairs=15, max_features=40),
    }

    # ==================== ML Baselines (3) ====================
    if SKLEARN_BASELINES_AVAILABLE:
        # 1. Random Forest
        classifiers['ML Baseline: Random Forest'] = RandomForestClassifier(
            n_estimators=150, max_depth=20, n_jobs=-1,
            class_weight='balanced', random_state=42)

        # 2. XGBoost (or Gradient Boosting fallback)
        if XGBOOST_AVAILABLE:
            classifiers['ML Baseline: XGBoost'] = xgb.XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric='logloss', random_state=42, n_jobs=-1)
        else:
            classifiers['ML Baseline: Gradient Boosting'] = \
                GradientBoostingClassifier(
                    n_estimators=150, max_depth=5, random_state=42)

        # 3. SVM (RBF kernel)
        classifiers['ML Baseline: SVM'] = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0, gamma='scale',
                        class_weight='balanced', probability=False,
                        cache_size=500))
        ])

    # ==================== DL Baselines (2) ====================
    if SKLEARN_BASELINES_AVAILABLE:
        # 1. MLP (scikit-learn)
        classifiers['DL Baseline: MLP'] = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64),
                                  activation='relu', solver='adam',
                                  max_iter=200, early_stopping=True,
                                  n_iter_no_change=10, random_state=42))
        ])

    if TORCH_AVAILABLE:
        # 2. 1D-CNN
        classifiers['DL Baseline: 1D-CNN'] = CNN1DBaseline(
            epochs=20, batch_size=256, lr=1e-3,
            conv_channels=(32, 64), fc_hidden=32, random_state=42)

    return classifiers


# ============================================================================
# 11. Main
# ============================================================================
def main():
    bayesian = False
    args = sys.argv[1:]
    if 'bayesian' in args or 'optuna' in args:
        bayesian = True
        args = [a for a in args if a not in ['bayesian', 'optuna']]

    auto_drop = True
    if '--no-drop' in args:
        auto_drop = False
        args = [a for a in args if a != '--no-drop']

    SOURCE_MAP = {
        '1': ('csv',    None),
        '2': ('series', None),
    }

    if args and args[0] in SOURCE_MAP:
        source = args[0]
        args = args[1:]
    else:
        source = choose_data_source()

    kind, difficulty = SOURCE_MAP[source]

    # ============================================================
    # Mode 2 — Difficulty series
    # ============================================================
    if kind == 'series':
        n_syn = 50000
        if args and args[0].isdigit():
            n_syn = int(args[0])
        print(f"\n🔁 Running difficulty series: VERY HARD → HARD → EASY "
              f"({n_syn} samples each)")
        series_results = run_difficulty_series(
            n_samples=n_syn, random_state=42,
            classifiers_factory=build_classifier_dict, verbose=True)

        print("\n" + "=" * 120)
        print("📊 DIFFICULTY SERIES SUMMARY (F1 mean)")
        print("=" * 120)
        methods = list(series_results['very_hard'].keys())
        rows = []
        for m in methods:
            row = {'Category': _category_of(m), 'Method': m}
            for d in ['very_hard', 'hard', 'easy']:
                row[d.replace('_', ' ').upper()] = \
                    f"{series_results[d][m]['f1'][0]:.3f}"
            row['Δ(EASY−VH)'] = \
                f"{series_results['easy'][m]['f1'][0] - series_results['very_hard'][m]['f1'][0]:+.3f}"
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
        print("=" * 120)

        plot_difficulty_curve(series_results,
                              save_path="granular_difficulty.png")
        plot_efficiency_score(series_results['easy'],
                              save_path="granular_efficiency.png")
        print("\n✅ Difficulty series complete.")
        return

    # ============================================================
    # Mode 1 — Real CSV dataset(s)
    # ============================================================
    path = args[0] if args else None
    try:
        X, y = load_dataset(path)
    except Exception as e:
        print(f"Error: {e}")
        X, y = load_dataset()

    print(f"\n✅ Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"attack ratio {y.mean():.2%}")
    if len(X) > 50000:
        X = X.sample(50000, random_state=42)
        y = y.loc[X.index]
        print(f"   Downsampled to {len(X)} for speed")

    if auto_drop:
        X, leak_report = drop_leaking_features(X, y, f1_threshold=0.90,
                                                verbose=True)
    else:
        diagnose_label_leakage(X, y, f1_threshold=0.90, verbose=True)
        print("ℹ️  Auto-drop disabled (--no-drop) — features retained.")

    if bayesian:
        best_params, best_f1 = run_bayesian_optimization(X, y, n_trials=30)
        if best_params:
            print("\nUse these parameters for GranularEntropyClassifier:")
            print(f"  n_bins = {best_params['n_bins']}")
            print(f"  min_samples_per_bin = {best_params['min_samples_per_bin']}")
            print(f"  min_confidence = {best_params['min_confidence']:.3f}")
        return

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    ssd_features = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    print(f"SSD features: {ssd_features}")

    classifiers = build_classifier_dict()

    results = {}
    for name, clf in classifiers.items():
        print(f"\n🔍 Evaluating {name} ...")
        try:
            scores = evaluate_classifier(clf, X, y, cv)
            results[name] = scores
            print(f"   F1: {scores['f1'][0]:.3f} ± {scores['f1'][1]:.3f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 145)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 145)
    rows = []
    for name, scores in results.items():
        rows.append({
            'Category': _category_of(name),
            'Method': name,
            'Acc':  f"{scores['accuracy'][0]:.3f} ± {scores['accuracy'][1]:.3f}",
            'Prec': f"{scores['precision'][0]:.3f} ± {scores['precision'][1]:.3f}",
            'Rec':  f"{scores['recall'][0]:.3f} ± {scores['recall'][1]:.3f}",
            'F1':   f"{scores['f1'][0]:.3f} ± {scores['f1'][1]:.3f}",
            'Train (s)': f"{scores['fit_time'][0]:.3f} ± {scores['fit_time'][1]:.3f}",
            'Infer (ms/sample)': f"{scores['predict_per_sample'][0]*1000:.3f} ± "
                                 f"{scores['predict_per_sample'][1]*1000:.3f}",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("=" * 145)

    try:
        plot_bars(results, "granular_comparison.png")
        plot_pareto(results, "granular_pareto.png")
        plot_efficiency_score(results, "granular_efficiency.png")
    except ImportError:
        print("matplotlib/seaborn not installed – skipping plots.")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()