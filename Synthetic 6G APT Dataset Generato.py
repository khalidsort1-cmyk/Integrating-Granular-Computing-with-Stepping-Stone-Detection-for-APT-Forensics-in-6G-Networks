import numpy as np
import pandas as pd

# ============================================================================
# Difficulty presets
# ============================================================================
DIFFICULTY_PRESETS = {
    'very_hard': dict(mimicry_rate=0.30, spread_scale=1.00, separation_scale=1.00),
    'hard':      dict(mimicry_rate=0.15, spread_scale=0.75, separation_scale=1.15),
    'easy':      dict(mimicry_rate=0.05, spread_scale=0.50, separation_scale=1.40),
}

DIFFICULTY_LABELS = {'very_hard': 'VERY HARD', 'hard': 'HARD', 'easy': 'EASY'}


def generate_synthetic_6g_apt(n_samples=50000, random_state=42,
                              difficulty='very_hard'):
    """Generate a synthetic 6G APT dataset with configurable difficulty.

    Parameters
    ----------
    n_samples : int
        Total number of samples (50/50 benign/attack split).
    random_state : int
        Seed for reproducibility.
    difficulty : str
        One of {'very_hard', 'hard', 'easy'}.

    Returns
    -------
    X_df : pandas.DataFrame  (n_samples, 23)
        Flow-statistics features across five APT stages.
    y_s  : pandas.Series     (n_samples,)
        Binary labels (0 = benign, 1 = attack).
    """
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

    # ---------- Distribution helpers (fold in the spread scale) ----------
    def logn(mu, sigma, size): return rng.lognormal(mu, sigma * spread_scale, size)
    def norm(mu, sigma, size): return rng.normal(mu, sigma * spread_scale, size)
    def pois(lam, size):       return rng.poisson(lam, size).astype(float)
    def beta(a, b, size):      return rng.beta(a, b, size)
    def gamma(sh, sc, size):   return rng.gamma(sh, sc * spread_scale, size)

    # ---------- Benign distribution ----------
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

    # ---------- Attack distribution across 5 APT stages ----------
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
            # Separable mainly by IAT variance → non-linear
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

    # ---------- Mimicry: fraction of attacks drawn from benign ----------
    mimic_mask = rng.rand(n_attack) < mimicry_rate
    n_mimic = int(mimic_mask.sum())
    for k in feature_names:
        benign_src = benign[k] if k in benign else attack[k][:n_benign]
        if n_mimic > 0:
            attack[k][mimic_mask] = rng.choice(benign_src, size=n_mimic, replace=True)

    # ---------- Combine, shuffle, clip ----------
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