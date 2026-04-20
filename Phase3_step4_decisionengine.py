"""
=============================================================
PHASE 3 — STEP 4: Goal-Oriented Decision Engine + Baselines
Project: Goal-Driven DT-Based Adaptive IRS & Waveform
         Optimization for 6G Wireless Systems
=============================================================
INPUT  : phase3_outputs/ (from steps 1-3)
OUTPUTS: phase3_outputs/step4_*.png
         phase3_outputs/decision_engine_results.pkl
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle, os, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cpu')   # CPU fine for inference
OUT    = 'phase3_outputs'
assert os.path.exists(OUT), "Run steps 1-3 first"

# ============================================================
# LOAD MODELS AND SCALERS
# ============================================================
print("=" * 60)
print("PHASE 3 — STEP 4: Decision Engine & Evaluation")
print("=" * 60)

# --- Rebuild and load Model 1 ---
class PerformancePredictor(nn.Module):
    def __init__(self, n_in=24, n_out=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, n_out)
        )
    def forward(self, x): return self.net(x)

class ChannelTransitionLSTM(nn.Module):
    def __init__(self, n_in=19, n_out=8,
                 hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_in, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden2, 64)
        self.fc2   = nn.Linear(64, 32)
        self.out   = nn.Linear(32, n_out)
        self.relu  = nn.ReLU()
    def forward(self, x):
        out, _ = self.lstm1(x); out = self.drop1(out)
        out, _ = self.lstm2(out); out = out[:, -1, :]
        out = self.drop2(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return self.out(out)

cfg1 = pickle.load(open(f'{OUT}/model1_config.pkl', 'rb'))
cfg2 = pickle.load(open(f'{OUT}/model2_config.pkl', 'rb'))

model1 = PerformancePredictor(cfg1['n_in'], cfg1['n_out'])
model1.load_state_dict(torch.load(f'{OUT}/model1_performance_predictor.pth',
                                   map_location='cpu'))
model1.eval()

model2 = ChannelTransitionLSTM(cfg2['n_in'], cfg2['n_out'])
model2.load_state_dict(torch.load(f'{OUT}/model2_channel_transition.pth',
                                   map_location='cpu'))
model2.eval()

sc_m1_X = pickle.load(open(f'{OUT}/scaler_m1_X.pkl', 'rb'))
sc_m1_y = pickle.load(open(f'{OUT}/scaler_m1_y.pkl', 'rb'))
sc_m2_X = pickle.load(open(f'{OUT}/scaler_m2_X.pkl', 'rb'))
sc_m2_y = pickle.load(open(f'{OUT}/scaler_m2_y.pkl', 'rb'))

ref      = pickle.load(open(f'{OUT}/reference_values.pkl', 'rb'))
meta     = pickle.load(open(f'{OUT}/metadata.pkl', 'rb'))
cands    = pickle.load(open(f'{OUT}/candidates_list.pkl', 'rb'))

df_test  = pd.read_csv(f'{OUT}/df_test.csv')
print(f"\nLoaded models, scalers, {len(cands)} candidates")
print(f"Test set: {len(df_test)} rows")

# ============================================================
# PREFERENCE SETUP
# ============================================================
PREF_WEIGHTS = {
    'MaxReliability':   [0.70, 0.15, 0.10, 0.05],
    'MaxThroughput':    [0.10, 0.70, 0.10, 0.10],
    'UltraLowLatency':  [0.05, 0.10, 0.75, 0.10],
    'EnergyEfficient':  [0.10, 0.15, 0.10, 0.65],
    'Balanced':         [0.25, 0.25, 0.25, 0.25],
    'ReliabilitySpeed': [0.45, 0.35, 0.15, 0.05],
    'GreenThroughput':  [0.10, 0.45, 0.10, 0.35],
    'MissionCritical':  [0.40, 0.05, 0.45, 0.10]
}
SHORT = ['MxR','MxT','ULL','Enr','Bal','R+S','GTP','MC']
PREF_LIST = list(PREF_WEIGHTS.keys())

M1_FEATURES = meta['M1_FEATURES']
M2_FEATURES = meta['M2_FEATURES']

# ============================================================
# DECISION ENGINE — BATCHED INFERENCE OVER 104 CANDIDATES
# ============================================================
def build_m2_batch(channel_row, candidates, pref_idx):
    """
    Build (104, 19) input matrix for Model 2 batch inference.
    channel_row: dict of current channel features
    """
    rows = []
    for c in candidates:
        row = [
            channel_row['snr_input_dB'],
            channel_row['sinr_dB'],
            channel_row['channel_gain'],
            channel_row['effective_rank'],
            channel_row['path_loss_dB'],
            channel_row['shadow_dB'],
            channel_row['rms_delay_spread'],
            channel_row['doppler_spread_Hz'],
            channel_row['distance_m'],
            channel_row['irs_gain_dB'],
            channel_row['AoD_mean'],
            channel_row['AoA_mean'],
            c['irs_phase_mean'],
            c['irs_phase_std'],
            c['irs_is_dft'],
            c['wf_scs_kHz'],
            c['wf_cp'],
            c['wf_mod'],
            float(pref_idx)
        ]
        rows.append(row)
    return np.array(rows, dtype=np.float32)  # (104, 19)


def build_m1_batch(next_channel_batch, candidates, weights):
    """
    Build (104, 24) input matrix for Model 1 batch inference.
    next_channel_batch: (104, 8) predicted next channel
    """
    rows = []
    for i, c in enumerate(candidates):
        nc = next_channel_batch[i]
        # next_channel has 8 features; take first 10 of M1 channel state
        # mapping: next_snr→1, next_gain→3 etc. Use available next features
        row = [
            nc[0],   # next_snr_dB
            0.0,     # sinr_dB (not predicted, use 0 placeholder)
            nc[1],   # next_channel_gain
            nc[2],   # next_effective_rank
            nc[3],   # next_path_loss_dB
            nc[6],   # next_shadow_dB
            nc[4],   # next_rms_delay_spread
            nc[5],   # next_doppler_Hz
            nc[7],   # next_distance_m
            0.0,     # irs_gain_dB placeholder
            c['irs_phase_mean'],
            c['irs_phase_std'],
            c['irs_is_dft'],
            c['wf_scs_kHz'],
            c['wf_cp'],
            c['wf_mod'],
            weights[0], weights[1], weights[2], weights[3],
            0.0, 0.0, 0.0, 0.0   # delta features = 0 at inference
        ]
        rows.append(row)
    return np.array(rows, dtype=np.float32)  # (104, 24)


def compute_goal_errors(pred_orig, weights, ref):
    """
    Compute goal error for each candidate.
    pred_orig: (104, 4) — [log_BER, throughput, latency, energy]
    weights:   [w_BER, w_TP, w_Lat, w_E]
    """
    BER_best, TP_best   = ref['BER_best'], ref['TP_best']
    Lat_best, E_best    = ref['Lat_best'], ref['E_best']
    BER_ref,  TP_ref    = ref['BER_ref'],  ref['TP_ref']
    Lat_ref,  E_ref     = ref['Lat_ref'],  ref['E_ref']

    errors = (
        weights[0] * (pred_orig[:, 0] - BER_best) / (BER_ref + 1e-8) +
        weights[1] * (TP_best - pred_orig[:, 1])  / (TP_ref  + 1e-8) +
        weights[2] * (pred_orig[:, 2] - Lat_best) / (Lat_ref + 1e-8) +
        weights[3] * (pred_orig[:, 3] - E_best)   / (E_ref   + 1e-8)
    )
    return errors   # (104,)


def select_best_config(channel_row, preference_label, pref_idx):
    """
    Full decision engine: current channel + preference → best config.
    Returns: best_config dict, predicted metrics, all goal errors
    """
    weights = PREF_WEIGHTS[preference_label]

    # Step 1: Build Model 2 batch input
    m2_in  = build_m2_batch(channel_row, cands, pref_idx)
    m2_sc  = sc_m2_X.transform(m2_in)          # (104, 19)
    m2_t   = torch.FloatTensor(m2_sc).unsqueeze(1)  # (104, 1, 19)

    # Step 2: Predict next channel state (batched)
    with torch.no_grad():
        next_ch_sc = model2(m2_t).numpy()       # (104, 8)
    next_ch = sc_m2_y.inverse_transform(next_ch_sc)

    # Step 3: Build Model 1 batch input using predicted next channel
    m1_in  = build_m1_batch(next_ch, cands, weights)
    m1_sc  = sc_m1_X.transform(m1_in)          # (104, 24)
    m1_t   = torch.FloatTensor(m1_sc)

    # Step 4: Predict performance metrics (batched)
    with torch.no_grad():
        perf_sc = model1(m1_t).numpy()          # (104, 4)
    perf = sc_m1_y.inverse_transform(perf_sc)   # (104, 4)

    # Step 5: Compute goal errors and select best
    errors   = compute_goal_errors(perf, weights, ref)
    best_idx = int(np.argmin(errors))

    return cands[best_idx], perf[best_idx], errors, perf


# ============================================================
# EVALUATION: 500 TEST SAMPLES (50 per preference)
# ============================================================
print("\n=== Running Decision Engine on Test Set ===")
print("  Sampling 50 rows per preference (400 total)...")

N_PER_PREF = 50

# Columns needed for channel_row
CH_COLS = ['snr_input_dB', 'sinr_dB', 'channel_gain', 'effective_rank',
           'path_loss_dB', 'shadow_dB', 'rms_delay_spread',
           'doppler_spread_Hz', 'distance_m', 'irs_gain_dB',
           'AoD_mean', 'AoA_mean']

# Ground truth columns
GT_COLS = ['BER', 'throughput_bpsHz', 'latency_ms', 'energy_per_bit']

results = {p: {
    'your_system':    {'BER':[], 'TP':[], 'Lat':[], 'E':[], 'goal_err':[]},
    'baseline_rand':  {'BER':[], 'TP':[], 'Lat':[], 'E':[], 'goal_err':[]},
    'baseline_fixed': {'BER':[], 'TP':[], 'Lat':[], 'E':[], 'goal_err':[]},
    'baseline_irs':   {'BER':[], 'TP':[], 'Lat':[], 'E':[], 'goal_err':[]},
    'ground_truth':   {'BER':[], 'TP':[], 'Lat':[], 'E':[]}
} for p in PREF_LIST}

for pref_i, pref in enumerate(PREF_LIST):
    pref_df  = df_test[df_test['preference_label'] == pref]
    n_sample = min(N_PER_PREF, len(pref_df))
    sample   = pref_df.sample(n_sample, random_state=42)
    weights  = PREF_WEIGHTS[pref]
    pref_idx = pref_i + 1

    print(f"  Processing {pref} ({n_sample} samples)...")

    for _, row in sample.iterrows():
        channel_row = {c: float(row[c]) for c in CH_COLS}

        # Ground truth (what actually happened in simulation)
        gt_BER = float(row['BER'])
        gt_TP  = float(row['throughput_bpsHz'])
        gt_Lat = float(row['latency_ms'])
        gt_E   = float(row['energy_per_bit'])

        results[pref]['ground_truth']['BER'].append(gt_BER)
        results[pref]['ground_truth']['TP'].append(gt_TP)
        results[pref]['ground_truth']['Lat'].append(gt_Lat)
        results[pref]['ground_truth']['E'].append(gt_E)

        # --- YOUR SYSTEM: joint IRS + waveform + goal-aware ---
        _, perf_sys, errors_sys, all_perf = select_best_config(
            channel_row, pref, pref_idx)
        log_ber = perf_sys[0]
        ber_sys = 10 ** log_ber
        results[pref]['your_system']['BER'].append(ber_sys)
        results[pref]['your_system']['TP'].append(perf_sys[1])
        results[pref]['your_system']['Lat'].append(perf_sys[2])
        results[pref]['your_system']['E'].append(perf_sys[3])
        results[pref]['your_system']['goal_err'].append(min(errors_sys))

        # --- BASELINE 1: Random config ---
        rand_idx  = np.random.randint(0, len(cands))
        rand_perf = all_perf[rand_idx]
        rand_err  = compute_goal_errors(
            all_perf, weights, ref)[rand_idx]
        results[pref]['baseline_rand']['BER'].append(10**rand_perf[0])
        results[pref]['baseline_rand']['TP'].append(rand_perf[1])
        results[pref]['baseline_rand']['Lat'].append(rand_perf[2])
        results[pref]['baseline_rand']['E'].append(rand_perf[3])
        results[pref]['baseline_rand']['goal_err'].append(float(rand_err))

        # --- BASELINE 2: Fixed waveform W2, random IRS ---
        # W2 = 15kHz, CP16, 16QAM → wf_idx=1 in cands (0-indexed)
        w2_cands = [c for c in cands if
                    c['wf_scs_kHz']==15 and c['wf_cp']==16 and c['wf_mod']==2]
        if w2_cands:
            w2_rand_idx = np.random.randint(0, len(w2_cands))
            # Find this candidate in all_perf
            all_cand_keys = [(c['irs_idx'], c['wf_scs_kHz'],
                              c['wf_cp'], c['wf_mod']) for c in cands]
            w2_c = w2_cands[w2_rand_idx]
            w2_key = (w2_c['irs_idx'], w2_c['wf_scs_kHz'],
                      w2_c['wf_cp'], w2_c['wf_mod'])
            try:
                w2_pos = all_cand_keys.index(w2_key)
            except ValueError:
                w2_pos = 0
            w2_perf = all_perf[w2_pos]
            w2_err  = compute_goal_errors(all_perf, weights, ref)[w2_pos]
        else:
            w2_pos  = 0
            w2_perf = all_perf[0]
            w2_err  = compute_goal_errors(all_perf, weights, ref)[0]
        results[pref]['baseline_fixed']['BER'].append(10**w2_perf[0])
        results[pref]['baseline_fixed']['TP'].append(w2_perf[1])
        results[pref]['baseline_fixed']['Lat'].append(w2_perf[2])
        results[pref]['baseline_fixed']['E'].append(w2_perf[3])
        results[pref]['baseline_fixed']['goal_err'].append(float(w2_err))

        # --- BASELINE 3: IRS-only (optimize IRS, fix waveform W2) ---
        w2_all_perf = all_perf[[i for i, c in enumerate(cands)
                                 if c['wf_scs_kHz']==15 and
                                    c['wf_cp']==16 and c['wf_mod']==2]]
        if len(w2_all_perf) > 0:
            w2_errors = compute_goal_errors(w2_all_perf, weights, ref)
            best_irs_idx = int(np.argmin(w2_errors))
            irs_perf     = w2_all_perf[best_irs_idx]
            irs_err      = w2_errors[best_irs_idx]
        else:
            irs_perf = all_perf[0]
            irs_err  = errors_sys[0]
        results[pref]['baseline_irs']['BER'].append(10**irs_perf[0])
        results[pref]['baseline_irs']['TP'].append(irs_perf[1])
        results[pref]['baseline_irs']['Lat'].append(irs_perf[2])
        results[pref]['baseline_irs']['E'].append(irs_perf[3])
        results[pref]['baseline_irs']['goal_err'].append(float(irs_err))

print("\n  Done. Computing aggregate statistics...")

# ============================================================
# COMPUTE MEAN RESULTS PER PREFERENCE
# ============================================================
METHODS = ['your_system', 'baseline_rand',
           'baseline_fixed', 'baseline_irs']
METHOD_LABELS = ['Proposed\nSystem', 'Random\nConfig',
                 'Fixed WF\n(W2)', 'IRS-Only\n(W2)']
METHOD_COLORS = ['steelblue', 'tomato', 'darkorange', 'mediumpurple']

summary = {}
for pref in PREF_LIST:
    summary[pref] = {}
    for method in METHODS:
        summary[pref][method] = {
            'BER_mean':  np.mean(results[pref][method]['BER']),
            'TP_mean':   np.mean(results[pref][method]['TP']),
            'Lat_mean':  np.mean(results[pref][method]['Lat']),
            'E_mean':    np.mean(results[pref][method]['E']),
            'goal_mean': np.mean(results[pref][method]['goal_err'])
        }

# Print summary table
print(f"\n{'Preference':<22} {'Method':<18} "
      f"{'BER':>10} {'Tput':>8} {'Lat(ms)':>9} {'GoalErr':>10}")
print("-" * 80)
for pref in PREF_LIST:
    for method, mlabel in zip(METHODS, METHOD_LABELS):
        s = summary[pref][method]
        print(f"  {pref:<20} {method:<18} "
              f"{s['BER_mean']:>10.4f} {s['TP_mean']:>8.3f} "
              f"{s['Lat_mean']:>9.4f} {s['goal_mean']:>10.4f}")
    print()

# ============================================================
# PLOT 6 — GOAL ERROR COMPARISON (1x8 bar chart per preference)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

for pi, (pref, short) in enumerate(zip(PREF_LIST, SHORT)):
    ax = axes[pi]
    vals   = [summary[pref][m]['goal_mean'] for m in METHODS]
    bars   = ax.bar(METHOD_LABELS, vals, color=METHOD_COLORS,
                    edgecolor='white', alpha=0.85)

    # Highlight best (lowest)
    best_i = int(np.argmin(vals))
    bars[best_i].set_edgecolor('black')
    bars[best_i].set_linewidth(2.5)

    ax.set_title(f'{short} — {pref[:12]}', fontsize=9)
    ax.set_ylabel('Mean Goal Error')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=7)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

plt.suptitle('Goal Error Comparison — Proposed vs Baselines\n'
             '(Lower is Better, Black Border = Best)',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/step4_goal_error_comparison.png', dpi=150)
plt.show()
print("\nSaved: step4_goal_error_comparison.png")

# ============================================================
# PLOT 7 — MAIN RESULTS: BER + THROUGHPUT + LATENCY + ENERGY
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 10))

metrics_plot  = ['BER_mean', 'TP_mean', 'Lat_mean', 'E_mean']
metric_labels = ['Mean BER', 'Mean Throughput (bps/Hz)',
                 'Mean Latency (ms)', 'Mean Energy/bit (J)']
better        = ['lower', 'higher', 'lower', 'lower']

for row in range(2):
    metric_key = metrics_plot[row * 2 if row == 0 else row]

for mi, (mk, ml, bt) in enumerate(
        zip(metrics_plot, metric_labels, better)):

    ax  = axes[mi // 4][mi % 4] if len(axes.shape) > 1 else axes[mi]
    ax  = axes.ravel()[mi]

    x   = np.arange(len(PREF_LIST))
    w   = 0.18
    off = [-1.5, -0.5, 0.5, 1.5]

    for j, (method, mlabel, color) in enumerate(
            zip(METHODS, METHOD_LABELS, METHOD_COLORS)):
        vals = [summary[p][method][mk] for p in PREF_LIST]
        ax.bar(x + off[j]*w, vals, w*0.9, label=mlabel.replace('\n', ' '),
               color=color, edgecolor='white', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=8)
    ax.set_ylabel(ml, fontsize=9)
    ax.set_title(f'{ml}\n({bt} is better)', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    if mi == 0:
        ax.legend(fontsize=7, loc='upper right')

plt.suptitle('Performance Comparison Across Preferences and Methods',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/step4_performance_comparison.png', dpi=150)
plt.show()
print("Saved: step4_performance_comparison.png")

# ============================================================
# PLOT 8 — CONFIG SELECTION HEATMAP
# ============================================================
print("\n  Computing config selection heatmap...")

selection_counts = np.zeros((len(PREF_LIST), len(cands)), dtype=int)

N_HEATMAP = 200   # channel realizations per preference
df_heat   = df_test.copy()

for pi, (pref, pref_idx) in enumerate(
        zip(PREF_LIST, range(1, 9))):
    pref_df  = df_heat[df_heat['preference_label'] == pref]
    n_sample = min(N_HEATMAP, len(pref_df))
    sample   = pref_df.sample(n_sample, random_state=99)

    for _, row in sample.iterrows():
        channel_row = {c: float(row[c]) for c in CH_COLS}
        _, _, _, all_perf = select_best_config(
            channel_row, pref, pref_idx)
        errors   = compute_goal_errors(all_perf, PREF_WEIGHTS[pref], ref)
        best_idx = int(np.argmin(errors))
        selection_counts[pi, best_idx] += 1

fig, ax = plt.subplots(figsize=(18, 6))
im = ax.imshow(selection_counts, cmap='YlOrRd', aspect='auto')
ax.set_yticks(range(len(PREF_LIST)))
ax.set_yticklabels(PREF_LIST, fontsize=9)
ax.set_xlabel('Config Index (IRS × Waveform, 0–103)')
ax.set_title('Config Selection Frequency Heatmap\n'
             '(Each preference selects different configs → goal-orientation confirmed)')
plt.colorbar(im, ax=ax, label='Selection Count')
plt.tight_layout()
plt.savefig(f'{OUT}/step4_config_selection_heatmap.png', dpi=150)
plt.show()
print("Saved: step4_config_selection_heatmap.png")

# ============================================================
# PLOT 9 — BER VS SNR BY PREFERENCE (your system vs best baseline)
# ============================================================
print("\n  BER vs SNR analysis...")

snr_bins  = sorted(df_test['snr_input_dB'].unique())
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes      = axes.ravel()

for pi, (pref, short) in enumerate(zip(PREF_LIST, SHORT)):
    ax       = axes[pi]
    pref_idx = pi + 1
    weights  = PREF_WEIGHTS[pref]
    pref_df  = df_test[df_test['preference_label'] == pref]

    snr_sys  = []
    snr_rand = []
    snr_ax_v = []

    for snr_v in snr_bins:
        snr_df = pref_df[pref_df['snr_input_dB'] == snr_v]
        if len(snr_df) == 0:
            continue
        sample = snr_df.sample(min(15, len(snr_df)), random_state=7)
        sys_bers, rand_bers = [], []

        for _, row in sample.iterrows():
            channel_row = {c: float(row[c]) for c in CH_COLS}
            _, perf_s, errs_s, all_p = select_best_config(
                channel_row, pref, pref_idx)
            sys_bers.append(10 ** perf_s[0])

            ri      = np.random.randint(0, len(cands))
            rand_p  = all_p[ri]
            rand_bers.append(10 ** rand_p[0])

        snr_ax_v.append(snr_v)
        snr_sys.append(np.mean(sys_bers))
        snr_rand.append(np.mean(rand_bers))

    if snr_ax_v:
        ax.semilogy(snr_ax_v, snr_sys,  'b-o', linewidth=2,
                    markersize=5, label='Proposed')
        ax.semilogy(snr_ax_v, snr_rand, 'r--s', linewidth=1.5,
                    markersize=4, label='Random')
        ax.axhline(1e-3, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel('SNR (dB)', fontsize=8)
    ax.set_ylabel('BER', fontsize=8)
    ax.set_title(f'{short}', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle('BER vs SNR — Proposed System vs Random Baseline\n'
             'by User Preference', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/step4_ber_vs_snr_by_preference.png', dpi=150)
plt.show()
print("Saved: step4_ber_vs_snr_by_preference.png")

# ============================================================
# SAVE RESULTS
# ============================================================
all_results = {
    'summary':           summary,
    'results':           results,
    'selection_counts':  selection_counts,
    'PREF_LIST':         PREF_LIST,
    'SHORT':             SHORT,
    'METHODS':           METHODS,
    'METHOD_LABELS':     METHOD_LABELS
}
pickle.dump(all_results,
            open(f'{OUT}/decision_engine_results.pkl', 'wb'))
print(f"\nSaved: decision_engine_results.pkl")

# ============================================================
# FINAL SUMMARY PRINT
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3 COMPLETE — FINAL SUMMARY")
print("=" * 60)

m1_metrics = pickle.load(open(f'{OUT}/model1_metrics.pkl', 'rb'))
m2_metrics = pickle.load(open(f'{OUT}/model2_metrics.pkl', 'rb'))

print("\nModel 1 — Performance Predictor:")
for name, vals in m1_metrics.items():
    print(f"  {name:<25} R²={vals['R2']:.4f}  MAE={vals['MAE']:.4f}")

print("\nModel 2 — Channel Transition LSTM:")
for name, vals in m2_metrics.items():
    print(f"  {name:<28} R²={vals['R2']:.4f}  MAE={vals['MAE']:.4f}")

print("\nDecision Engine — Goal Error (lower = better):")
for pref, short in zip(PREF_LIST, SHORT):
    sys_err  = summary[pref]['your_system']['goal_mean']
    rand_err = summary[pref]['baseline_rand']['goal_mean']
    improv   = (rand_err - sys_err) / (abs(rand_err) + 1e-8) * 100
    print(f"  {short:<5} {pref:<22} "
          f"System={sys_err:.4f}  Random={rand_err:.4f}  "
          f"Improvement={improv:.1f}%")

print("\nOutputs saved to: phase3_outputs/")
print("=" * 60)
