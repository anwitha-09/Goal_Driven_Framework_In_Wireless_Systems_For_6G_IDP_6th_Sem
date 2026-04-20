"""
=============================================================
PHASE 3 — STEP 1: Data Loading & Preprocessing
Project: Goal-Driven DT-Based Adaptive IRS & Waveform
         Optimization for 6G Wireless Systems
=============================================================
RUN THIS FIRST before any model training.
Outputs: preprocessed data splits + 4 saved scalers
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ============================================================
# 1.1  LOAD
# ============================================================
print("=" * 60)
print("PHASE 3 — STEP 1: Data Preprocessing")
print("=" * 60)

CSV_PATH = 'dataset_6G_DT.csv'
assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"

df = pd.read_csv(CSV_PATH)
print(f"\nLoaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ============================================================
# 1.2  BASIC HEALTH CHECK
# ============================================================
print("\n--- Health Check ---")

# Missing values
missing = df.isnull().sum()
n_missing = missing.sum()
print(f"Missing values: {n_missing}")

# Infinite values
num_cols = df.select_dtypes(include=[np.number]).columns
inf_cols = [c for c in num_cols if np.isinf(df[c]).any()]
print(f"Inf values in: {inf_cols if inf_cols else 'None'}")

# Fix
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
print("Inf/NaN handled.")

# Verify all 8 preferences present
print(f"\nPreference distribution:")
pref_counts = df['preference_label'].value_counts().sort_index()
for pref, cnt in pref_counts.items():
    print(f"  {pref:<25} {cnt:>7} rows")

# Key metric ranges
print(f"\nKey metric ranges:")
print(f"  BER        : [{df['BER'].min():.4f}, {df['BER'].max():.4f}]")
print(f"  Throughput : [{df['throughput_bpsHz'].min():.3f}, "
      f"{df['throughput_bpsHz'].max():.3f}] bps/Hz")
print(f"  Latency    : [{df['latency_ms'].min():.4f}, "
      f"{df['latency_ms'].max():.4f}] ms")
print(f"  SNR input  : [{df['snr_input_dB'].min():.1f}, "
      f"{df['snr_input_dB'].max():.1f}] dB")

# ============================================================
# 1.3  FEATURE ENGINEERING
# ============================================================
print("\n--- Feature Engineering ---")

# Log-transform BER (spans 5 orders of magnitude)
df['log_BER']          = np.log10(df['BER']          + 1e-10)
df['log_baseline_BER'] = np.log10(df['baseline_BER'] + 1e-10)

# Improvement delta features
df['BER_improvement'] = df['log_baseline_BER'] - df['log_BER']
df['TP_improvement']  = df['throughput_bpsHz'] - df['baseline_throughput']
df['Lat_improvement'] = df['baseline_latency'] - df['latency_ms']
df['E_improvement']   = df['baseline_energy']  - df['energy_per_bit']

print(f"  log_BER range        : [{df['log_BER'].min():.2f}, "
      f"{df['log_BER'].max():.2f}]")
print(f"  BER_improvement mean : {df['BER_improvement'].mean():.4f}")
print(f"  TP_improvement  mean : {df['TP_improvement'].mean():.4f}")
print("  Feature engineering done.")

# ============================================================
# 1.4  DEFINE FEATURE SETS
# ============================================================

# ---------- MODEL 1: Performance Predictor ----------
M1_FEATURES = [
    # Channel state (10)
    'snr_input_dB', 'sinr_dB', 'channel_gain', 'effective_rank',
    'path_loss_dB', 'shadow_dB', 'rms_delay_spread', 'doppler_spread_Hz',
    'distance_m', 'irs_gain_dB',
    # IRS config (3)
    'irs_phase_mean', 'irs_phase_std', 'irs_is_dft',
    # Waveform (3)
    'subcarrier_spacing_kHz', 'cp_length', 'mod_order',
    # Goal weights (4)
    'w_BER', 'w_Throughput', 'w_Latency', 'w_Energy',
    # Delta improvement features (4)
    'BER_improvement', 'TP_improvement', 'Lat_improvement', 'E_improvement'
]  # total = 24 features

M1_TARGETS = [
    'log_BER', 'throughput_bpsHz', 'latency_ms', 'energy_per_bit'
]  # 4 targets

# Also keep the Pareto weights as a separate array for weighted loss
M1_WEIGHT_COLS = ['w_BER', 'w_Throughput', 'w_Latency', 'w_Energy']

# ---------- MODEL 2: Channel Transition (LSTM) ----------
M2_FEATURES = [
    # Channel state (12)
    'snr_input_dB', 'sinr_dB', 'channel_gain', 'effective_rank',
    'path_loss_dB', 'shadow_dB', 'rms_delay_spread', 'doppler_spread_Hz',
    'distance_m', 'irs_gain_dB', 'AoD_mean', 'AoA_mean',
    # IRS config (3)
    'irs_phase_mean', 'irs_phase_std', 'irs_is_dft',
    # Waveform (4)
    'subcarrier_spacing_kHz', 'cp_length', 'mod_order', 'preference_idx'
]  # total = 19 features

M2_TARGETS = [
    'next_snr_dB', 'next_channel_gain', 'next_effective_rank',
    'next_path_loss_dB', 'next_rms_delay_spread', 'next_doppler_Hz',
    'next_shadow_dB', 'next_distance_m'
]  # 8 targets

print(f"\n--- Feature Dimensions ---")
print(f"  Model 1 inputs  : {len(M1_FEATURES)} features")
print(f"  Model 1 targets : {len(M1_TARGETS)} features")
print(f"  Model 2 inputs  : {len(M2_FEATURES)} features")
print(f"  Model 2 targets : {len(M2_TARGETS)} features")

# ============================================================
# 1.5  TRAIN / VAL / TEST SPLIT (stratified by preference)
# ============================================================
print("\n--- Train/Val/Test Split (stratified by preference) ---")

# First: 85% train+val, 15% test
df_trainval, df_test = train_test_split(
    df, test_size=0.15, random_state=42,
    stratify=df['preference_idx']
)

# Second: 70% train, 15% val (from the 85%)
df_train, df_val = train_test_split(
    df_trainval, test_size=0.1765, random_state=42,
    stratify=df_trainval['preference_idx']
)
# 0.1765 * 0.85 ≈ 0.15 of total

print(f"  Train : {len(df_train):>7} rows ({len(df_train)/len(df)*100:.1f}%)")
print(f"  Val   : {len(df_val):>7} rows ({len(df_val)/len(df)*100:.1f}%)")
print(f"  Test  : {len(df_test):>7} rows ({len(df_test)/len(df)*100:.1f}%)")

# Verify stratification
for split_name, split_df in [('Train', df_train),
                               ('Val',   df_val),
                               ('Test',  df_test)]:
    counts = split_df['preference_idx'].value_counts().sort_index()
    print(f"  {split_name} preference counts: "
          f"{counts.values.tolist()}")

# ============================================================
# 1.6  SCALE FEATURES
# ============================================================
print("\n--- Feature Scaling (fit on train only) ---")

def fit_scaler(train_df, columns):
    sc = StandardScaler()
    sc.fit(train_df[columns].values)
    return sc

def apply_scaler(sc, df, columns):
    return sc.transform(df[columns].values).astype(np.float32)

# Model 1 scalers
sc_m1_X = fit_scaler(df_train, M1_FEATURES)
sc_m1_y = fit_scaler(df_train, M1_TARGETS)

# Model 2 scalers
sc_m2_X = fit_scaler(df_train, M2_FEATURES)
sc_m2_y = fit_scaler(df_train, M2_TARGETS)

# Apply to all splits
X1_train = apply_scaler(sc_m1_X, df_train, M1_FEATURES)
X1_val   = apply_scaler(sc_m1_X, df_val,   M1_FEATURES)
X1_test  = apply_scaler(sc_m1_X, df_test,  M1_FEATURES)

y1_train = apply_scaler(sc_m1_y, df_train, M1_TARGETS)
y1_val   = apply_scaler(sc_m1_y, df_val,   M1_TARGETS)
y1_test  = apply_scaler(sc_m1_y, df_test,  M1_TARGETS)

X2_train = apply_scaler(sc_m2_X, df_train, M2_FEATURES)
X2_val   = apply_scaler(sc_m2_X, df_val,   M2_FEATURES)
X2_test  = apply_scaler(sc_m2_X, df_test,  M2_FEATURES)

y2_train = apply_scaler(sc_m2_y, df_train, M2_TARGETS)
y2_val   = apply_scaler(sc_m2_y, df_val,   M2_TARGETS)
y2_test  = apply_scaler(sc_m2_y, df_test,  M2_TARGETS)

# Pareto weights (unscaled — used in loss function)
W1_train = df_train[M1_WEIGHT_COLS].values.astype(np.float32)
W1_val   = df_val[M1_WEIGHT_COLS].values.astype(np.float32)
W1_test  = df_test[M1_WEIGHT_COLS].values.astype(np.float32)

print(f"  X1 train shape: {X1_train.shape}")
print(f"  y1 train shape: {y1_train.shape}")
print(f"  X2 train shape: {X2_train.shape}")
print(f"  y2 train shape: {y2_train.shape}")

# ============================================================
# 1.7  COMPUTE REFERENCE VALUES (for decision engine)
# ============================================================
print("\n--- Reference Values (from training set) ---")

# Use UNSCALED training targets
y1_train_raw = df_train[M1_TARGETS].values

BER_best  = float(df_train['log_BER'].min())
TP_best   = float(df_train['throughput_bpsHz'].max())
Lat_best  = float(df_train['latency_ms'].min())
E_best    = float(df_train['energy_per_bit'].min())

BER_ref   = float(df_train['log_BER'].std())
TP_ref    = float(df_train['throughput_bpsHz'].std())
Lat_ref   = float(df_train['latency_ms'].std())
E_ref     = float(df_train['energy_per_bit'].std())

print(f"  BER  best={BER_best:.2f}  ref(std)={BER_ref:.4f}")
print(f"  TP   best={TP_best:.3f}  ref(std)={TP_ref:.4f}")
print(f"  Lat  best={Lat_best:.4f}  ref(std)={Lat_ref:.4f}")
print(f"  E    best={E_best:.2e}  ref(std)={E_ref:.2e}")

reference_values = {
    'BER_best': BER_best, 'TP_best': TP_best,
    'Lat_best': Lat_best, 'E_best':  E_best,
    'BER_ref':  BER_ref,  'TP_ref':  TP_ref,
    'Lat_ref':  Lat_ref,  'E_ref':   E_ref
}

# ============================================================
# 1.8  BUILD CANDIDATE LIBRARY (104 configs for decision engine)
# ============================================================
print("\n--- Building Candidate Library ---")

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

WF_CANDIDATES = [
    [15, 16, 1], [15, 16, 2], [15, 8, 2], [30, 8, 2],
    [30, 8, 3],  [60, 4, 3],  [15, 16, 3], [30, 16, 1]
]

IRS_STATS = (df_train
             .groupby('irs_candidate_idx')[
                 ['irs_phase_mean', 'irs_phase_std', 'irs_is_dft']
             ].first()
             .sort_index()
             .reset_index())

candidates = []
for _, irs_row in IRS_STATS.iterrows():
    for wf in WF_CANDIDATES:
        candidates.append({
            'irs_idx':         int(irs_row['irs_candidate_idx']),
            'irs_phase_mean':  float(irs_row['irs_phase_mean']),
            'irs_phase_std':   float(irs_row['irs_phase_std']),
            'irs_is_dft':      float(irs_row['irs_is_dft']),
            'wf_scs_kHz':      wf[0],
            'wf_cp':           wf[1],
            'wf_mod':          wf[2]
        })

print(f"  Total candidates: {len(candidates)}")

# ============================================================
# 1.9  SAVE EVERYTHING
# ============================================================
print("\n--- Saving Preprocessed Data ---")

os.makedirs('phase3_outputs', exist_ok=True)

# Scalers
pickle.dump(sc_m1_X, open('phase3_outputs/scaler_m1_X.pkl', 'wb'))
pickle.dump(sc_m1_y, open('phase3_outputs/scaler_m1_y.pkl', 'wb'))
pickle.dump(sc_m2_X, open('phase3_outputs/scaler_m2_X.pkl', 'wb'))
pickle.dump(sc_m2_y, open('phase3_outputs/scaler_m2_y.pkl', 'wb'))

# Reference values and candidates
pickle.dump(reference_values,
            open('phase3_outputs/reference_values.pkl', 'wb'))
pickle.dump(candidates,
            open('phase3_outputs/candidates_list.pkl', 'wb'))

# Data splits as numpy arrays
np.save('phase3_outputs/X1_train.npy', X1_train)
np.save('phase3_outputs/X1_val.npy',   X1_val)
np.save('phase3_outputs/X1_test.npy',  X1_test)
np.save('phase3_outputs/y1_train.npy', y1_train)
np.save('phase3_outputs/y1_val.npy',   y1_val)
np.save('phase3_outputs/y1_test.npy',  y1_test)
np.save('phase3_outputs/W1_train.npy', W1_train)
np.save('phase3_outputs/W1_val.npy',   W1_val)
np.save('phase3_outputs/W1_test.npy',  W1_test)

np.save('phase3_outputs/X2_train.npy', X2_train)
np.save('phase3_outputs/X2_val.npy',   X2_val)
np.save('phase3_outputs/X2_test.npy',  X2_test)
np.save('phase3_outputs/y2_train.npy', y2_train)
np.save('phase3_outputs/y2_val.npy',   y2_val)
np.save('phase3_outputs/y2_test.npy',  y2_test)

# Save test split raw (for evaluation)
df_test.to_csv('phase3_outputs/df_test.csv', index=False)

# Save metadata
meta = {
    'M1_FEATURES':    M1_FEATURES,
    'M1_TARGETS':     M1_TARGETS,
    'M1_WEIGHT_COLS': M1_WEIGHT_COLS,
    'M2_FEATURES':    M2_FEATURES,
    'M2_TARGETS':     M2_TARGETS,
    'PREF_WEIGHTS':   PREF_WEIGHTS,
    'WF_CANDIDATES':  WF_CANDIDATES,
    'n_train':        len(df_train),
    'n_val':          len(df_val),
    'n_test':         len(df_test)
}
pickle.dump(meta, open('phase3_outputs/metadata.pkl', 'wb'))

print("  Saved to phase3_outputs/")
print("  scalers, reference values, candidates, splits, metadata")

# ============================================================
# 1.10  SUMMARY PLOT
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# BER distribution
ax = axes[0, 0]
ax.hist(df['log_BER'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('log10(BER)'); ax.set_ylabel('Count')
ax.set_title('BER Distribution (log scale)'); ax.grid(True, alpha=0.3)

# Throughput distribution
ax = axes[0, 1]
ax.hist(df['throughput_bpsHz'], bins=40, color='seagreen',
        edgecolor='white', alpha=0.8)
ax.set_xlabel('Throughput (bps/Hz)'); ax.set_title('Throughput Distribution')
ax.grid(True, alpha=0.3)

# Latency distribution
ax = axes[0, 2]
ax.hist(df['latency_ms'], bins=40, color='tomato', edgecolor='white', alpha=0.8)
ax.set_xlabel('Latency (ms)'); ax.set_title('Latency Distribution')
ax.grid(True, alpha=0.3)

# Energy distribution
ax = axes[0, 3]
ax.hist(df['energy_per_bit'], bins=40, color='mediumpurple',
        edgecolor='white', alpha=0.8)
ax.set_xlabel('Energy/bit (J)'); ax.set_title('Energy Distribution')
ax.grid(True, alpha=0.3)

# BER vs SNR
ax = axes[1, 0]
snr_vals  = df['snr_input_dB'].values
ber_vals  = df['log_BER'].values
pref_vals = df['preference_idx'].values
sc = ax.scatter(snr_vals, ber_vals, c=pref_vals, cmap='tab10',
                s=1, alpha=0.3)
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('log10(BER)')
ax.set_title('BER vs SNR (color=preference)'); ax.grid(True, alpha=0.3)

# Throughput vs SNR
ax = axes[1, 1]
tp_vals  = df['throughput_bpsHz'].values
ig_vals  = df['irs_gain_dB'].values
ax.scatter(snr_vals, tp_vals, c=ig_vals, cmap='coolwarm',
           s=1, alpha=0.3)
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Throughput (bps/Hz)')
ax.set_title('Throughput vs SNR'); ax.grid(True, alpha=0.3)

# IRS gain distribution
ax = axes[1, 2]
ax.hist(df['irs_gain_dB'], bins=40, color='darkorange',
        edgecolor='white', alpha=0.8)
ax.set_xlabel('IRS Gain (dB)'); ax.set_title('IRS Gain Distribution')
ax.grid(True, alpha=0.3)

# Preference distribution
ax = axes[1, 3]
labels = list(PREF_WEIGHTS.keys())
counts = [len(df[df['preference_label'] == l]) for l in labels]
short  = ['MxR','MxT','ULL','Enr','Bal','R+S','GTP','MC']
bars   = ax.bar(short, counts, color='cornflowerblue', edgecolor='white')
ax.set_title('Preference Distribution'); ax.set_ylabel('Count')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Phase 3 — Step 1: Dataset Overview', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('phase3_outputs/step1_dataset_overview.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: phase3_outputs/step1_dataset_overview.png")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE — Ready for Model Training")
print("Run: python phase3_step2_model1.py")
print("=" * 60)
