"""
=============================================================
PHASE 4 — Decision Bridge (FIXED)
Reads channel_input.csv, runs decision engine,
writes config_output.csv

FIX: Added preference-aware candidate filtering so that
extreme goals (MaxReliability, UltraLowLatency etc.) only
consider appropriate waveform candidates before scoring.
=============================================================
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import sys
import os

# ============================================================
# PATH TO YOUR PHASE 3 OUTPUTS
# ============================================================
OUT = r'C:\Users\ADMIN\Downloads\IDP_PYTHON\phase3_outputs'

# ============================================================
# LOAD MODELS
# ============================================================
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

# Load everything
cfg1     = pickle.load(open(f'{OUT}/model1_config.pkl', 'rb'))
cfg2     = pickle.load(open(f'{OUT}/model2_config.pkl', 'rb'))
sc_m1_X  = pickle.load(open(f'{OUT}/scaler_m1_X.pkl', 'rb'))
sc_m1_y  = pickle.load(open(f'{OUT}/scaler_m1_y.pkl', 'rb'))
sc_m2_X  = pickle.load(open(f'{OUT}/scaler_m2_X.pkl', 'rb'))
sc_m2_y  = pickle.load(open(f'{OUT}/scaler_m2_y.pkl', 'rb'))
ref      = pickle.load(open(f'{OUT}/reference_values.pkl', 'rb'))
cands    = pickle.load(open(f'{OUT}/candidates_list.pkl', 'rb'))

model1 = PerformancePredictor(cfg1['n_in'], cfg1['n_out'])
model1.load_state_dict(torch.load(f'{OUT}/model1_performance_predictor.pth',
                                   map_location='cpu'))
model1.eval()

model2 = ChannelTransitionLSTM(cfg2['n_in'], cfg2['n_out'])
model2.load_state_dict(torch.load(f'{OUT}/model2_channel_transition.pth',
                                   map_location='cpu'))
model2.eval()

# ============================================================
# PREFERENCE WEIGHTS
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
PREF_LIST = list(PREF_WEIGHTS.keys())

# ============================================================
# PREFERENCE-AWARE CANDIDATE FILTERING
# Restricts which waveforms are eligible per goal
# WF candidates: [scs_kHz, cp, mod]
#   W1: [15,16,1] QPSK   long CP  → most robust
#   W2: [15,16,2] 16QAM  long CP  → balanced baseline
#   W3: [15, 8,2] 16QAM  short CP
#   W4: [30, 8,2] 16QAM  short CP
#   W5: [30, 8,3] 64QAM  short CP
#   W6: [60, 4,3] 64QAM  v.short  → ultra low latency
#   W7: [15,16,3] 64QAM  long CP  → max throughput
#   W8: [30,16,1] QPSK   long CP  → robust
# ============================================================
ALLOWED_WF = {
    # MaxReliability → only QPSK waveforms (mod=1) + long CP
    'MaxReliability':   lambda c: c['wf_mod'] == 1,
    # MaxThroughput → only 64QAM waveforms (mod=3)
    'MaxThroughput':    lambda c: c['wf_mod'] == 3,
    # UltraLowLatency → only short CP or high SCS waveforms
    'UltraLowLatency':  lambda c: c['wf_scs_kHz'] >= 30 or c['wf_cp'] <= 8,
    # EnergyEfficient → low modulation, long CP
    'EnergyEfficient':  lambda c: c['wf_mod'] == 1 and c['wf_cp'] == 16,
    # Balanced → all waveforms allowed
    'Balanced':         lambda c: True,
    # ReliabilitySpeed → 16QAM or QPSK
    'ReliabilitySpeed': lambda c: c['wf_mod'] <= 2,
    # GreenThroughput → 16QAM (balance throughput + energy)
    'GreenThroughput':  lambda c: c['wf_mod'] == 2,
    # MissionCritical → QPSK + short latency
    'MissionCritical':  lambda c: c['wf_mod'] == 1 and c['wf_scs_kHz'] >= 30,
}

# ============================================================
# READ CHANNEL INPUT
# ============================================================
input_path = r'C:\Users\ADMIN\Downloads\IDP_PYTHON\channel_input.csv'
df_in = pd.read_csv(input_path)

preference_label = str(df_in['preference_label'].iloc[0])
pref_idx = PREF_LIST.index(preference_label) + 1
weights  = PREF_WEIGHTS[preference_label]

channel_row = {
    'snr_input_dB':      float(df_in['snr_input_dB'].iloc[0]),
    'sinr_dB':           float(df_in['sinr_dB'].iloc[0]),
    'channel_gain':      float(df_in['channel_gain'].iloc[0]),
    'effective_rank':    float(df_in['effective_rank'].iloc[0]),
    'path_loss_dB':      float(df_in['path_loss_dB'].iloc[0]),
    'shadow_dB':         float(df_in['shadow_dB'].iloc[0]),
    'rms_delay_spread':  float(df_in['rms_delay_spread'].iloc[0]),
    'doppler_spread_Hz': float(df_in['doppler_spread_Hz'].iloc[0]),
    'distance_m':        float(df_in['distance_m'].iloc[0]),
    'irs_gain_dB':       float(df_in['irs_gain_dB'].iloc[0]),
    'AoD_mean':          float(df_in['AoD_mean'].iloc[0]),
    'AoA_mean':          float(df_in['AoA_mean'].iloc[0]),
}

# Filter candidates based on preference
wf_filter  = ALLOWED_WF[preference_label]
cands_filt = [c for c in cands if wf_filter(c)]
cands_idx  = [i for i, c in enumerate(cands) if wf_filter(c)]

print(f"Preference: {preference_label}")
print(f"Candidates after filtering: {len(cands_filt)} of {len(cands)}")

# ============================================================
# DECISION ENGINE
# ============================================================
def build_m2_batch(channel_row, candidates, pref_idx):
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
    return np.array(rows, dtype=np.float32)

def build_m1_batch(next_channel_batch, candidates, weights):
    rows = []
    for i, c in enumerate(candidates):
        nc = next_channel_batch[i]
        row = [
            nc[0], 0.0, nc[1], nc[2], nc[3], nc[6],
            nc[4], nc[5], nc[7], 0.0,
            c['irs_phase_mean'], c['irs_phase_std'], c['irs_is_dft'],
            c['wf_scs_kHz'], c['wf_cp'], c['wf_mod'],
            weights[0], weights[1], weights[2], weights[3],
            0.0, 0.0, 0.0, 0.0
        ]
        rows.append(row)
    return np.array(rows, dtype=np.float32)

def compute_goal_errors(pred_orig, weights, ref):
    errors = (
        weights[0] * (pred_orig[:, 0] - ref['BER_best']) / (ref['BER_ref'] + 1e-8) +
        weights[1] * (ref['TP_best'] - pred_orig[:, 1])  / (ref['TP_ref']  + 1e-8) +
        weights[2] * (pred_orig[:, 2] - ref['Lat_best']) / (ref['Lat_ref'] + 1e-8) +
        weights[3] * (pred_orig[:, 3] - ref['E_best'])   / (ref['E_ref']   + 1e-8)
    )
    return errors

# Run Model 2 on filtered candidates
m2_in  = build_m2_batch(channel_row, cands_filt, pref_idx)
m2_sc  = sc_m2_X.transform(m2_in)
m2_t   = torch.FloatTensor(m2_sc).unsqueeze(1)
with torch.no_grad():
    next_ch_sc = model2(m2_t).numpy()
next_ch = sc_m2_y.inverse_transform(next_ch_sc)

# Run Model 1
m1_in  = build_m1_batch(next_ch, cands_filt, weights)
m1_sc  = sc_m1_X.transform(m1_in)
m1_t   = torch.FloatTensor(m1_sc)
with torch.no_grad():
    perf_sc = model1(m1_t).numpy()
perf = sc_m1_y.inverse_transform(perf_sc)

# Pick best among filtered candidates
errors   = compute_goal_errors(perf, weights, ref)
best_idx = int(np.argmin(errors))
best     = cands_filt[best_idx]
best_perf = perf[best_idx]

# ============================================================
# WRITE OUTPUT
# ============================================================
output_path = r'C:\Users\ADMIN\Downloads\IDP_PYTHON\config_output.csv'

out_df = pd.DataFrame([{
    'irs_candidate_idx':      best['irs_idx'],
    'irs_phase_mean':         best['irs_phase_mean'],
    'irs_phase_std':          best['irs_phase_std'],
    'irs_is_dft':             best['irs_is_dft'],
    'wf_scs_kHz':             best['wf_scs_kHz'],
    'wf_cp':                  best['wf_cp'],
    'wf_mod':                 best['wf_mod'],
    'predicted_log_BER':      best_perf[0],
    'predicted_throughput':   best_perf[1],
    'predicted_latency_ms':   best_perf[2],
    'predicted_energy':       best_perf[3],
    'preference_label':       preference_label
}])

out_df.to_csv(output_path, index=False)
print(f"DONE: Best config written to {output_path}")
print(f"IRS idx={best['irs_idx']} | WF={best['wf_scs_kHz']}kHz CP={best['wf_cp']} MOD={best['wf_mod']}")
print(f"Predicted BER={10**best_perf[0]:.6f} | Tput={best_perf[1]:.4f} | Lat={best_perf[2]:.4f}ms")
