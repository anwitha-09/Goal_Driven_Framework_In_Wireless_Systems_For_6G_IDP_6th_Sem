"""
=============================================================
PHASE 3 — STEP 3: Model 2 — Channel Transition (LSTM)
Project: Goal-Driven DT-Based Adaptive IRS & Waveform
         Optimization for 6G Wireless Systems
=============================================================
INPUT  : phase3_outputs/ (from step1)
OUTPUTS: phase3_outputs/model2_channel_transition.pth
         phase3_outputs/model2_metrics.pkl
         phase3_outputs/step3_model2_*.png
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, os, time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

OUT = 'phase3_outputs'
assert os.path.exists(OUT), "Run step1 first"

# ============================================================
# LOAD DATA
# ============================================================
print("\n=== Loading preprocessed data ===")

X_train = torch.FloatTensor(np.load(f'{OUT}/X2_train.npy')).to(device)
X_val   = torch.FloatTensor(np.load(f'{OUT}/X2_val.npy')).to(device)
X_test  = torch.FloatTensor(np.load(f'{OUT}/X2_test.npy')).to(device)
y_train = torch.FloatTensor(np.load(f'{OUT}/y2_train.npy')).to(device)
y_val   = torch.FloatTensor(np.load(f'{OUT}/y2_val.npy')).to(device)
y_test  = torch.FloatTensor(np.load(f'{OUT}/y2_test.npy')).to(device)

sc_y = pickle.load(open(f'{OUT}/scaler_m2_y.pkl', 'rb'))
meta = pickle.load(open(f'{OUT}/metadata.pkl', 'rb'))

print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_val  : {X_val.shape}    y_val  : {y_val.shape}")
print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")

N_IN  = X_train.shape[1]   # 19
N_OUT = y_train.shape[1]   # 8

# ============================================================
# MODEL ARCHITECTURE — LSTM
# ============================================================
class ChannelTransitionLSTM(nn.Module):
    """
    2-layer LSTM for predicting next channel state.
    
    Input shape:  (batch, seq_len=1, n_features=19)
    Output shape: (batch, 8)  — next channel state features
    
    The LSTM learns state transition dynamics:
      channel(t) + config → channel(t+1)
    Single-step prediction captures temporal channel evolution.
    """
    def __init__(self, n_in=19, n_out=8,
                 hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=n_in, hidden_size=hidden1,
            batch_first=True, dropout=0.0
        )
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden1, hidden_size=hidden2,
            batch_first=True, dropout=0.0
        )
        self.drop2  = nn.Dropout(dropout)
        self.fc1    = nn.Linear(hidden2, 64)
        self.fc2    = nn.Linear(64, 32)
        self.out    = nn.Linear(32, n_out)
        self.relu   = nn.ReLU()

    def forward(self, x):
        # x: (batch, 1, 19)
        out, _  = self.lstm1(x)            # (batch, 1, 128)
        out     = self.drop1(out)
        out, _  = self.lstm2(out)          # (batch, 1, 64)
        out     = out[:, -1, :]            # (batch, 64) — last step
        out     = self.drop2(out)
        out     = self.relu(self.fc1(out)) # (batch, 64)
        out     = self.relu(self.fc2(out)) # (batch, 32)
        out     = self.out(out)            # (batch, 8)
        return out

model = ChannelTransitionLSTM(N_IN, N_OUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel 2 — ChannelTransitionLSTM")
print(f"  Parameters: {total_params:,}")
print(f"  Architecture: LSTM({N_IN}→128) → LSTM(128→64) "
      f"→ Dense(64) → Dense(32) → {N_OUT}")

# ============================================================
# DATALOADERS
# For LSTM, reshape X to (batch, seq_len=1, features)
# ============================================================
BATCH_SIZE = 128

def make_lstm_dataset(X, y):
    # Add sequence dimension: (batch, 1, features)
    X_seq = X.unsqueeze(1)   # (N, 1, 19)
    return TensorDataset(X_seq, y)

train_ds = make_lstm_dataset(X_train, y_train)
val_ds   = make_lstm_dataset(X_val,   y_val)
test_ds  = make_lstm_dataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False)

# ============================================================
# LOSS — MSE + physics constraint
# ============================================================
mse_loss = nn.MSELoss()

def physics_constrained_loss(pred_scaled, target_scaled,
                              input_scaled, sc_y, sc_X, lambda_c=0.1):
    """
    MSE loss + soft physics constraint:
      |next_SNR - current_SNR| should be small (channel changes slowly)
    
    pred_scaled  : (batch, 8) scaled predictions
    input_scaled : (batch, 1, 19) scaled inputs
    """
    # Primary MSE loss
    loss = mse_loss(pred_scaled, target_scaled)

    # Physics constraint: first output is next_snr_dB
    # In scaled domain, check pred SNR is close to input SNR
    # Input col 0 = snr_input_dB (first feature)
    curr_snr_scaled = input_scaled[:, 0, 0]  # (batch,)
    pred_snr_scaled = pred_scaled[:, 0]       # (batch,)
    constraint = torch.relu(torch.abs(pred_snr_scaled - curr_snr_scaled) - 2.0)
    loss = loss + lambda_c * constraint.mean()
    return loss

# ============================================================
# TRAINING SETUP
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)

MAX_EPOCHS    = 100
EARLY_STOP    = 10
best_val_loss = float('inf')
patience_ctr  = 0
best_state    = None

train_losses = []
val_losses   = []

# ============================================================
# TRAINING LOOP
# ============================================================
print(f"\n=== Training Model 2 ===")
print(f"  Epochs: {MAX_EPOCHS} | Batch: {BATCH_SIZE} | "
      f"Early stop: {EARLY_STOP}")
print(f"  Loss: MSE + Physics Constraint\n")
print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
      f"{'LR':>10} | {'Status'}")
print("-" * 65)

t_start = time.time()

for epoch in range(1, MAX_EPOCHS + 1):

    # --- Train ---
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = physics_constrained_loss(pred, yb, xb, sc_y, None)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    train_loss = epoch_loss / len(train_ds)

    # --- Validate ---
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss_sum += mse_loss(pred, yb).item() * len(xb)
    val_loss = val_loss_sum / len(val_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    status = ''
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        patience_ctr  = 0
        status = '✓ best'
    else:
        patience_ctr += 1
        if patience_ctr >= EARLY_STOP:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0 or epoch <= 5 or status:
        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
              f"{current_lr:>10.2e} | {status}")

t_elapsed = time.time() - t_start
print(f"\nTraining complete: {t_elapsed:.1f}s | "
      f"Best val loss: {best_val_loss:.6f}")

model.load_state_dict(best_state)

# ============================================================
# EVALUATION ON TEST SET
# ============================================================
print("\n=== Evaluating Model 2 on Test Set ===")

model.eval()
all_preds   = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

pred_scaled   = np.vstack(all_preds)
target_scaled = np.vstack(all_targets)

pred_orig   = sc_y.inverse_transform(pred_scaled)
target_orig = sc_y.inverse_transform(target_scaled)

target_names = meta['M2_TARGETS']

metrics = {}
print(f"\n{'Feature':<28} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
print("-" * 60)

for i, name in enumerate(target_names):
    mae  = mean_absolute_error(target_orig[:, i], pred_orig[:, i])
    rmse = np.sqrt(np.mean((target_orig[:, i] - pred_orig[:, i])**2))
    r2   = r2_score(target_orig[:, i], pred_orig[:, i])
    metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"  {name:<26} {mae:>10.4f} {rmse:>10.4f} {r2:>8.4f}")

# ============================================================
# PLOT 4 — TRAINING CURVES
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
epochs_ran = range(1, len(train_losses) + 1)
ax.plot(epochs_ran, train_losses, 'b-', linewidth=2, label='Train Loss')
ax.plot(epochs_ran, val_losses,   'r-', linewidth=2, label='Val Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.set_title('Model 2 — LSTM Training Curves')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/step3_model2_training_curves.png', dpi=150)
plt.show()
print("\nSaved: step3_model2_training_curves.png")

# ============================================================
# PLOT 5 — PREDICTED VS ACTUAL (2x4)
# ============================================================
display_names = [
    'Next SNR (dB)', 'Next Channel Gain', 'Next Eff. Rank',
    'Next Path Loss (dB)', 'Next Delay Spread', 'Next Doppler (Hz)',
    'Next Shadow (dB)', 'Next Distance (m)'
]
colors = ['steelblue','seagreen','tomato','mediumpurple',
          'darkorange','teal','crimson','olive']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

for i, (name, dname, color) in enumerate(
        zip(target_names, display_names, colors)):

    ax   = axes[i]
    true = target_orig[:, i]
    pred = pred_orig[:, i]
    r2   = metrics[name]['R2']
    mae  = metrics[name]['MAE']

    idx  = np.random.choice(len(true), min(2000, len(true)), replace=False)
    ax.scatter(true[idx], pred[idx], alpha=0.3, s=5, color=color)

    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.5)

    ax.set_xlabel(f'Actual {dname}', fontsize=8)
    ax.set_ylabel(f'Predicted {dname}', fontsize=8)
    ax.set_title(f'{dname}\nR²={r2:.3f}  MAE={mae:.4f}', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Model 2 (LSTM) — Predicted vs Actual Next Channel State',
             fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/step3_model2_pred_vs_actual.png', dpi=150)
plt.show()
print("Saved: step3_model2_pred_vs_actual.png")

# ============================================================
# PLOT 6 — SNR PREDICTION DETAIL (most important feature)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: predicted vs actual next SNR
ax = axes[0]
true_snr = target_orig[:, 0]
pred_snr = pred_orig[:, 0]
idx = np.random.choice(len(true_snr), min(3000, len(true_snr)), replace=False)
ax.scatter(true_snr[idx], pred_snr[idx], alpha=0.4, s=8, color='steelblue')
lims = [true_snr.min(), true_snr.max()]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
ax.set_xlabel('Actual Next SNR (dB)')
ax.set_ylabel('Predicted Next SNR (dB)')
ax.set_title(f"Next SNR Prediction\nR²={metrics['next_snr_dB']['R2']:.4f}")
ax.legend(); ax.grid(True, alpha=0.3)

# Error distribution
ax = axes[1]
snr_error = pred_snr - true_snr
ax.hist(snr_error, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.axvline(snr_error.mean(), color='orange', linestyle='-',
           linewidth=2, label=f'Mean={snr_error.mean():.3f}')
ax.set_xlabel('Prediction Error (dB)')
ax.set_ylabel('Count')
ax.set_title('Next SNR Prediction Error Distribution')
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Model 2 — SNR Transition Prediction Detail', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/step3_model2_snr_detail.png', dpi=150)
plt.show()
print("Saved: step3_model2_snr_detail.png")

# ============================================================
# SAVE
# ============================================================
torch.save(model.state_dict(),
           f'{OUT}/model2_channel_transition.pth')
pickle.dump(metrics, open(f'{OUT}/model2_metrics.pkl', 'wb'))

model_config = {'n_in': N_IN, 'n_out': N_OUT}
pickle.dump(model_config, open(f'{OUT}/model2_config.pkl', 'wb'))

print(f"\nSaved: model2_channel_transition.pth")
print(f"Saved: model2_metrics.pkl")

print("\n" + "=" * 60)
print("STEP 3 COMPLETE — Model 2 trained and evaluated")
print("Run: python phase3_step4_decision_engine.py")
print("=" * 60)
