"""
=============================================================
PHASE 3 — STEP 2: Model 1 — Performance Predictor (NN)
Project: Goal-Driven DT-Based Adaptive IRS & Waveform
         Optimization for 6G Wireless Systems
=============================================================
INPUT  : phase3_outputs/ (from step1)
OUTPUTS: phase3_outputs/model1_performance_predictor.pth
         phase3_outputs/model1_metrics.pkl
         phase3_outputs/step2_model1_*.png
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

X_train = torch.FloatTensor(np.load(f'{OUT}/X1_train.npy')).to(device)
X_val   = torch.FloatTensor(np.load(f'{OUT}/X1_val.npy')).to(device)
X_test  = torch.FloatTensor(np.load(f'{OUT}/X1_test.npy')).to(device)
y_train = torch.FloatTensor(np.load(f'{OUT}/y1_train.npy')).to(device)
y_val   = torch.FloatTensor(np.load(f'{OUT}/y1_val.npy')).to(device)
y_test  = torch.FloatTensor(np.load(f'{OUT}/y1_test.npy')).to(device)
W_train = torch.FloatTensor(np.load(f'{OUT}/W1_train.npy')).to(device)
W_val   = torch.FloatTensor(np.load(f'{OUT}/W1_val.npy')).to(device)
W_test  = torch.FloatTensor(np.load(f'{OUT}/W1_test.npy')).to(device)

sc_y = pickle.load(open(f'{OUT}/scaler_m1_y.pkl', 'rb'))
meta = pickle.load(open(f'{OUT}/metadata.pkl', 'rb'))

print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_val  : {X_val.shape}    y_val  : {y_val.shape}")
print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")

N_IN  = X_train.shape[1]   # 24
N_OUT = y_train.shape[1]   # 4

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class PerformancePredictor(nn.Module):
    """
    3-layer feedforward NN for predicting communication
    performance metrics given channel + config + goal weights.
    
    Input:  24 features (channel state + IRS + waveform + goal)
    Output: 4 features (log_BER, throughput, latency, energy)
    """
    def __init__(self, n_in=24, n_out=4):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(n_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Output
            nn.Linear(64, n_out)
        )

    def forward(self, x):
        return self.net(x)

model = PerformancePredictor(N_IN, N_OUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel 1 — PerformancePredictor")
print(f"  Parameters: {total_params:,}")
print(f"  Architecture: {N_IN} → 256 → 128 → 64 → {N_OUT}")

# ============================================================
# PARETO-WEIGHTED LOSS FUNCTION
# ============================================================
def pareto_weighted_mse(pred, target, weights):
    """
    Weighted MSE where each output is penalized proportional
    to the user's Pareto preference weight.
    
    Args:
        pred    : (batch, 4) — model predictions
        target  : (batch, 4) — ground truth
        weights : (batch, 4) — Pareto weights [w_BER,w_TP,w_Lat,w_E]
    
    Returns: scalar loss
    """
    mse_per_output = (pred - target) ** 2          # (batch, 4)
    weighted_mse   = weights * mse_per_output       # (batch, 4)
    return weighted_mse.mean()

# ============================================================
# DATALOADERS
# ============================================================
BATCH_SIZE = 256

train_ds = TensorDataset(X_train, y_train, W_train)
val_ds   = TensorDataset(X_val,   y_val,   W_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False)

# ============================================================
# TRAINING SETUP
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

MAX_EPOCHS    = 150
EARLY_STOP    = 15
best_val_loss = float('inf')
patience_ctr  = 0
best_state    = None

train_losses = []
val_losses   = []

# ============================================================
# TRAINING LOOP
# ============================================================
print(f"\n=== Training Model 1 ===")
print(f"  Epochs: {MAX_EPOCHS} | Batch: {BATCH_SIZE} | "
      f"Early stop: {EARLY_STOP}")
print(f"  Loss: Pareto-Weighted MSE\n")
print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
      f"{'LR':>10} | {'Status'}")
print("-" * 65)

t_start = time.time()

for epoch in range(1, MAX_EPOCHS + 1):

    # --- Train ---
    model.train()
    epoch_loss = 0.0
    for xb, yb, wb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = pareto_weighted_mse(pred, yb, wb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    train_loss = epoch_loss / len(train_ds)

    # --- Validate ---
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb, wb in val_loader:
            pred = model(xb)
            val_loss_sum += pareto_weighted_mse(pred, yb, wb).item() * len(xb)
    val_loss = val_loss_sum / len(val_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # --- Early stopping ---
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

# Load best model
model.load_state_dict(best_state)

# ============================================================
# EVALUATION ON TEST SET
# ============================================================
print("\n=== Evaluating Model 1 on Test Set ===")

model.eval()
with torch.no_grad():
    pred_scaled = model(X_test).cpu().numpy()

# Inverse transform to original scale
pred_orig   = sc_y.inverse_transform(pred_scaled)
target_orig = sc_y.inverse_transform(y_test.cpu().numpy())

target_names = meta['M1_TARGETS']
# ['log_BER', 'throughput_bpsHz', 'latency_ms', 'energy_per_bit']

metrics = {}
print(f"\n{'Metric':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
print("-" * 55)

for i, name in enumerate(target_names):
    mae  = mean_absolute_error(target_orig[:, i], pred_orig[:, i])
    rmse = np.sqrt(np.mean((target_orig[:, i] - pred_orig[:, i])**2))
    r2   = r2_score(target_orig[:, i], pred_orig[:, i])
    metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"  {name:<23} {mae:>10.4f} {rmse:>10.4f} {r2:>8.4f}")

# ============================================================
# PLOT 1 — TRAINING CURVES
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
epochs_ran = range(1, len(train_losses) + 1)
ax.plot(epochs_ran, train_losses, 'b-', linewidth=2, label='Train Loss')
ax.plot(epochs_ran, val_losses,   'r-', linewidth=2, label='Val Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Pareto-Weighted MSE Loss')
ax.set_title('Model 1 — Training Curves')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/step2_model1_training_curves.png', dpi=150)
plt.show()
print("\nSaved: step2_model1_training_curves.png")

# ============================================================
# PLOT 2 — PREDICTED VS ACTUAL (2x2)
# ============================================================
display_names = ['log₁₀(BER)', 'Throughput (bps/Hz)',
                 'Latency (ms)', 'Energy/bit (J)']
colors        = ['steelblue', 'seagreen', 'tomato', 'mediumpurple']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (name, dname, color) in enumerate(
        zip(target_names, display_names, colors)):

    ax   = axes[i]
    true = target_orig[:, i]
    pred = pred_orig[:, i]
    r2   = metrics[name]['R2']
    mae  = metrics[name]['MAE']

    # Sample points for clarity
    idx  = np.random.choice(len(true), min(3000, len(true)), replace=False)
    ax.scatter(true[idx], pred[idx], alpha=0.3, s=5, color=color)

    # y=x line
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlabel(f'Actual {dname}')
    ax.set_ylabel(f'Predicted {dname}')
    ax.set_title(f'{dname}\nR²={r2:.4f}  MAE={mae:.4f}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('Model 1 — Predicted vs Actual (Test Set)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/step2_model1_pred_vs_actual.png', dpi=150)
plt.show()
print("Saved: step2_model1_pred_vs_actual.png")

# ============================================================
# PLOT 3 — ERROR BY PREFERENCE (box plots)
# ============================================================
df_test = pd.read_csv(f'{OUT}/df_test.csv')
pref_labels_ordered = [
    'MaxReliability', 'MaxThroughput', 'UltraLowLatency',
    'EnergyEfficient', 'Balanced', 'ReliabilitySpeed',
    'GreenThroughput', 'MissionCritical'
]
short_labels = ['MxR','MxT','ULL','Enr','Bal','R+S','GTP','MC']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, (name, dname, color) in enumerate(
        zip(target_names, display_names, colors)):

    ax     = axes[i]
    errors = np.abs(target_orig[:, i] - pred_orig[:, i])
    prefs  = df_test['preference_label'].values

    box_data = []
    for pl in pref_labels_ordered:
        mask = prefs == pl
        box_data.append(errors[mask])

    bp = ax.boxplot(box_data, labels=short_labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('User Preference')
    ax.set_ylabel(f'|Error| in {dname}')
    ax.set_title(f'{dname} — Error by Preference')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Model 1 — Prediction Error by User Preference', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/step2_model1_error_by_preference.png', dpi=150)
plt.show()
print("Saved: step2_model1_error_by_preference.png")

# ============================================================
# SAVE MODEL AND METRICS
# ============================================================
torch.save(model.state_dict(),
           f'{OUT}/model1_performance_predictor.pth')
pickle.dump(metrics, open(f'{OUT}/model1_metrics.pkl', 'wb'))

# Save architecture info for loading later
model_config = {'n_in': N_IN, 'n_out': N_OUT}
pickle.dump(model_config, open(f'{OUT}/model1_config.pkl', 'wb'))

print(f"\nSaved: model1_performance_predictor.pth")
print(f"Saved: model1_metrics.pkl")

print("\n" + "=" * 60)
print("STEP 2 COMPLETE — Model 1 trained and evaluated")
print("Run: python phase3_step3_model2.py")
print("=" * 60)
