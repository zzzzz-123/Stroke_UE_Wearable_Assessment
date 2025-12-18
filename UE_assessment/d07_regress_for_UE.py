import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# # Total score prediction based on Refined scores

# ------------------ Data Preparation ------------------
move = ["000", "002", "004", "008", "014", "016", "020", "023"]

label_path = r"D:\pycharm project\Score—refinement\data\label.npy"
label = np.load(label_path, allow_pickle=True)

sc = label[:, move]

# Total score
total_score = np.load(r"D:\pycharm project\Score—refinement\data\UE_score.npy", allow_pickle=True)

# Refined scores as features (222 × 8)
re_score = []
i=1
for m in move:
    re_path = fr"D:\pycharm project\Score—refinement\feature\refined_result\item{i}_{m}_refined_score.npy"
    re_fine = np.load(re_path)
    re_score.append(re_fine)
    i+=1
X = np.array(re_score).T  # (222, 8)
y = total_score

print("✅ Data shape:", X.shape, y.shape)

# ------------------ Model Definition ------------------
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# ------------------ 5-Fold Cross Validation ------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_all = np.zeros_like(y, dtype=float)
r2_list, spearman_list, rmse_list = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    rf.fit(X[train_idx], y[train_idx])
    y_pred = rf.predict(X[test_idx])
    y_pred_all[test_idx] = y_pred

    # Metrics
    r2_fold = r2_score(y[test_idx], y_pred)
    rho_fold, _ = spearmanr(y[test_idx], y_pred)
    rmse_fold = np.sqrt(mean_squared_error(y[test_idx], y_pred))

    r2_list.append(r2_fold)
    spearman_list.append(rho_fold)
    rmse_list.append(rmse_fold)

    print(f"Fold {fold+1}: R²={r2_fold:.3f}, Spearman={rho_fold:.3f}, RMSE={rmse_fold:.3f}")

# ------------------ Mean Performance ------------------
r2_mean = np.mean(r2_list)
rho_mean = np.mean(spearman_list)
rmse_mean = np.mean(rmse_list)

print(f"\n📊 Mean 5-Fold R² = {r2_mean:.3f}")
print(f"📊 Mean 5-Fold Spearman = {rho_mean:.3f}")
print(f"📊 Mean 5-Fold RMSE = {rmse_mean:.3f}")



