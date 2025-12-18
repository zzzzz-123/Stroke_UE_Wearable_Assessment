import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.over_sampling import SMOTE

# This script evaluates the optimal number of features selected by SFS
# based on classification performance.


# ===============================
# Step 0: Data loading
# ===============================
move = ["000", "002", "004", "008", "014", "016", "020", "023"]
mov = move[6]
path = fr"D:\pycharm project\Score—refinement\feature\{mov}_feature_mat.npy"

feature_mat = np.load(path, allow_pickle=True)
label = np.load(r"D:\pycharm project\Score—refinement\data\label.npy", allow_pickle=True)  # 222 × 27
label1 = label[:, int(mov)]

X = feature_mat
y = label1.ravel().astype(int)

print(f"Data successfully loaded: X.shape={X.shape}, y.shape={y.shape}")
print("Original class distribution:", np.bincount(y))


# ===============================
# Step 1: Class balancing using SMOTE
# ===============================
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", np.bincount(y_bal))


# ===============================
# Step 2: Model definition and cross-validation
# ===============================
rf_base = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Container for all experimental results
all_results = []

print("\n==================== Evaluating SFS feature subset sizes ====================")

max_feat = X_bal.shape[1]  # Total number of original features
for n_feat in range(4, 11):

    if n_feat > max_feat:
        print(
            f"\nWarning: n_features_to_select={n_feat} exceeds "
            f"the total number of available features ({max_feat}). "
            "Loop terminated."
        )
        break

    print(f"\nEvaluating n_features_to_select = {n_feat}")

    # Step 2.1: Feature selection
    if n_feat == max_feat:
        print(
            f"Note: n_features_to_select ({n_feat}) equals the total number "
            f"of features ({max_feat}). SFS is skipped and all features are used."
        )
        selected_features = np.arange(max_feat)
    else:
        sfs = SequentialFeatureSelector(
            rf_base,
            n_features_to_select=n_feat,
            direction="forward",
            n_jobs=-1
        )
        sfs.fit(X_bal, y_bal)
        selected_features = np.where(sfs.get_support())[0]

    X_sfs = X_bal[:, selected_features]

    fold_metrics = []
    feature_importances_accum = np.zeros(len(selected_features))

    # Step 2.2: Five-fold stratified cross-validation
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X_sfs, y_bal), start=1
    ):
        X_train, X_test = X_sfs[train_idx], X_sfs[test_idx]
        y_train, y_test = y_bal[train_idx], y_bal[test_idx]

        rf_base.fit(X_train, y_train)
        y_pred = rf_base.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        rec = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        feature_importances_accum += rf_base.feature_importances_

        print(
            f"  Fold {fold}: "
            f"Accuracy={acc:.4f}, "
            f"Precision={prec:.4f}, "
            f"Recall={rec:.4f}, "
            f"F1-score={f1:.4f}"
        )

        fold_metrics.append({
            "n_features": n_feat,
            "Fold": fold,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Selected_Features": selected_features
        })

    # Mean feature importance across folds
    mean_importances = feature_importances_accum / skf.get_n_splits()

    # Mean performance across folds
    mean_acc = np.mean([f["Accuracy"] for f in fold_metrics])
    mean_prec = np.mean([f["Precision"] for f in fold_metrics])
    mean_rec = np.mean([f["Recall"] for f in fold_metrics])
    mean_f1 = np.mean([f["F1-Score"] for f in fold_metrics])

    print(
        f"Average performance (n_features={n_feat}): "
        f"Accuracy={mean_acc:.4f}, "
        f"Precision={mean_prec:.4f}, "
        f"Recall={mean_rec:.4f}, "
        f"F1-score={mean_f1:.4f}"
    )

    # Store results
    for f in fold_metrics:
        f.update({
            "Mean_Accuracy": mean_acc,
            "Mean_Precision": mean_prec,
            "Mean_Recall": mean_rec,
            "Mean_F1-Score": mean_f1,
            "Feature_Importances": mean_importances
        })
        all_results.append(f)


# ===============================
# Step 3: Result summarization and analysis
# ===============================
df_results = pd.DataFrame(all_results)

# Identify the optimal number of features based on the highest mean F1-score
best_n = (
    df_results
    .groupby("n_features")["Mean_F1-Score"]
    .mean()
    .idxmax()
)

best_row = df_results[df_results["n_features"] == best_n].iloc[0]
best_features = best_row["Selected_Features"]
best_importances = best_row["Feature_Importances"]

print("\n==================== Summary of results ====================")

summary = (
    df_results
    .groupby("n_features")[[
        "Mean_Accuracy",
        "Mean_Precision",
        "Mean_Recall",
        "Mean_F1-Score"
    ]]
    .mean()
    .reset_index()
)
print(summary)

print(f"\nOptimal number of selected features: {best_n}")
print("Corresponding feature indices:")
print(best_features)

print("Corresponding feature importances:")
for idx, imp in zip(best_features, best_importances):
    print(f"  Feature {idx}: {imp:.4f}")


# ===============================
# Step 4: Export results to Excel
# ===============================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(
    desktop,
    f"SFS_Feature_Selection_Results_rf_{mov}.xlsx"
)

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_results.to_excel(
        writer, sheet_name="All_Folds_Details", index=False
    )
    summary.to_excel(
        writer, sheet_name="Summary", index=False
    )

print(f"\nAll results have been saved to: {output_path}")
