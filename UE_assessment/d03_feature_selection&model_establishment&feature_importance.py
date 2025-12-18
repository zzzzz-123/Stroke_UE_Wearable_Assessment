import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.over_sampling import SMOTE

# Single-action scoring was performed using SFS-based feature selection
# (with the number of selected features fixed at 10), followed by model
# training and feature importance analysis.


# -----------------------------------
# data loading
move = ["000", "002", "004", "008", "014", "016", "020", "023"]
mov = move[0]
path = rf"D:\pycharm project\Score—refinement\feature\{mov}_feature_mat.npy"

feature_mat = np.load(path, allow_pickle=True)
label = np.load(r"D:\pycharm project\Score—refinement\data\label.npy", allow_pickle=True)
label1 = label[:, int(mov)]

X = feature_mat
y = label1.ravel().astype(int)

# -----------------------------------
# Step 1: data balancing
print("\n SMOTE")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)
print(" After SMOTE:", np.bincount(y_bal))

# -----------------------------------
# Step 2: Feature selection (SFS)
print("\n SFS")
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

sfs = SequentialFeatureSelector(
    rf_base,
    n_features_to_select=10,
    direction='forward',
    n_jobs=-1
)
sfs.fit(X_bal, y_bal)

selected_features = np.where(sfs.get_support())[0]
X_sfs = X_bal[:, selected_features]
print(f"Key features: {selected_features}")

# -----------------------------------
# Step 3: Model development
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}

# -----------------------------------
# Step 4: 5-fold validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'='*60}\n🤖 Model: {name} — 5-fold\n{'='*60}")
    fold = 1
    all_reports = []
    feature_importances = []

    for train_idx, test_idx in kf.split(X_sfs, y_bal):
        print(f"\n📂  {fold} fold training...")

        X_train, X_test = X_sfs[train_idx], X_sfs[test_idx]
        y_train, y_test = y_bal[train_idx], y_bal[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 各项指标
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        print(f"📊  {fold} fold performance:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")

        all_reports.append((acc, precision, recall, f1))
        feature_importances.append(model.feature_importances_)
        fold += 1

    # -----------------------------------
    # Step 5: averaged performance
    avg_acc = np.mean([r[0] for r in all_reports])
    avg_precision = np.mean([r[1] for r in all_reports])
    avg_recall = np.mean([r[2] for r in all_reports])
    avg_f1 = np.mean([r[3] for r in all_reports])

    print("\n✅ averaged performance:")
    print(f"  Accuracy:  {avg_acc:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1-score:  {avg_f1:.4f}")

    # -----------------------------------
    # Step 6: Feature importance output
    mean_importance = np.mean(feature_importances, axis=0)
    sorted_idx = np.argsort(mean_importance)[::-1]

    print("\nFeature importance ranked in descending order:")
    for i, idx in enumerate(sorted_idx):
        print(f"  Feature {selected_features[idx]}: {mean_importance[idx]:.4f}")

    # -----------------------------------
    # Step 7: Hold-out validation using full dataset (70% training / 30% testing)
    print(f"\n{'-' * 60}\nModel: {name} — Hold-out validation (70/30 split)\n{'-' * 60}")
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_sfs, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    model.fit(X_train_all, y_train_all)
    y_pred_all = model.predict(X_test_all)

    acc_all = accuracy_score(y_test_all, y_pred_all)
    report_all = classification_report(y_test_all, y_pred_all, output_dict=True, zero_division=0)
    precision_all = report_all['weighted avg']['precision']
    recall_all = report_all['weighted avg']['recall']
    f1_all = report_all['weighted avg']['f1-score']

    print("Performance on the held-out test set (30% of the full dataset):")
    print(f"  Accuracy:  {acc_all:.4f}")
    print(f"  Precision: {precision_all:.4f}")
    print(f"  Recall:    {recall_all:.4f}")
    print(f"  F1-score:  {f1_all:.4f}")

