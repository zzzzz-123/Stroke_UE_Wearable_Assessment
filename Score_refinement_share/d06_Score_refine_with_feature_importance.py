import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------------
# Feature configuration for each item
# -------------------------------
items = {
 'Item1': {'feature_index': [4, 13, 17, 23, 36, 44, 51],
           'feature_re': [1, 0, 0, 0, 0, 0, 0],
           'weights': [0.29210469, 0.20143329, 0.09092329, 0.09632829, 0.12540061, 0.09440064, 0.09940919],
           'feature_name': ['AMP(ay1)', 'DTW(yaw1)', 'DTW(az1)', 'APEN(az1)', 'SMO(ay2)', 'DTW(az2)', 'APEN(az2)']},
 'Item2': {'feature_index': [1, 3, 6, 7, 8, 12, 13, 15, 17, 26],
           'feature_re': [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
           'weights': [0.12305616, 0.13740978, 0.07045743, 0.0530685, 0.0465663, 0.0678832, 0.27982914, 0.04492149, 0.14601466, 0.03079333],
           'feature_name': ['AMP(yaw1)', 'SMO(pitch1)', 'DTW(pitch1)', 'DTW(yaw1)', 'DTW(roll1)', 'AMP(pitch2-pitch1)', 'AMP(pitch2)', 'AMP(roll2)', 'SMO(pitch2)', 'APEN(yaw2)']},
 'Item3': {'feature_index': [0, 3, 4, 11, 13, 17, 19, 22, 24],
           'feature_re': [1, 0, 0, 0, 0, 0, 0, 0, 0],
           'weights': [0.35000515, 0.27433434, 0.15129635, 0.02618414, 0.05379379, 0.05407706, 0.03012926, 0.03428762, 0.02589228],
           'feature_name': ['AMP(pitch1)', 'SMO(pitch1)', 'SMO(yaw1)', 'APEN(roll1)', 'AMP(pitch2)', 'SMO(pitch2)', 'SMO(roll2)', 'DTW(yaw2)', 'APEN(pitch2-pitch1)']},
 'Item4': {'feature_index': [0, 2, 6, 9, 11, 12, 14, 15, 17, 23, 24],
           'feature_re': [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
           'weights': [0.14012588, 0.27818149, 0.11952935, 0.05157937, 0.10793358, 0.05422403, 0.043729, 0.05905265, 0.04180299, 0.05116351, 0.05267814],
           'feature_name': ['AMP(pitch1)', 'AMP(roll1)', 'DTW(pitch1)', 'APEN(pitch1)', 'APEN(roll1)', 'AMP(pitch2-pitch1)', 'AMP(yaw2)', 'AMP(roll2)', 'SMO(pitch2)', 'DTW(roll2)', 'APEN(pitch2-pitch1)']},
 'Item5': {'feature_index': [3, 5, 7, 8, 9, 11, 15, 21, 22, 25, 26],
           'feature_re': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           'weights': [0.11407673, 0.17614814, 0.0660293, 0.21481615, 0.04212979, 0.04916458, 0.11837532, 0.05183075, 0.05945399, 0.04363046, 0.06434478],
           'feature_name': ['SMO(pitch1)', 'SMO(roll1)', 'DTW(yaw1)', 'DTW(roll1)', 'APEN(pitch1)', 'APEN(roll1)', 'AMP(roll2)', 'DTW(pitch2)', 'DTW(yaw2)', 'APEN(pitch2)', 'APEN(yaw2)']},
 'Item6': {'feature_index': [0, 1, 2, 4, 5, 6, 7],
           'feature_re': [1, 1, 0, 0, 0, 0, 0],
           'weights': [0.23975099, 0.23927745, 0.22635899, 0.10336305, 0.09416538, 0.04667804, 0.05040609],
           'feature_name': ['AMP(pitch3)', 'AMP(pitch3 - pitch1)', 'SMO(pitch3)', 'DTW(pitch3)', 'DTW(pitch3 - pitch1)', 'APEN(pitch3)', 'APEN(pitch3 - pitch1)']},
 'Item7': {'feature_index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           'feature_re': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           'weights': [0.02853802, 0.15008419, 0.27986116, 0.04599448, 0.10774502, 0.22603306, 0.024938, 0.04206131, 0.06304784, 0.03169693],
           'feature_name': ['AMP(f1)', 'AMP(f12)', 'AMP(fAll)', 'SMO(f1)', 'SMO(f12)', 'SMO(fAll)', 'DTW(f1)', 'DTW(f12)', 'DTW(fAll)', 'APEN(f1)']},
 'Item8': {'feature_index': [1, 2, 3, 4, 5, 6, 7, 8, 11],
           'feature_re': [1, 1, 0, 0, 0, 0, 0, 0, 0],
           'weights': [0.17263515, 0.11278849, 0.13187759, 0.16984302, 0.05469764, 0.12208075, 0.10437456, 0.06734844, 0.06435435],
           'feature_name': ['AMP(f12)', 'AMP(fAll)', 'SMO(f1)', 'SMO(f12)', 'SMO(fAll)', 'DTW(f1)', 'DTW(f12)', 'DTW(fAll)', 'APEN(fAll)']}
 }

# -------------------------------
# Parameter settings
# -------------------------------
key = 8  # Corresponds to the target item
move = ["000", "002", "004", "008", "014", "016", "020", "023"]
mov = move[key - 1]

# Path configuration
feature_path = fr"D:\pycharm project\Score—refinement\feature\stretched_feature\{mov}_feature_stretched.npy"
label_path = r"D:\pycharm project\Score—refinement\data\label.npy"

# -------------------------------
# Data loading
# -------------------------------
feature = np.load(feature_path)
label_all = np.load(label_path)

col_index = int(mov)
label = label_all[:, col_index].astype(int)

item_name = f"Item{key}"
item = items[item_name]

feature_index = item["feature_index"]
feature_re = item["feature_re"]
weights = np.array(item["weights"])

# -------------------------------
# Feature extraction and direction adjustment
# -------------------------------
selected_feature = feature[:, feature_index].astype(float)

# Reverse negatively oriented features (feature_re == 0)
for i, re_flag in enumerate(feature_re):
    if re_flag == 0:
        selected_feature[:, i] = 2.0 - selected_feature[:, i]

# -------------------------------
# Linear weighted aggregation (refined score computation)
# -------------------------------
refined_score = np.dot(selected_feature, weights)
refined_score = (
    (refined_score - np.min(refined_score)) /
    (np.max(refined_score) - np.min(refined_score)) * 2.0
)  # Rescaled to the range [0, 2]

# -------------------------------
# Output refined scores
# -------------------------------
print(f"\n=== Refined scoring results for {item_name} ({mov}) ===")
for i in range(len(label)):
    print(
        f"Sample {i:03d} | "
        f"Original label: {label[i]} | "
        f"Refined score: {refined_score[i]:.3f}"
    )

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(8, 5))

scatter = plt.scatter(
    range(len(label)),
    refined_score,
    c=label,
    cmap="viridis_r",
    s=50,
    edgecolor="k",
    alpha=0.8
)

cbar = plt.colorbar(scatter)
cbar.set_label("Original label", fontsize=12)

plt.title(
    f"{item_name} ({mov}) - Refined score versus original label",
    fontsize=14
)
plt.xlabel("Sample index", fontsize=12)
plt.ylabel("Refined score", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# -------------------------------
# Save results
# -------------------------------
save_dir = r"D:\pycharm project\Score—refinement\feature\refined_result"
os.makedirs(save_dir, exist_ok=True)

plt.show()

save_path = os.path.join(save_dir, f"{item_name}_{mov}_refined_score.npy")
np.save(save_path, refined_score)

print(f"\nRefined scores have been saved to: {save_path}")
