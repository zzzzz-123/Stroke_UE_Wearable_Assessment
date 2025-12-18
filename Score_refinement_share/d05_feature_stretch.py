import numpy as np
import os
# Feature normalization was applied to facilitate subsequent score refinement.

# -------------------------------
# Parameter configuration
# -------------------------------
key = 2  # Corresponds to Item 2
move = ["000", "002", "004", "008", "014", "016", "020", "023"]
mov = move[key - 1]

# Path configuration
feature_path = fr"D:\pycharm project\Score—refinement\feature\{mov}_feature_mat.npy"

# -------------------------------
# Data loading
# -------------------------------
feature = np.load(feature_path)  # shape = (222, n_features)

col_index = int(mov)

# -------------------------------
# Step 1: Column-wise normalization to the range [0, 2]
# -------------------------------
def normalize_to_0_2(matrix):
    matrix = np.asarray(matrix, dtype=float)
    normed = np.zeros_like(matrix)
    for i in range(matrix.shape[1]):
        col = matrix[:, i]
        col_min, col_max = np.min(col), np.max(col)
        if col_max - col_min == 0:
            normed[:, i] = 1.0  # Assign the midpoint value if the column is constant
        else:
            normed[:, i] = 2 * (col - col_min) / (col_max - col_min)
    return normed

feature_norm = normalize_to_0_2(feature)

# -------------------------------
# Save results
# -------------------------------
save_dir = fr"D:\pycharm project\Score—refinement\feature\stretched_feature"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{mov}_feature_stretched.npy")
np.save(save_path, feature_norm)

print(
    f"All features have been normalized successfully. "
    f"Results saved to: {save_path}"
)
print(f"Output matrix shape: {feature_norm.shape}")
