import numpy as np
import os
from fastdtw import fastdtw
# Feature extraction

def unwrap_signal(signal_matrix):
    """
    Unwrap folded angular signals

    Input:
        signal_matrix: (length × channels)
    Output:
        unwrapped_matrix
    """
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    n_samples, n_channels = signal_matrix.shape
    unwrapped_matrix = np.zeros_like(signal_matrix)

    for ch in range(n_channels):
        rad = np.deg2rad(signal_matrix[:, ch])
        unwrapped_rad = np.unwrap(rad)
        unwrapped_matrix[:, ch] = np.rad2deg(unwrapped_rad)

    return unwrapped_matrix


def compute_amplitude(signal_matrix):
    """
    Compute peak-to-peak amplitude for each channel

    Input:
        signal_matrix: (length × channels)
    Output:
        amplitudes: (channels,)
    """
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    n_samples, n_channels = signal_matrix.shape
    amplitudes = np.zeros(n_channels)

    for ch in range(n_channels):
        amplitudes[ch] = np.max(signal_matrix[:, ch]) - np.min(signal_matrix[:, ch])

    return amplitudes


def compute_rms_derivative(signal_matrix):
    """
    Compute RMS of first-order derivative for each channel
    """
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    n_samples, n_channels = signal_matrix.shape
    rms_features = np.zeros(n_channels)

    for ch in range(n_channels):
        first_diff = np.diff(signal_matrix[:, ch], n=1)
        rms_features[ch] = np.sqrt(np.mean(first_diff ** 2))

    return rms_features


def compute_dtw(signal1, signal2):
    """
    Compute DTW distance for each channel between two multichannel signals

    Parameters:
        signal1, signal2: np.ndarray, shape (length × channels)

    Returns:
        dtw_distances: np.ndarray, shape (channels,)
    """
    signal1 = np.asarray(signal1, dtype=float)
    signal2 = np.asarray(signal2, dtype=float)

    if signal1.ndim == 1:
        signal1 = signal1[:, np.newaxis]
    if signal2.ndim == 1:
        signal2 = signal2[:, np.newaxis]

    if signal1.shape[1] != signal2.shape[1]:
        raise ValueError(
            f"Inconsistent channel numbers: signal1={signal1.shape[1]}, signal2={signal2.shape[1]}"
        )

    n_channels = signal1.shape[1]
    dtw_distances = np.zeros(n_channels)

    for ch in range(n_channels):
        x = np.asarray(signal1[:, ch], dtype=float).ravel()
        symmetric_signal = [-i for i in x]
        y = np.asarray(signal2[:, ch], dtype=float).ravel()

        dtw_distance, _ = fastdtw(x, y, dist=lambda a, b: abs(a - b))
        sy_dtw_distance, _ = fastdtw(symmetric_signal, y, dist=lambda a, b: abs(a - b))

        dtw_distances[ch] = min(dtw_distance, sy_dtw_distance)

    return dtw_distances


def compute_apen(signal_matrix, m=2, r=0.2):
    """
    Compute Approximate Entropy (ApEn) for each channel

    Parameters:
        signal_matrix: np.ndarray, shape (length × channels)
        m: embedding dimension (default = 2)
        r: tolerance threshold (default = 0.2 × std)

    Returns:
        apen_values: np.ndarray, shape (channels,)
    """
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    n_samples, n_channels = signal_matrix.shape
    apen_values = np.zeros(n_channels)

    def _phi(x, m, r):
        N = len(x)
        X = np.array([x[i:i + m] for i in range(N - m + 1)])
        dist = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        C = np.sum(dist <= r, axis=0) / (N - m + 1)
        return np.mean(np.log(C + 1e-10))  # avoid log(0)

    for ch in range(n_channels):
        x = signal_matrix[:, ch]
        std_x = np.std(x)
        r_eff = r * std_x
        apen_values[ch] = abs(_phi(x, m, r_eff) - _phi(x, m + 1, r_eff))

    return apen_values


# ===============================
# Feature extraction pipeline
# ===============================

move = ["000", "002", "004", "008", "014", "016", "020", "023"]
mov = move[0]  # iterate over movements
path = r"D:\pycharm project\Score—refinement\data"

std_path1 = rf"D:\pycharm project\Score—refinement\data\std\{mov}_1.npy"
std1 = np.load(std_path1, allow_pickle=True)

std_path2 = rf"D:\pycharm project\Score—refinement\data\std\{mov}_2.npy"
std2 = np.load(std_path2, allow_pickle=True)

std_path3 = rf"D:\pycharm project\Score—refinement\data\std\{mov}_3.npy"
std3 = np.load(std_path3, allow_pickle=True)

path1 = os.path.join(path, mov)

for file_name in os.listdir(path1):

    # -------- Sensor 1 --------
    if file_name.endswith("_1.npy"):
        print(file_name)

        file_path = os.path.join(path1, file_name)
        feature = []

        dt1 = np.load(file_path, allow_pickle=True)
        un_dt1 = unwrap_signal(dt1)

        feature.append(compute_amplitude(un_dt1))
        feature.append(compute_rms_derivative(un_dt1))
        feature.append(compute_dtw(un_dt1, std1))
        feature.append(compute_apen(un_dt1))

        print(np.mat(feature).shape)

        save_path = os.path.join(path1, file_name.split(".")[0] + "_features.npy")
        np.save(save_path, feature, allow_pickle=True)

    # -------- Sensor 2 --------
    if file_name.endswith("_2.npy"):
        print(file_name)

        file_path = os.path.join(path1, file_name)
        feature = []

        dt2 = np.load(file_path, allow_pickle=True)
        un_dt2 = unwrap_signal(dt2)

        feature.append(compute_amplitude(un_dt2))
        feature.append(compute_rms_derivative(un_dt2))
        feature.append(compute_dtw(un_dt2, std2))
        feature.append(compute_apen(un_dt2))

        print(np.mat(feature).shape)

        save_path = os.path.join(path1, file_name.split(".")[0] + "_features.npy")
        np.save(save_path, feature, allow_pickle=True)

    # -------- Sensor 3 --------
    if file_name.endswith("_3.npy"):
        print(file_name)

        file_path = os.path.join(path1, file_name)
        feature = []

        dt3 = np.load(file_path, allow_pickle=True)
        un_dt3 = unwrap_signal(dt3)

        feature.append(compute_amplitude(un_dt3))
        feature.append(compute_rms_derivative(un_dt3))
        feature.append(compute_dtw(un_dt3, std3))
        feature.append(compute_apen(un_dt3))

        print(np.mat(feature).shape)

        save_path = os.path.join(path1, file_name.split(".")[0] + "_features.npy")
        np.save(save_path, feature, allow_pickle=True)
