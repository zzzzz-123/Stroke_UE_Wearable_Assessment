import numpy as np
from scipy.signal import convolve
from scipy.spatial.transform import Rotation as R

# functions for data preprocessing

# -------------------------------------------------
# Step 1. Moving average smoothing
# -------------------------------------------------

def moving_average(signal, window_size=5):
    """Apply moving average filter to 1D signal."""
    window = np.ones(window_size) / window_size
    return convolve(signal, window, mode='same')


def smooth_multichannel(data, window_size=5):
    """Apply moving average to each channel.
    data: (T, C)
    """
    return np.stack([
        moving_average(data[:, c], window_size)
        for c in range(data.shape[1])
    ], axis=1)


# -------------------------------------------------
# Step 2. Segment signals by therapist-marked points
# -------------------------------------------------

def segment_signal(data, segment_points):
    """
    data: (T, C)
    segment_points: list of (start, end) indices
    return: list of segments
    """
    return [data[s:e] for (s, e) in segment_points]


# -------------------------------------------------
# Step 3. Linear interpolation to fixed length
# -------------------------------------------------

def resample_segment(segment, target_len=150):
    """Resample a segment to target length using linear interpolation.
    segment: (T, C)
    """
    T, C = segment.shape
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, C))
    for c in range(C):
        resampled[:, c] = np.interp(x_new, x_old, segment[:, c])
    return resampled


# -------------------------------------------------
# Step 4. Quaternion to Euler angle conversion
# -------------------------------------------------

def quaternion_to_euler(quat):
    """
    quat: (T, 4) in [x, y, z, w]
    return: pitch, yaw, roll (T, 3)
    """
    rot = R.from_quat(quat)
    # xyz corresponds to pitch(x), roll(y), yaw(z)
    euler = rot.as_euler('xyz', degrees=False)
    pitch = euler[:, 0]
    roll = euler[:, 1]
    yaw = euler[:, 2]
    return pitch, yaw, roll


# -------------------------------------------------
# Step 5. Differential pitch features
# -------------------------------------------------

def differential_pitch(pitch_a, pitch_b):
    """Compute differential pitch angle."""
    return pitch_a - pitch_b


# -------------------------------------------------
# Step 6. Finger coordination features
# -------------------------------------------------

def finger_pairwise_sum(finger_signals):
    """
    finger_signals: (T, 5) -> F1 ... F5
    return: (T, 10) pairwise combinations
    """
    pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            pairs.append(finger_signals[:, i] + finger_signals[:, j])
    return np.stack(pairs, axis=1)


def finger_total_sum(finger_signals):
    """Total finger flexion-extension capability."""
    return np.sum(finger_signals, axis=1, keepdims=True)


# -------------------------------------------------
# Step 7. Feature construction for one movement cycle
# -------------------------------------------------

def build_feature_tensor(S1, S2, S3, quat1, quat2, quat3, fingers):
    """
    Inputs (already smoothed, segmented, resampled):
    S1, S2, S3: IMU raw signals (T, 9)
    quat1, quat2, quat3: quaternion (T, 4)
    fingers: finger flexion signals (T, 5)

    Output:
    X: (150, 55)
    """
    # Euler angles
    p1, y1, r1 = quaternion_to_euler(quat1)
    p2, y2, r2 = quaternion_to_euler(quat2)
    p3, y3, r3 = quaternion_to_euler(quat3)

    # Differential pitch
    diff_p12 = differential_pitch(p1, p2).reshape(-1, 1)
    diff_p31 = differential_pitch(p3, p1).reshape(-1, 1)

    # Finger features
    Fij = finger_pairwise_sum(fingers)       # (T, 10)
    Fall = finger_total_sum(fingers)         # (T, 1)

    # S1 features (13)
    feat_S1 = np.concatenate([
        S1,
        p1.reshape(-1, 1), y1.reshape(-1, 1), r1.reshape(-1, 1),
        diff_p12
    ], axis=1)

    # S2 features (13)
    feat_S2 = np.concatenate([
        S2,
        p2.reshape(-1, 1), y2.reshape(-1, 1), r2.reshape(-1, 1),
        diff_p12
    ], axis=1)

    # S3 features (29)
    feat_S3 = np.concatenate([
        S3,
        p3.reshape(-1, 1), y3.reshape(-1, 1), r3.reshape(-1, 1),
        diff_p31,
        fingers,
        Fij,
        Fall
    ], axis=1)

    # Final feature tensor X ∈ R^{150 × 55}
    X = np.concatenate([feat_S1, feat_S2, feat_S3], axis=1)
    return X
