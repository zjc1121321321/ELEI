import numpy as np

def compute_scaling_factors(X, y, lower_thresh=1e-6, upper_thresh=1e3):
    x_scales = []
    X_scaled = X.copy()
    for i in range(X.shape[1]):
        col = X[:, i]
        max_val = np.max(np.abs(col))
        if max_val > 0 and (max_val < lower_thresh or max_val > upper_thresh):
            exponent = int(np.floor(np.log10(max_val)))
            scale = 10 ** exponent
        else:
            scale = 1.0
        x_scales.append(scale)
        X_scaled[:, i] = col / scale

    max_y = np.max(np.abs(y))
    if max_y > 0 and (max_y < lower_thresh or max_y > upper_thresh):
        exponent = int(np.floor(np.log10(max_y)))
        y_scale = 10 ** exponent
    else:
        y_scale = 1.0
    y_scaled = y / y_scale

    return X_scaled, y_scaled, x_scales, y_scale
