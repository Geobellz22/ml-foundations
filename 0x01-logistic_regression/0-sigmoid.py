#!/usr/bin/env python3
import numpy as np
def sigmoid(z):
    z_arr = np.asarray(z, dtype=float)
    t = np.exp(-np.abs(z_arr))
    out = np.where(z_arr >= 0, 1.0 / (1.0 + t), t / (1.0 + t))
    if np.isscalar(z):
        return float(out)
    return out
