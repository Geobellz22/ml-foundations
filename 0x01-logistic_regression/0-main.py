#!/usr/bin/env python3
import numpy as np
sigmoid = __import__('0-sigmoid').sigmoid

with np.errstate(over='raise', divide='raise', invalid='raise'):
    print(sigmoid(0))                    # 0.5
    print(sigmoid(1000), sigmoid(-1000)) # 1.0 0.0

    z = np.array([-1000., -2., 0., 2., 1000.])
    print(sigmoid(z).round(4))           # [0. 0.1192 0.5 0.8808 1.]

    zs = np.linspace(-1000, 1000, 9)
    print(np.allclose(sigmoid(-zs), 1 - sigmoid(zs)))   # symmetry

    A = np.arange(-3., 3.).reshape(2, 3)
    print(sigmoid(A).shape)              # (2, 3)

    ints = np.array([-2, 0, 2])
    print(sigmoid(ints).round(4), ints)  # input unchanged
    print(sigmoid([0, 1, -1]).round(4))  # a plain list works
    print(type(sigmoid(0)))              # <class 'float'>

    try:
        from scipy.special import expit
        print(np.allclose(sigmoid(zs), expit(zs)))
    except ImportError:
        print("scipy not installed, skipped")
