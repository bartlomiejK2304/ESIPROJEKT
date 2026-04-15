# ===============================================
# Neural Network Tests

PARAM_GRID = {
    "layers": [
        [3, 5, 1],
        [3, 10, 1],
        [3, 8, 4, 1],
        [3, 16, 8, 1]
    ],
    "activation": [
        'relu',
        'tanh',
        'sigmoid',
        'linear'
    ],
    "learning_rate": [
        0.1,
        0.01,
        0.001,
        0.0001
    ],
    "multiplier": [
        0.1,
        0.01,
        0.001,
        0.0001
    ],
    "division_coefficient": [
        1,
        10,
        100,
        1000
    ],
    "epochs": [
        100,
        500,
        1000,
        2000
    ],
    "min_change": [
        1e-2,
        1e-4,
        1e-6,
        1e-8
    ],
    "target_loss": [
        1e-1,
        1e-2,
        1e-3,
        1e-4
    ]
}

# ===============================================