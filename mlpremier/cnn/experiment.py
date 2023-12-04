"""
Run end-to-end experiments on the FPL Regression CNN.

Wrapper for preprocessing, model construction, model training, model evaluation,
and results visualization.
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..','..'))
import tensorflow as tf
import numpy as np
import pandas as pd
from mlpremier.cnn.model import build_train_cnn
import tensorflow as tf
import itertools

def gridsearch_cnn(epochs: int = 200,
                   batch_size: int = 32,
                   learning_rate: float = 0.01,
                   seed: int = 229,
                   verbose: bool = False):
    """
    GridSearch for Best Hyperparameters
    """
    STANDARD_NUM_FEATURES = ['minutes', 'goals_scored', 'assists', 'goals_conceded',
                          'clean_sheets', 'bps', 'yellow_cards', 'red_cards', 
                          'own_goals', 'saves', 'penalties_missed', 'penalties_saved',
                          'ict_index', 'total_points']
    STANDARD_CAT_FEATURES = []

    if verbose:
        print("======= Running GridSearch Experiment ========")

    tf.random.set_seed(seed)
    np.random.seed(seed)

    DATA_DIR = os.path.join(os.getcwd(), '..', 'data', 'clean_data')
    SEASONS = ['2020-21', '2021-22', ]
    POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    WINDOW_SIZES = [3, 6, 9]
    KERNEL_SIZES = [1,2,3,4]
    NUM_FILTERS = [16, 32, 64, 128, 256]
    NUM_DENSE = [8, 16, 32, 64, 128, 256]
    DROP_LOW_PLAYTIME = True
    LOW_PLAYTIME_CUTOFF = [0, 10, 20, 40, 60]
    NUM_FEATURES = {'ptsonly': ['total_points'],
                    'small': ['total_points', 'minutes', 'goals_scored', 'assists'],
                    'medium': ['total_points', 'minutes', 'goals_scored', 'assists', 'goals_conceded', 'yellow_cards'],
                    'large': STANDARD_NUM_FEATURES}
    REGULARIZATIONS = [0, 0.01, 0.05]

    # Loop through all combinations of parameters
    for (season, position, window_size, kernel_size, num_filters, num_dense,
        low_playtime_cutoff, num_feature_key, regularization) in itertools.product(
        SEASONS, POSITIONS, WINDOW_SIZES, KERNEL_SIZES, NUM_FILTERS, NUM_DENSE,
        LOW_PLAYTIME_CUTOFF, NUM_FEATURES.keys(), REGULARIZATIONS):

        if kernel_size >= window_size:  #skip invalid configurations of kernel size
            continue

        hyperparameters = {
            'season': season,
            'position': position,
            'window_size': window_size,
            'kernel_size': kernel_size,
            'num_filters': num_filters,
            'num_dense': num_dense,
            'low_playtime_cutoff': low_playtime_cutoff,
            'num_feature_key': num_feature_key,
            'regularization': regularization
        }

        if verbose:
            print(f"===== Running Expt for Parameters: =====\n {hyperparameters}\n")

        # Run the experiment for the current set of hyperparameters
        model, history = build_train_cnn(DATA_DIR,
                                        season=season,
                                        position=position,
                                        window_size=window_size,
                                        kernel_size=kernel_size,
                                        num_filters=num_filters,
                                        num_dense=num_dense,
                                        low_playtime_cutoff=low_playtime_cutoff,
                                        num_features=NUM_FEATURES[num_feature_key],
                                        regularization=regularization,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        drop_low_playtime=True,
                                        cat_features=STANDARD_CAT_FEATURES,
                                        conv_activation='relu',
                                        dense_activation='relu',
                                        learning_rate=learning_rate,
                                        loss='mse',
                                        metrics=['mae'],
                                        verbose=verbose,
                                        tolerance=1e-5, #not used rn
                                        plot=False,
                                        log=True, #log_file = if i want to set new file
                                        standardize=True)

    if verbose:
        print("======= Done with GridSearch Experiment ========")

    return
