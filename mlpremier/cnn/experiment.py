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
from mlpremier.cnn.evaluate import eval_cnn, log_evals
import tensorflow as tf
import itertools

def gridsearch_cnn(epochs: int = 200,
                   batch_size: int = 32,
                   learning_rate: float = 0.01,
                   seed: int = 229,
                   verbose: bool = False,
                   log_file: str = os.path.join('results', 'gridsearch.csv')):
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
    SEASONS = ['2020-21']
    POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    WINDOW_SIZES = [3, 6, 9]
    KERNEL_SIZES = [1, 2, 3]
    NUM_FILTERS = [64]
    NUM_DENSE = [64]
    DROP_LOW_PLAYTIME = True
    LOW_PLAYTIME_CUTOFF = [0, 30]
    NUM_FEATURES = {'ptsonly': ['total_points'],
                    'small': ['total_points', 'minutes', 'goals_scored', 'assists'],
                    'medium': ['total_points', 'minutes', 'goals_scored', 'assists', 'goals_conceded', 'yellow_cards'],
                    'large': STANDARD_NUM_FEATURES}
    OPTIMIZERS = ['adam', 'sgd']
    REGULARIZATIONS = [0]  #

    # Loop through all combinations of parameters
    expt_results = []

    for (season, position, window_size, kernel_size, num_filters, num_dense,
        low_playtime_cutoff, num_feature_key, optimizer, regularization) in itertools.product(
        SEASONS, POSITIONS, WINDOW_SIZES, KERNEL_SIZES, NUM_FILTERS, NUM_DENSE,
        LOW_PLAYTIME_CUTOFF, NUM_FEATURES.keys(), OPTIMIZERS, REGULARIZATIONS):

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
        model, expt_res = build_train_cnn(DATA_DIR,
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
                                        optimizer=optimizer,
                                        learning_rate=learning_rate,
                                        loss='mse',
                                        metrics=['mae'],
                                        verbose=verbose,
                                        tolerance=1e-5, #not used rn
                                        plot=False,
                                        standardize=True)
        
        expt_results.append(expt_res)

    if verbose:
        print(f"Updating GridSearch Results Log File: {log_file}...")
    log_evals(log_file, expt_results, verbose=verbose)

    if verbose:
        print("======= Done with GridSearch Experiment ========")

    return
