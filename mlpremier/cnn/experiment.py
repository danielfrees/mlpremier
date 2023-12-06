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
from tqdm import tqdm

from config import STANDARD_CAT_FEATURES, STANDARD_NUM_FEATURES

def gridsearch_cnn(epochs: int = 200,
                   batch_size: int = 32,
                   learning_rate: float = 0.01,
                   seed: int = 229,
                   verbose: bool = False,
                   log_file: str = os.path.join('results', 'gridsearch.csv')):
    """
    GridSearch for Best Hyperparameters
    """

    if verbose:
        print("======= Running GridSearch Experiment ========")

    tf.random.set_seed(seed)
    np.random.seed(seed)

    DATA_DIR = os.path.join(os.getcwd(), '..', 'data', 'clean_data')
    SEASONS = [['2020-21', '2021-22']]
    POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    WINDOW_SIZES = [3, 6, 9]
    KERNEL_SIZES = [1, 2, 3, 4]
    NUM_FILTERS = [64, 128, 256]
    NUM_DENSE = [64, 128, 256]
    CONV_ACTIVATION = 'relu'
    DENSE_ACTIVATION = 'relu'
    DROP_LOW_PLAYTIME = [False, True]
    LOW_PLAYTIME_CUTOFF = [15]
    AMT_NUM_FEATURES = ['ptsonly','small', 'medium', 'large'] #,'medium','large']
    NUM_FEATURES_DICT = {
        'GK': {
            'ptsonly': ['total_points'],
            'small': ['total_points', 'minutes', 'saves'],
            'medium': ['total_points', 'minutes', 'saves', 'bps', 'goals_conceded'],
            'large': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'yellow_cards', 'red_cards']
        },
        'DEF': {
            'ptsonly': ['total_points'],
            'small': ['total_points', 'minutes', 'goals_scored', 'assists'],
            'medium': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps'],
            'large': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'yellow_cards', 'red_cards', 'own_goals', 'penalties_saved']
        },
        'MID': {
            'ptsonly': ['total_points'],
            'small': ['total_points', 'minutes', 'goals_scored', 'assists'],
            'medium': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps'],
            'large': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'yellow_cards', 'red_cards']
        },
        'FWD': {
            'ptsonly': ['total_points'],
            'small': ['total_points', 'minutes', 'goals_scored', 'assists'],
            'medium': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps'],
            'large': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'yellow_cards', 'red_cards']
        }
    }
    CAT_FEATURES = STANDARD_CAT_FEATURES
    OPTIMIZERS = ['adam'] #, 'sgd'
    REGULARIZATIONS = [0.001]  #L1 L2 reg strength
    TOLERANCE = 1e-4 #early stopping tolderance 
    PATIENCE = 20  #num of iterations of no minimization of val loss
    STANDARDIZE = True

    # Loop through all combinations of parameters
    expt_results = []

    total_iterations = (
        len(SEASONS) * len(POSITIONS) * len(WINDOW_SIZES) * len(KERNEL_SIZES) *
        len(NUM_FILTERS) * len(NUM_DENSE) * len(LOW_PLAYTIME_CUTOFF) *
        len(AMT_NUM_FEATURES) * len(OPTIMIZERS) * len(REGULARIZATIONS))

    for (season, position, window_size, kernel_size, num_filters, num_dense,
        low_playtime_cutoff, amt_num_feature, 
        optimizer, regularization) in tqdm(itertools.product(
        SEASONS, POSITIONS, WINDOW_SIZES, KERNEL_SIZES, NUM_FILTERS, NUM_DENSE,
        LOW_PLAYTIME_CUTOFF, AMT_NUM_FEATURES, 
        OPTIMIZERS, REGULARIZATIONS), total=total_iterations):

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
            'amt_num_features': amt_num_feature,
            'optimizer': optimizer,
            'regularization': regularization,
        }
        num_features = NUM_FEATURES_DICT[position][amt_num_feature]

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
                                        drop_low_playtime=DROP_LOW_PLAYTIME,
                                        low_playtime_cutoff=low_playtime_cutoff,
                                        num_features=num_features,
                                        regularization=regularization,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        cat_features=CAT_FEATURES,
                                        conv_activation=CONV_ACTIVATION,
                                        dense_activation=DENSE_ACTIVATION,
                                        optimizer=optimizer,
                                        learning_rate=learning_rate,
                                        loss='mse',
                                        metrics=['mae'],
                                        verbose=verbose,
                                        early_stopping=True,
                                        tolerance=TOLERANCE,
                                        patience=PATIENCE,
                                        plot=False,
                                        standardize=STANDARDIZE)
        
        expt_res['amt_num_features'] = amt_num_feature
        expt_results.append(expt_res)

    if verbose:
        print(f"Updating GridSearch Results Log File: {log_file}...")
    log_evals(log_file, expt_results, verbose=verbose)

    if verbose:
        print("======= Done with GridSearch Experiment ========")

    return

def main():
    """
    Run the experiment specified by gridsearch_cnn constants
    """
    gridsearch_cnn(epochs=100, verbose=False)
    return

if __name__ == '__main__':
    main()
