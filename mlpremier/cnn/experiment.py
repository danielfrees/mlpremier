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
from mlpremier.cnn.model import build_train_cnn, generate_datasets
from mlpremier.cnn.evaluate import eval_cnn, log_evals
import tensorflow as tf
import itertools
from tqdm import tqdm
import pickle

from config import STANDARD_CAT_FEATURES, STANDARD_NUM_FEATURES, NUM_FEATURES_DICT

def gridsearch_cnn(experiment_name: str = 'gridsearch',
                   verbose : bool = False):
    """
    GridSearch for Best Hyperparameters
    """
    log_file = os.path.join('results', 'gridsearch', f'{experiment_name}.csv')
    data_log_file = os.path.join('results', 'gridsearch', f'{experiment_name}_data.pkl')

    if verbose:
        print("======= Running GridSearch Experiment ========")

    EPOCHS = 250 
    SEED = 229
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    DATA_DIR = os.path.join(os.getcwd(), '..', 'data', 'clean_data')
    SEASONS = [['2020-21', '2021-22']]
    POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    WINDOW_SIZES = [3, 6, 9]
    KERNEL_SIZES = [1, 2, 3, 4]
    NUM_FILTERS = [64] #, 128, 256]
    NUM_DENSE = [64]# , 128, 256]
    CONV_ACTIVATION = 'relu'
    DENSE_ACTIVATION = 'relu'
    DROP_LOW_PLAYTIME = [False, True]
    LOW_PLAYTIME_CUTOFF = [1e-6]   # drop players who never play
    AMT_NUM_FEATURES = ['ptsonly', 'pts_ict'] #,'small', 'medium', 'large'] 
    CAT_FEATURES = STANDARD_CAT_FEATURES
    STRATIFY_BY = ['skill', 'stdev'] 
    OPTIMIZERS = ['adam'] #, 'sgd'
    REGULARIZATIONS = [0.01]  #L1 L2 reg strength
    TOLERANCE = 1e-4 #early stopping tolderance 
    PATIENCE = 20  #num of iterations of no minimization of val loss
    STANDARDIZE = True

    # Loop through all combinations of parameters
    expt_results = []

    total_iterations = (
        len(SEASONS) * len(POSITIONS) * len(WINDOW_SIZES) * len(KERNEL_SIZES) *
        len(NUM_FILTERS) * len(NUM_DENSE) * len(DROP_LOW_PLAYTIME) * 
        len(LOW_PLAYTIME_CUTOFF) *  len(AMT_NUM_FEATURES) * len(STRATIFY_BY) * 
        len(OPTIMIZERS) * len(REGULARIZATIONS))
    

    for (season, position, window_size, kernel_size, num_filters, num_dense,
        drop_low_playtime, low_playtime_cutoff, amt_num_feature, stratify_by,
        optimizer, regularization) in tqdm(itertools.product(
        SEASONS, POSITIONS, WINDOW_SIZES, KERNEL_SIZES, NUM_FILTERS, NUM_DENSE,
        DROP_LOW_PLAYTIME, LOW_PLAYTIME_CUTOFF, AMT_NUM_FEATURES, STRATIFY_BY,
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
        (X_train, d_train, y_train, 
        X_val, d_val, y_val, 
        X_test, d_test, y_test, pipeline) = generate_datasets(data_dir=DATA_DIR,
                                            season=season,
                                            position=position, 
                                            window_size=window_size,
                                            num_features=num_features,
                                            cat_features=CAT_FEATURES,
                                            stratify_by=stratify_by,
                                            drop_low_playtime=drop_low_playtime,
                                            low_playtime_cutoff=low_playtime_cutoff,
                                            verbose=verbose)
    
        #call build_train_cnn passing on all params 
        model, expt_res = build_train_cnn(
                X_train=X_train, d_train=d_train, y_train=y_train,
                X_val=X_val, d_val=d_val, y_val=y_val,
                X_test=X_test, d_test=d_test, y_test=y_test,
                season=season,
                position=position,
                window_size=window_size,
                kernel_size=kernel_size,
                num_filters=num_filters,
                num_dense=num_dense,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                drop_low_playtime=drop_low_playtime,
                low_playtime_cutoff=low_playtime_cutoff,
                num_features=num_features,
                cat_features=CAT_FEATURES,
                conv_activation=CONV_ACTIVATION,
                dense_activation=DENSE_ACTIVATION,
                optimizer=optimizer,
                learning_rate=LEARNING_RATE,
                loss='mse',
                metrics=['mae'],
                verbose=verbose,
                regularization=regularization,
                early_stopping=True,
                tolerance=TOLERANCE,
                patience=PATIENCE,
                plot=False,
                draw_model=False,
                standardize=STANDARDIZE
            )
        
        expt_res['amt_num_features'] = amt_num_feature
        dataset_info = {'X_train': X_train, 'd_train': d_train, 'y_train': y_train, 
                        'X_val': X_val, 'd_val': d_val, 'y_val': y_val, 
                        'X_test': X_test, 'd_test': d_test, 'y_test': y_test}
        serialized_dataset = pickle.dumps(dataset_info)
        expt_res['dataset'] = serialized_dataset
        expt_res['pipeline'] = pickle.dumps(pipeline)
        expt_res['stratify_by'] = stratify_by
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
    gridsearch_cnn(verbose=False)
    return

if __name__ == '__main__':
    main()
