"""
Evaluate and visualize the performance of trained FPL CNN models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from pandas.plotting import table
from IPython.display import display
from typing import Union, List
import pickle
from scipy.stats import spearmanr
import tensorflow as tf

def get_spearman_rankcor(y_true, y_pred):
    """ 
    Calculate Spearman correlation of model given predictions and true values.
    """
    spearman_corr, _ = tf.py_function(spearmanr, 
                                      [y_pred, y_true], 
                                      (tf.float32, tf.float32))
    return float(spearman_corr)

def plot_preds(y_true, y_pred, 
               season, 
               position,
               verbose: bool = False):
    """ 
    Plot a scatterplot of Predictions vs. True Player Pts with mean predictions
    
    Parameters:
    - y_true: List or array containing true player points
    - y_pred: List or array containing predicted player points
    """
    unique_true_points = np.unique(y_true)
    
    plt.scatter(y_true, y_pred, alpha=0.15)
    plt.scatter(unique_true_points, [np.median(y_pred[y_true == x]) for x in unique_true_points], color='red', marker='D', label='Median Prediction')
    
    plt.title(f'Predictions vs. True Player Points (Season: {season}, Pos: {position})')
    plt.xlabel('True Player Points')
    plt.ylabel('Predicted Player Points')
    plt.legend()
    if verbose:
        plt.show()


def eval_cnn(season, position, model, 
             X_train, d_train, y_train, 
             X_val, d_val, y_val, 
             X_test, d_test, y_test,
             verbose: bool = False,
             **hyperparameters) -> dict:
    """
    Evaluate the given model on the provided data. 
    Returns a dict of evaluation results and the hyperparameters used 
    for the model.
    """
    train_mse, train_mae = model.evaluate([X_train, d_train], y_train, verbose=0)
    val_mse, val_mae = model.evaluate([X_val, d_val], y_val, verbose=0)
    test_mse, test_mae = model.evaluate([X_test, d_test], y_test, verbose=0)
    y_pred = model.predict([X_test, d_test])
    spearman_corr = get_spearman_rankcor(y_test, y_pred)

    if verbose:
        plot_preds(y_test, y_pred, season, position)
    
    eval_data = {
        'season': season,
        'position': position,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'val_mse': val_mse,
        'val_mae': val_mae,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'spear_corr': spearman_corr,
        **hyperparameters
    }

    return eval_data

def log_evals(log_file, eval_data_list, verbose=False):
    """
    Log the results of several evaluations to a csv file specified by log_file.

    Creates the log_file if it doesn't exists, appends results otherwise.
    """
    COLUMNS = ['season', 'position', 'train_mse', 
               'val_mse', 'test_mse', 'train_mae', 
               'val_mae', 'test_mae'] + list(eval_data_list[0].keys())[8:]

    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a pandas DataFrame if the log_file exists, otherwise create a new one
    log_df = None
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=COLUMNS)

    eval_df = pd.DataFrame(eval_data_list, columns=COLUMNS)
    log_df = pd.concat([log_df, eval_df], ignore_index=True)

    if verbose:
        print(f"Logging experiment results to {log_file}.")

    log_df.to_csv(log_file, index=False)

    return
    
def plot_learning_curve(season,
                        position,
                        history,
                        expt_res,  # experimental results (dictionary)
                        verbose: bool = False):
    """
    Plot train/validation learning curve, barplot of train/valid/test MSE results,
    and a simple text display of the hyperparameters used in the experiment.
    """
    # Plot learning curve
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'FPL Regression Model Training, Season: {season}, Pos: {position}', fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot histograms of scores
    plt.subplot(1, 2, 2)
    train_mse, train_mae = expt_res['train_mse'], expt_res['train_mae']
    val_mse, val_mae = expt_res['val_mse'], expt_res['val_mae']
    test_mse, test_mae = expt_res['test_mse'], expt_res['test_mae']

    # Provide 'labels' as 'x' and 'values' as 'height' in the plt.bar function
    plt.bar(['Train', 'Validation', 'Test'], [train_mse, val_mse, test_mse], color=['blue', 'orange', 'green'])
    plt.title('Final Loss for the Model')
    plt.xlabel('Dataset')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # Plot legend with hyperparameters
    not_hyperparams = ['train_mse', 'train_mae', 'val_mse', 'val_mae', 'test_mse', 'test_mae']
    hyperparameters = {key: value for key, value in expt_res.items() if key not in not_hyperparams}

    if hyperparameters:
        plt.figure(figsize=(8, 2))
        plt.axis('off')

        hyperparam_str = '\n'.join(f'{key}: {value}' for key, value in hyperparameters.items())
        plt.text(0, 0.5, f'Hyperparameters:\n{hyperparam_str}', fontsize=12, va='center', bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    
    plt.show()

    return

def eda_and_plot(features_df,
                 verbose: bool  =False):
    """
    Perform Exploratory Data Analysis (EDA) on the given DataFrame and plots
    histograms for numerical variables.

    Parameters:
    - features_df (pd.DataFrame): The DataFrame containing features for analysis.

    Returns:
    None
    """
    # Display selected statistics for each numerical feature
    numerical_statistics = features_df.describe().loc[['mean', 'min', 'max', 'std']]
    print("Selected Statistics:")
    print(numerical_statistics)

    # Plot histograms for numerical variables in a single row
    numerical_columns = features_df.select_dtypes(include=['int64', 'float64']).columns
    num_numerical_columns = len(numerical_columns)
    
    # Define the number of subplots per row (adjust as needed)
    subplots_per_row = 4

    # Calculate the number of rows needed
    num_rows = (num_numerical_columns + subplots_per_row - 1) // subplots_per_row

    # Create subplots dynamically
    plt.figure(figsize=(16, 4 * num_rows))

    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(num_rows, subplots_per_row, i)
        features_df[column].plot(kind='hist', bins=20, edgecolor='black')
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.tight_layout()
    if verbose:
        plt.show()

def gridsearch_analysis(expt_name: str = 'gridsearch', 
                        season: str = "['2020-21', '2021-22']",
                        eval_top: int = 5, 
                        get_dataset: bool = False, #whether to return datasets for a closer look (not supported by pre-v9 gridsearch results)
                        **kwargs):
    """
    Visualizes and investigates the results of a gridsearch for best CNN hyperparams.
    """
    DATA_DIR = os.path.join(os.getcwd(), '..', 'data', 'clean_data')

    expt_res_path = os.path.join('results', 'gridsearch', f'{expt_name}.csv')

    # =========== Get Gridsearch MSE + Hyperparams Results and Sort ========
    df = pd.read_csv(expt_res_path)   # get the gridsearch MSE and hyperparams data
    df = df[df['season'].apply(lambda x: repr(x) == repr(season))]
    # Filter by additional keyword arguments
    for key, value in kwargs.items():
        df = df[df[key] == value]
    df = df.sort_values(by='val_mse')

    # ======== Color MSE Results from Green (good) to Red (bad) =============
    mse_columns = df.filter(like='mse')
    color_range = [mse_columns.min().min(), mse_columns.max().max()]
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)  # green good red bad
    heatmap = sns.heatmap(mse_columns, cmap=cmap, vmin=color_range[0], vmax=color_range[1])
    heatmap.set_yticklabels(heatmap.get_yticklabels(), size=5)
    plt.title(f"{season} CNN FPL Experiments, sorted by Val MSE")
    plt.ylabel("Experiment Index")
    plt.show()

    # ========= Analyze Mode Params for Top  Models for Each Posn ===========
    positions = ['GK', 'DEF', 'MID', 'FWD']

    best_params_df = pd.DataFrame()
    performance_df = pd.DataFrame()

    for position in positions:
        top_models = df[df['position'] == position].head(eval_top)
        top_models_filtered = top_models.loc[:, ~top_models.columns.str.contains('mse|mae|verbose|corr')]

        top_hyperparams = top_models_filtered.mode().iloc[0]
        top_means = top_models.loc[:, top_models.columns.str.contains('mse|mae|corr')].astype(float).mean()

        best_params_df[position] = top_hyperparams
        performance_df[position] = top_means

    best_params_df = best_params_df.T
    performance_df = performance_df.T

    # Print or display the tables
    print(f"\n{season} Mode Best Hyperparameters for Each Position")
    print(f"Via Top {eval_top} Models by Position")
    display(best_params_df)

    print(f"\n{season} Mean Performance of Top {eval_top} Model by Position")
    display(performance_df)

    print("\nAverage Val MSE:")
    print(performance_df['val_mse'].mean())

    print("\nAverage Test MSE:")
    print(performance_df['test_mse'].mean())

    return best_params_df

