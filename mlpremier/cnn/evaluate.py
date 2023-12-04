"""
Evaluate and visualize the performance of trained FPL CNN models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def eval_cnn(season, position, model, 
             X_train, y_train, 
             X_val, y_val, 
             X_test, y_test,
             **hyperparameters) -> dict:
    """
    Gets a dict of hyperparameter and MSE, MAE Evaluation data.
    """
    train_mse, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_mse, val_mae = model.evaluate(X_val, y_val, verbose=0)
    test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)

    eval_data = {
        'season': season,
        'position': position,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'val_mse': val_mse,
        'val_mae': val_mae,
        'test_mse': test_mse,
        'test_mae': test_mae,
        **hyperparameters
    }

    return eval_data

def log_evals(log_file, eval_data_list, verbose=False):
    """
    Creates a df based on a list of eval_cnn dicts, then concats this 
    with the log_file df.
    """
    COLUMNS = ['season', 'position', 'train_mse', 
               'val_mse', 'test_mse', 'train_mae', 
               'val_mae', 'test_mae'] + list(eval_data_list[0].keys())[8:]

    # Check if the log_file parent directory exists, if not, create it
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a pandas DataFrame if the log_file exists, otherwise create a new one
    log_df = None
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=COLUMNS)

    # Create a DataFrame from the list of eval_cnn dicts
    eval_df = pd.DataFrame(eval_data_list, columns=COLUMNS)

    # Concatenate the evaluation DataFrame with the log DataFrame
    log_df = pd.concat([log_df, eval_df], ignore_index=True)

    if verbose:
        print(f"Logging experiment results to {log_file}.")

    # Save the updated DataFrame back to the CSV file
    log_df.to_csv(log_file, index=False)

    return
    
def plot_learning_curve(season,
                        position,
                        history,
                        expt_res,  # experimental results (dictionary)
                        verbose: bool = False):
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

