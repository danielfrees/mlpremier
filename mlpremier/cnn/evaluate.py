"""
Evaluate and visualize the performance of trained FPL CNN models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


#TODO: make a separate evaluate func, called by both log_evaluation and plot_learning_curve
# collect evaluations in a loop in the model.py code, then log_evaluations all at once
def log_evaluation(season,
                position,
                model, 
                history, 
                X_train, y_train, 
                X_val, y_val, 
                X_test, y_test, 
                log: bool = False,
                log_file: str = None,
                verbose: bool = False,
                **hyperparameters):
    train_mse, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_mse, val_mae = model.evaluate(X_val, y_val, verbose=0)
    test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    if log_file is None:
            log_file = os.path.join('results', 'gridsearch.csv')

    # Check if the log_file parent directory exists, if not, create it
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a pandas DataFrame if the log_file exists, otherwise create a new one
    COLUMNS = ['season', 'position', 'train_mse', 'val_mse', 'test_mse', 'train_mae', 'val_mae', 'test_mae'] + list(hyperparameters.keys())

    log_df = None
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=COLUMNS)

    # Create a pandas Series with hyperparameters and scalar values
    param_data = hyperparameters.copy()
    param_data['season'] = season
    param_data['position'] = position
    param_data['train_mse'] = train_mse
    param_data['train_mae'] = train_mae
    param_data['val_mse'] = val_mse
    param_data['val_mae'] = val_mae
    param_data['test_mse'] = test_mse
    param_data['test_mae'] = test_mae

    # Create the series with the explicitly defined column order
    series = pd.Series(param_data, index=COLUMNS)  

    # Append the series row-wise to the DataFrame if no equivalent row is already present
    if not any((log_df == series).all(1)):
        log_df = pd.concat([log_df, pd.DataFrame([param_data], columns=log_df.columns)], ignore_index=True)

    if verbose:
        print(f"Logging experiment results to {log_file}.")
    # Save the updated DataFrame back to the CSV file
    log_df.to_csv(log_file, index=False)

    return

    
def plot_learning_curve(season,
                        position,
                        model, 
                        history, 
                        X_train, y_train, 
                        X_val, y_val, 
                        X_test, y_test, 
                        log: bool = False,
                        log_file: str = None,
                        verbose: bool = False,
                        **hyperparameters):
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
    train_mse, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_mse, val_mae = model.evaluate(X_val, y_val, verbose=0)
    test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)

    # Provide 'labels' as 'x' and 'values' as 'height' in the plt.bar function
    plt.bar(['Train', 'Validation', 'Test'], [train_mse, val_mse, test_mse], color=['blue', 'orange', 'green'])
    plt.title('Final Loss for the Model')
    plt.xlabel('Dataset')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # Plot legend with hyperparameters
    if hyperparameters:
        plt.figure(figsize=(8, 2))
        plt.axis('off')

        hyperparam_str = '\n'.join(f'{key}: {value}' for key, value in hyperparameters.items())
        plt.text(0, 0.5, f'Hyperparameters:\n{hyperparam_str}', fontsize=12, va='center', bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    
    plt.show()

    return
