"""
Construct and train CNN models for the FPL Regression Task.
"""
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping 
from mlpremier.cnn.preprocess import generate_cnn_data, split_preprocess_cnn_data
from typing import Tuple, List
import os
from mlpremier.cnn.evaluate import plot_learning_curve, eval_cnn, log_evals

STANDARD_NUM_FEATURES = ['minutes', 'goals_scored', 'assists', 'goals_conceded',
                          'clean_sheets', 'bps', 'yellow_cards', 'red_cards', 
                          'own_goals', 'saves', 'penalties_missed', 'penalties_saved',
                          'ict_index', 'total_points']
STANDARD_CAT_FEATURES = []

def create_cnn(input_shape: Tuple, 
               kernel_size: int,
               num_filters: int,
               num_dense: int,
               conv_activation: str = 'relu',
               dense_activation: str = 'relu',
               optimizer: str = 'adam',
               learning_rate: float = 0.001,
               loss: str = 'mse',
               metrics: List[str] = ['mae'],
               verbose: bool = False,
               regularization: float = 0.001, 
               tolerance: float = 1e-3,
               ):  

    if verbose:
        print("====== Building CNN Architecture ======")

    """
    model = Sequential()
    # 1D convolution over the performance window inputs with regularization
    model.add(Conv1D(filters=num_filters, 
                     kernel_size=kernel_size, 
                     activation=conv_activation, 
                     input_shape=input_shape,
                     kernel_regularizer=l2(regularization)))  # Add regularization here
    model.add(MaxPool1D)
    # Fully connected NN layers with regularization
    model.add(Dense(num_filters, activation=dense_activation, kernel_regularizer=l2(regularization)))
    model.add(Dense(num_filters, activation=dense_activation, kernel_regularizer=l2(regularization)))
    # Output layer with linear activation for regression
    model.add(Dense(1, activation='linear', kernel_regularizer=l2(regularization)))
    """
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=num_filters,
                               kernel_size=kernel_size,
                               activation=conv_activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=num_dense,activation=dense_activation),
        tf.keras.layers.Dense(units=1,activation='linear'),
    ])

    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)  
    
    # Stop early if change in validation loss for 10 iterations is 
    # < tolerance
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=10, 
                                   mode='min', 
                                   min_delta=tolerance, 
                                   verbose=1)

    if verbose:
        print("====== Done Building CNN Architecture ======")

    return model, early_stopping

def build_train_cnn(data_dir: str,
              season: str, 
              position: str,  
              window_size: int,
              kernel_size: int,
              num_filters: int,
              num_dense: int,
              batch_size: int = 50,
              epochs: int = 500,  
              drop_low_playtime : bool = True,
              low_playtime_cutoff : int = 25,
              num_features: List[str] = STANDARD_NUM_FEATURES,
              cat_features: List[str] = STANDARD_CAT_FEATURES, 
              conv_activation: str = 'relu',
              dense_activation: str = 'relu',
              optimizer: str = 'adam',
              learning_rate: float = 0.001,
              loss: str = 'mse',
              metrics: List[str] = ['mae'],
              verbose: bool = False,
              regularization: float = 0.001, 
              tolerance: float = 1e-3,
              plot: bool = False, 
              standardize: bool = True):
    
    df, features_df = generate_cnn_data(data_dir=data_dir,
                         season=season, 
                         position=position, 
                         window_size=window_size,
                         num_features=num_features, 
                         cat_features=cat_features,
                         drop_low_playtime=drop_low_playtime,
                         low_playtime_cutoff=low_playtime_cutoff,
                         verbose = verbose)
    X_train, y_train, X_val, y_val, X_test, y_test = split_preprocess_cnn_data(df, 
                                                            features_df, 
                                                            num_features=num_features,
                                                            cat_features=cat_features,
                                                            standardize=standardize,
                                                            verbose=verbose)

    # X has shape (num_examples, window_size, features)
    input_shape = (window_size, X_train.shape[2])  

    model, early_stopping = create_cnn(input_shape=input_shape, 
                       kernel_size=kernel_size, 
                       num_filters=num_filters, 
                       num_dense=num_dense,
                       conv_activation=conv_activation,
                       dense_activation=dense_activation,
                       learning_rate=learning_rate,
                       optimizer=optimizer,
                       loss=loss,
                       metrics=metrics,
                       verbose=verbose,
                       regularization=regularization,
                       tolerance=tolerance)

    # Train the model
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=(X_val, y_val)) #callbacks = [early_stopping]
    

    # Evaluate Loss and Metrics, Assign to dictionary along with experiment hyperparams
    expt_res = eval_cnn(season=season,
            position=position,
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            verbose=verbose,
            kernel_size=kernel_size, 
            num_filters=num_filters, 
            conv_activation=conv_activation,
            dense_activation=dense_activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            metrics=metrics,
            regularization=regularization,
            standardize=standardize,
            num_features=num_features,
            cat_features=cat_features)
    
    # Print Test Set Evaluation
    test_loss, test_mae = (expt_res['test_mse'], expt_res['test_mae'])
    print(f'Test Loss (MSE): {test_loss}, Test Mean Absolute Error (MAE): {test_mae}')

    if plot:
        plot_learning_curve(season,
                            position,
                            history, 
                            expt_res)

    return model, expt_res
