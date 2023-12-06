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

from config import STANDARD_CAT_FEATURES, STANDARD_NUM_FEATURES

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
               regularization: float = 0.001):  

    if verbose:
        print("====== Building CNN Architecture ======")

    # ================= Set up Model Architecture ==================
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=num_filters,
                               kernel_size=kernel_size,
                               activation=conv_activation,
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=regularization, 
                                                                             l2=regularization)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=num_dense,
                              activation=dense_activation, 
                              kernel_regularizer=tf.keras.regularizers.L1L2(l1=regularization, 
                                                                             l2=regularization)),                                                                    
        tf.keras.layers.Dense(units=1,activation='linear'),
    ])

    # ============= Set Optimizer and Compile Model =================
    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    elif optimizer == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        print(f"Using passed optimizer directly: {optimizer}. If you want a Keras optimizer pass 'adam' or 'sgd")
        optimizer = optimizer

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)  

    if verbose:
        print("====== Done Building CNN Architecture ======")

    return model

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
              early_stopping: bool = False,
              tolerance: float = 1e-5,
              patience: int = 40, 
              plot: bool = False, 
              standardize: bool = True,
              test_size: float = 0.15, 
              val_size: float = 0.3):
    
    # =========== Generate CNN Dataset  ============
    # == for Desired Season, Posn, Window Size =====
    # ===== and Feature Engineering Settings =======
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
                                                            test_size=test_size,
                                                            val_size=val_size,
                                                            num_features=num_features,
                                                            cat_features=cat_features,
                                                            standardize=standardize,
                                                            verbose=verbose)

    # X has shape (num_examples, window_size, features)
    input_shape = (window_size, X_train.shape[2])  

    model = create_cnn(input_shape=input_shape, 
                        kernel_size=kernel_size, 
                        num_filters=num_filters, 
                        num_dense=num_dense,
                        conv_activation=conv_activation,
                        dense_activation=dense_activation,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        loss=loss,
                        metrics=metrics,
                        regularization=regularization,
                        verbose=verbose,)
    
    # ============ Generate EarlyStopping Callback ==============
    early_stop = EarlyStopping(monitor='val_loss', 
                                   patience=patience, 
                                   mode='min', 
                                   min_delta=tolerance, 
                                   verbose=1)

    # =========== Run Model Fitting ==================
    history = None
    if early_stopping:
        history = model.fit(X_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks = [early_stop],
                            verbose = 0) 
    else:
        history = model.fit(X_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            verbose = 0) 
    

    # ============= Evaluate the Model ==============
    expt_res = eval_cnn(season=season,
            position=position,
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            verbose=verbose,
            window_size=window_size,
            kernel_size=kernel_size, 
            num_filters=num_filters, 
            num_dense=num_dense,
            conv_activation=conv_activation,
            dense_activation=dense_activation,
            drop_low_playtime=drop_low_playtime,
            low_playtime_cutoff=low_playtime_cutoff,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            metrics=metrics,
            regularization=regularization,
            early_stopping=early_stopping,
            tolerance=tolerance,
            patience=patience,
            standardize=standardize)
    
    # Print Test Set Evaluation
    test_loss, test_mae = (expt_res['test_mse'], expt_res['test_mae'])
    print(f'Test Loss (MSE): {test_loss}, Test Mean Absolute Error (MAE): {test_mae}')

    # =========== Plot Learning Curve for CNN ==================
    if plot:
        plot_learning_curve(season,
                            position,
                            history, 
                            expt_res)

    return model, expt_res
