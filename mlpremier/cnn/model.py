"""
Construct and train CNN models for the FPL Regression Task.
"""
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Conv1D
from tensorflow.keras.layers import Dense, Activation, Concatenate
from mlpremier.cnn.preprocess import generate_cnn_data, split_preprocess_cnn_data
from typing import Tuple, List
import os
from mlpremier.cnn.evaluate import plot_learning_curve, eval_cnn, log_evals

from config import STANDARD_CAT_FEATURES, STANDARD_NUM_FEATURES

def create_cnn(X_input_shape: Tuple, 
               d_input_shape: Tuple,
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
    X_input = Input(shape=X_input_shape)
    convnet = Conv1D(filters=num_filters,
                    kernel_size=kernel_size,
                    activation=conv_activation,
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=regularization,
                                                                  l2=regularization))(X_input)
    convnet = Flatten()(convnet)

    # add extra input point on matchup difficulty after convolution
    d_input = Input(shape=d_input_shape)

    # using the Functional API to concatenate match difficulty and the convolutional flattened output
    merged_layer = Concatenate(axis=-1)([convnet, d_input])

    dense_layer = Dense(units=num_dense,
                       activation=dense_activation,
                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=regularization,
                                                                     l2=regularization))(merged_layer)

    output_layer = Dense(units=1, activation='linear')(dense_layer)

    # Create the model using the Functional API!!
    model = Model(inputs=[X_input, d_input], outputs=output_layer)


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

def generate_datasets(data_dir: str,
              season: str, 
              position: str, 
              window_size: int,
              drop_low_playtime : bool = True,
              low_playtime_cutoff : int = 25,
              num_features: List[str] = STANDARD_NUM_FEATURES,
              cat_features: List[str] = STANDARD_CAT_FEATURES, 
              test_size: float = 0.15, 
              val_size: float = 0.3, 
              stratify_by: str = 'skill',
              standardize: bool = True,
              verbose: bool = False):
    
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
    
    (X_train, d_train, y_train, 
     X_val, d_val, y_val, 
     X_test, d_test, y_test, pipeline) = split_preprocess_cnn_data(df, 
                                                            features_df, 
                                                            test_size=test_size,
                                                            val_size=val_size,
                                                            stratify_by=stratify_by, 
                                                            num_features=num_features,
                                                            cat_features=cat_features,
                                                            standardize=standardize,
                                                            return_pipeline=True,
                                                            verbose=verbose)
    
    return X_train, d_train, y_train, X_val, d_val, y_val, X_test, d_test, y_test, pipeline  #return split data and stdscale pipe

def build_train_cnn(X_train, d_train, y_train,
                    X_val, d_val, y_val,
                    X_test, d_test, y_test,
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
                    draw_model: bool = False, 
                    standardize: bool = True):

    # X has shape (num_examples, window_size, features)
    X_input_shape = (window_size, X_train.shape[2])  

    # difficulty has shape (num_examples, 1, )
    d_input_shape = (1,)

    model = create_cnn(X_input_shape=X_input_shape, 
                       d_input_shape=d_input_shape,
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
        history = model.fit([X_train, d_train], y_train, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            validation_data=([X_val, d_val], y_val),
                            callbacks = [early_stop],
                            verbose = 0) 
    else:
        history = model.fit([X_train, d_train], y_train, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            validation_data=([X_val, d_val], y_val),
                            verbose = 0) 
    

    # ============= Evaluate the Model ==============
    expt_res = eval_cnn(season=season,
            position=position,
            model=model,
            X_train=X_train, d_train=d_train, y_train=y_train,
            X_val=X_val, d_val=d_val, y_val=y_val,
            X_test=X_test, d_test=d_test, y_test=y_test,
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
    if draw_model:
        tf.keras.utils.plot_model(
            model,
            to_file='model.png',
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=False,
            show_trainable=False
        )

    return model, expt_res #return model, results

def full_cnn_pipeline(data_dir: str, 
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
                    draw_model: bool = False, 
                    standardize: bool = True,
                    test_size: float = 0.15, 
                    val_size: float = 0.3,
                    stratify_by: str = 'skill' 
                    ):
    
    # Generate datasets

    (X_train, d_train, y_train, 
     X_val, d_val, y_val, 
     X_test, d_test, y_test, pipeline) = generate_datasets(data_dir=data_dir,
                                season=season,
                                position=position, 
                                window_size=window_size,
                                num_features=num_features,
                                cat_features=cat_features,
                                stratify_by=stratify_by,
                                test_size=test_size,
                                val_size=val_size,
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
        batch_size=batch_size,
        epochs=epochs,
        drop_low_playtime=drop_low_playtime,
        low_playtime_cutoff=low_playtime_cutoff,
        num_features=num_features,
        cat_features=cat_features,
        conv_activation=conv_activation,
        dense_activation=dense_activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss=loss,
        metrics=metrics,
        verbose=verbose,
        regularization=regularization,
        early_stopping=early_stopping,
        tolerance=tolerance,
        patience=patience,
        plot=plot,
        draw_model=draw_model,
        standardize=standardize
    )

    return model, expt_res
