from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from data_prep import preprocess_cnn_data, split_cnn_data

def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer with linear activation for regression

    model.compile(optimizer='sgd', loss='mse', metrics=['mae'])  # Using Mean Squared Error (MSE) loss for regression
    
    return model

def train_cnn():
    window_size = 6
    player_folder = 'path_to_player_folder'  # Change this to the actual path

    X, y = preprocess_cnn_data(player_folder, window_size)
    X_train, y_train, X_val, y_val, X_test, y_test = split_cnn_data(X, y)

    input_shape = (window_size, X.shape[2])  # Assuming the third dimension represents the number of features

    model = create_model(input_shape)

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=your_batch_size,
                        validation_data=(X_val, y_val))

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f'Test Loss (MSE): {test_loss}, Test Mean Absolute Error (MAE): {test_mae}')
