import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_folder, window_size):
    all_data = []

    # Iterate through files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            player_data = pd.read_csv(os.path.join(data_folder, filename))

            # Check if the player has a valid position (GKP, DEF, MID, FWD)
            if player_data['position'].iloc[0] in ['GKP', 'DEF', 'MID', 'FWD']:
                features = player_data.iloc[:, 1:-1]
                targets = player_data.iloc[:, -1].values

                # Create training samples using the specified window size
                X, y, player_names = [], [], []
                for i in range(len(player_data) - window_size):
                    X.append(features.iloc[i:i + window_size])
                    y.append(targets[i + window_size])
                    player_names.append(player_data['player_name'].iloc[i + window_size])

                all_data.extend(list(zip(player_names, X, y)))

    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(all_data, columns=['player_name', 'features', 'target'])

    return df

def split_data(df):
    # Split data into 70% train and 30% test (by player)
    players = df['player_name'].unique()
    players_train, players_test = train_test_split(players, test_size=0.3, shuffle=False)

    # Further split 10% of training data for validation
    players_train, players_val = train_test_split(players_train, test_size=0.1, shuffle=False)

    # Filter data for train, validation, and test sets
    train_data = df[df['player_name'].isin(players_train)]
    val_data = df[df['player_name'].isin(players_val)]
    test_data = df[df['player_name'].isin(players_test)]

    # Drop player name from features
    X_train = np.array(train_data['features'].tolist())
    X_val = np.array(val_data['features'].tolist())
    X_test = np.array(test_data['features'].tolist())

    y_train = np.array(train_data['target'])
    y_val = np.array(val_data['target'])
    y_test = np.array(test_data['target'])

    return X_train, y_train, X_val, y_val, X_test, y_test
