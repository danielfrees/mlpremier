import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List

def get_avg_playtime(player_data: pd.DataFrame) -> int:
    """
    Averages the 'minutes' column of the passed DataFrame and returns the 
    mean playtime as an integer.

    :param pd.DataFrame player_data: DataFrame containing player data.
    
    :return: Average playtime for the provided player data.
    :rtype: int
    """
    # Check if 'minutes' column exists in the DataFrame
    if 'minutes' in player_data.columns:
        avg_playtime = player_data['minutes'].mean()
        return int(avg_playtime)
    else:
        # If 'minutes' column is not present, return 0
        return 0

def generate_cnn_data(data_dir : str, 
                        season : str,
                        position : str, 
                        window_size : int,
                        drop_low_playtime : bool = True,
                        low_playtime_cutoff : int = 35,
                        verbose: bool = False) -> Tuple[pd.DataFrame]:
    """
    Load and shape cnn data for a specific season and position. 

    :param str data_dir: Path to the top-level directory containing player data.
    :param str season: Season of data to preprocess. (Should match title of 
        desired season folder). TODO: Pass 'all' to preprocess all seasons into
        one dataset.
    :param str position: Position (GK, DEF, MID, FWD).
    :param int window_size: Size of the data window.
    :param bool drop_low_playtime: Whether or not to drop players that have low 
        average game minutes (threshold defined by `low_playtime_cutoff`).
    :param int low_playtime_cutoff: The cutoff (in avg. minutes) for which to drop 
        players for low playtime. Only used if `drop_low_playtime` set to True. 

    :return: Tuple of DataFrames
        (1) DataFrame containing preprocessed data with columns as follows:
            'name' - name of the player
            'features' - DF containing window_size of data starting at some gw
            'target' - prediction value for the week following the 'features' window
        (2) DataFrame containing all player-weeks features combined for the full_data
    :rtype: Tuple(pd.DataFrame)
    """
    all_windowed_data = []
    all_features = []
    ct_players = 0
    ct_dropped_players = 0

    if verbose:
        print(f"======= Generating CNN Data for Season: {season}, Position: {position} =======")
        if drop_low_playtime:
            print(f"Dropping Players with Avg. Playtime < {low_playtime_cutoff}...\n")

    position_folder = os.path.join(data_dir, season, position)
    for player_folder in os.listdir(position_folder):
        ct_players += 1
        player_path = os.path.join(position_folder, player_folder)
        if os.path.isdir(player_path):
            player_csv = os.path.join(player_path, 'gw.csv')
            if os.path.isfile(player_csv):
                player_data = pd.read_csv(player_csv)

                # drop players with low avg playtime if requested
                if drop_low_playtime and get_avg_playtime(player_data) < low_playtime_cutoff:
                    ct_dropped_players += 1
                    continue

                # all features except name 
                features = player_data.iloc[:, 1:] 
                # all players should have same position because of how we split 
                # the data for separate position modeling
                features.drop('position', axis=1) 
                # FPL pts
                targets = player_data.iloc[:, -1].values

                # Create training samples using the specified window size
                X, y, player_names = [], [], []
                for i in range(len(player_data) - window_size):
                    X.append(features.iloc[i:i + window_size]) # get window of features
                    y.append(targets[i + window_size]) # get next weeks FPL points
                    player_names.append(player_data['name'].iloc[i + window_size])

                all_windowed_data.extend(list(zip(player_names, X, y)))
                all_features.append(features)

    windowed_df = pd.DataFrame(all_windowed_data, columns=['name', 'features', 'target'])
    combined_features_df =  pd.concat(all_features)

    if verbose:
        print(f"Total players of type {position} = {ct_players}.")
        print(f"{ct_dropped_players} players dropped due to low average playtime.")
        print(f"Generated windowed dataframe for CNN of shape: {windowed_df.shape}.")
        print(f"Generated combined features dataframe for preprocessing of shape: {combined_features_df.shape}.\n")
        
        print(f"========== Done Generating CNN Data ==========\n")
    

    return windowed_df, combined_features_df


def preprocess_cnn_data(windowed_df: pd.DataFrame, 
                         combined_features_df: pd.DataFrame,
                         train_players: List[str],
                         verbose: bool = False):
    """
    StandardScale and One Hot Encode the features in a DataFrame for preparation
        for training the 1D CNN.

    :param pd.DataFrame windowed_df: Input DataFrame of form output by 
        `generate_cnn_data`[0] with 'name', 'features', 'target' columns.
        All of the player-window dataframes in 'features' will be scaled and 
        one-hot-encoded.
    :param pd.DataFrame combined_features_df: Input DataFrame of form output by 
        `generate_cnn_data`[1] containing all player-week features combined together.
    :param List[str] train_players: List of player names in the training dataset.
        Only these rows will be used for fitting the preprocessing Pipeline so as 
        to avoid data leakage.

    :return: Edited windowed_df with standardized player-window dfs in the 
        'features' column.
    :rtype: pd.DataFrame
    """

    if verbose:
        print(f"========== Preprocessing CNN Data ==========\n")

    df = windowed_df.copy()

    numerical_features = ['minutes', 'goals_scored', 'assists', 'goals_conceded',
                          'clean_sheets', 'bps', 'yellow_cards', 'red_cards', 
                          'own_goals', 'saves', 'penalties_missed', 'penalties_saved',
                          'ict_index', 'influence', 'creativity', 'threat', 
                          'total_points']
    categorical_features = ['team', 'opponent_team'] #,'team','opponent_team'] 

    # Create transformers for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Include non-specified columns in the result
    )

    # Create a pipeline to apply the transformations
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit the pipeline on the combined DataFrame
    pipeline.fit(combined_features_df)

    if verbose:
        # Print mean and standard deviation of the Standard Scaler
        print("Mean of Standard Scaler:")
        print(pipeline.named_steps[
            'preprocessor'].named_transformers_['num'].mean_)
        print("\nStandard Deviation of Standard Scaler:")
        print(pipeline.named_steps
              ['preprocessor'].named_transformers_['num'].scale_)

    # Apply the fitted pipeline to each 'features' DataFrame 
    # from the original DataFrame
    if verbose: 
        print('Transforming features using StandardScaler + OHE Pipeline.')
    df['features'] = df['features'].apply(lambda features_df: 
                                          pipeline.transform(features_df))

    if verbose:
        print(f"========== Done Preprocessing CNN Data ==========\n")
    return df


def split_preprocess_cnn_data(windowed_df: pd.DataFrame,
                    combined_features_df: pd.DataFrame,
                    verbose: bool = False) -> Tuple[np.array]:
    """
    Split and preprocess CNN data into training, validation, and test sets. 

    :param pd.DataFrame windowed_df: Input DataFrame of form output by 
        `generate_cnn_data`[0] with 'name', 'features', 'target' columns.
        All of the player-window dataframes in 'features' will be scaled and 
        one-hot-encoded.
    :param pd.DataFrame combined_features_df: Input DataFrame of form output by 
        `generate_cnn_data`[1] containing all player-week features combined together.

    :return: Tuple of features and labels for training, validation, and test sets.
    :rtype: Tuple[np.array]
    """
    if verbose:
        print(f"========== Splitting CNN Data ==========\n")

    df = windowed_df.copy()
    if verbose:
        print(f"Shape of windowed_df: {windowed_df.shape}")
        print(f"Shape of a given window: {windowed_df.loc[0, 'features'].shape}")
    features_df = combined_features_df.copy()
    # Split data into 70% train and 30% test (by player)
    players = df['name'].unique()
    players_train, players_test = train_test_split(players, 
                                                   test_size=0.3, 
                                                   shuffle=True)

    # Further split 10% of training data for validation
    players_train, players_val = train_test_split(players_train, 
                                                  test_size=0.1, 
                                                  shuffle=True)
    
    df = preprocess_cnn_data(df, 
                             features_df, 
                             train_players = players_train, 
                             verbose = verbose)

    # Filter data for train, validation, and test sets
    train_data = df[df['name'].isin(players_train)]
    val_data = df[df['name'].isin(players_val)]
    test_data = df[df['name'].isin(players_test)]

    # Drop player name from features
    X_train = np.array(train_data['features'].tolist())
    X_val = np.array(val_data['features'].tolist())
    X_test = np.array(test_data['features'].tolist())

    y_train = np.array(train_data['target'])
    y_val = np.array(val_data['target'])
    y_test = np.array(test_data['target'])

    if verbose: 
        print(f"========== Done Splitting CNN Data ==========\n")

    return X_train, y_train, X_val, y_val, X_test, y_test

