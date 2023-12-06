"""
Preprocess data for an FPL Regression Model, given the desired architecture.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Union
from mlpremier.cnn.evaluate import eda_and_plot

from config import STANDARD_CAT_FEATURES, STANDARD_NUM_FEATURES


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
                        season : Union[str,List[str]],
                        position : str, 
                        window_size : int,
                        num_features: List[str] = STANDARD_NUM_FEATURES,
                        cat_features: List[str] = STANDARD_CAT_FEATURES,
                        drop_low_playtime : bool = True,
                        low_playtime_cutoff : int = 25,
                        verbose: bool = False) -> Tuple[pd.DataFrame]:
    """
    Load and shape cnn data for a specific season and position. 

    :param str data_dir: Path to the top-level directory containing player data.
    :param Union[str,List[str]] season: Season(s) of data to preprocess. 
        (Should match title(s) of desired season folder(s)). 
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

    if type(season) == list:
        seasons = season
    else:
        seasons = [season]

    # ================ Iterate through Clean Data and generate CNN data =================
    for season in seasons:
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

                    #DROP USELESS FEATURES
                    features = player_data.copy()
                    #name used only for subsetting to training data when running pipeline transform
                    cols_to_keep = num_features + cat_features + ['name'] 
                    features = features.loc[:, cols_to_keep] 
                    # FPL pts
                    targets = player_data.iloc[:, -1].values

                    # Create training samples using the specified window size
                    X, y, player_names, avg_scores, seasons = [], [], [], [], []
                    player_name = player_data['name'][0]
                    avg_score = player_data['total_points'].mean()
                    for i in range(len(player_data) - window_size):
                        X.append(features.iloc[i:i + window_size]) # get window of features
                        y.append(targets[i + window_size]) # get next weeks FPL points
                        player_names.append(player_name)
                        avg_scores.append(avg_score)
                        seasons.append(season)

                    all_windowed_data.extend(list(zip(player_names, avg_scores, seasons, X, y)))
                    all_features.append(features)

    windowed_df = pd.DataFrame(all_windowed_data, columns=['name', 'avg_score', 'season', 'features', 'target'])
    combined_features_df =  pd.concat(all_features)

    if verbose:
        print(f"Total players of type {position} = {ct_players}.")
        print(f"{ct_dropped_players} players dropped due to low average playtime.")
        print(f"Generated windowed dataframe for CNN of shape: {windowed_df.shape}.")
        print(f"Generated combined features dataframe for preprocessing of shape: {combined_features_df.shape}.\n")
        print(f"========== EDA ==========")
        # Distributions and EDA based on player-weeks (ie for DEF: ~38 gameweeks * 400 players)
        eda_and_plot(combined_features_df)
        print(f"========== Done Generating CNN Data ==========\n")
    

    return windowed_df, combined_features_df


def preprocess_cnn_data(windowed_df: pd.DataFrame, 
                         combined_features_df: pd.DataFrame,
                         train_players: List[str],
                         standardize: bool = True,
                         num_features: List[str] = STANDARD_NUM_FEATURES,
                         cat_features: List[str] = STANDARD_CAT_FEATURES,
                         verbose: bool = False):
    """
    StandardScale and One Hot Encode the features in a DataFrame for preparation
        for training the 1D CNN.

    :param pd.DataFrame windowed_df: Input DataFrame of form output by 
        `generate_cnn_data`[0] with 'name', 'features', 'target' columns.
        All of the player-window dataframes in 'features' will be scaled and 
        one-hot-encoded.
    :param pd.DataFrame train_features_df: Input DataFrame of form output by 
        `generate_cnn_data`[1] containing all player-week features combined together.
        Should only contain train players' features data to avoid leakage.
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

    numerical_features = num_features
    categorical_features = cat_features #,'team','opponent_team'] 

    # ========== Preprocessing pipeline for Numerical + Cat Features ===========
    numerical_transformer = None
    if standardize:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = 'passthrough'
    categorical_transformer = OneHotEncoder()

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        # remainder='passthrough'  # Include non-specified columns in the result
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # =========== Fit pipeline to training data features =================
    train_features_df = combined_features_df[combined_features_df['name'].isin(train_players)]
    pipeline.fit(train_features_df)

    if verbose:
        if standardize:
            # Print mean and standard deviation of the Standard Scaler
            print("Mean of Standard Scaler:")
            print(pipeline.named_steps[
                'preprocessor'].named_transformers_['num'].mean_)
            print("\nStandard Deviation of Standard Scaler:")
            print(pipeline.named_steps
                ['preprocessor'].named_transformers_['num'].scale_)
        else:
            print("No StandardScaler applied. standardize=False.")

    # ============ Actually apply pipeline to data ============
    if verbose: 
        print('Transforming features using StandardScaler + OHE Pipeline.')
    df['features'] = df['features'].apply(lambda features_df: 
                                          pipeline.transform(features_df))

    if verbose:
        print(f"========== Done Preprocessing CNN Data ==========\n")
    return df


def split_preprocess_cnn_data(windowed_df: pd.DataFrame,
                    combined_features_df: pd.DataFrame,
                    test_size: float = 0.15, 
                    val_size: float = 0.3, 
                    standardize: bool = True,
                    num_features: List[str] = STANDARD_NUM_FEATURES,
                    cat_features: List[str] = STANDARD_CAT_FEATURES,
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

    df = windowed_df.copy()  #names, player-window df, target data
    if verbose:
        print(f"Shape of windowed_df: {windowed_df.shape}")
        print(f"Shape of a given window (prior to preprocessing): {windowed_df.loc[0, 'features'].shape}")
    features_df = combined_features_df.copy()

    players_scores = df.groupby('name')[['name', 'avg_score']].first()
    num_players = len(df['name'].unique())

    # ===== Generate Skill Quantization for Stratify =========
    # quantize average score performance into bins for the stratification
    quantiles = None
    bins=None
    labels=None
    if num_players > 120:
        quantiles = np.percentile(players_scores['avg_score'], [10, 30, 50, 70, 90])
        bins = quantiles.tolist()
        bins = [-100] + bins 
        labels = ['shite', 'bad', 'bad+', 'mid', 'great', 'all-star']
        # Drop any quantiles that aren't unique - needed for highly modal dists.
        bins, labels = zip(*((b, l) for b, l in zip(bins, labels) if bins.count(b) == 1))
        bins = list(bins) + [100]
        labels = list(labels)
    else: #GKs
        quantiles = np.percentile(players_scores['avg_score'], [80, 90])
        bins = quantiles.tolist()
        bins = [-100] + bins 
        labels = ['shite', 'mid', 'great']
        # Drop any quantiles that aren't unique
        bins, labels = zip(*((b, l) for b, l in zip(bins, labels) if bins.count(b) == 1))
        bins = list(bins) + [100]
        labels = list(labels)
    players_scores['skill'] = pd.cut(players_scores['avg_score'], 
                                     bins=bins, 
                                     labels=labels)
    players = list(players_scores['name'])
    skills = list(players_scores['skill'])

    if verbose:
        print(f"Skill Distribution of Players:\n")
        skill_counts = players_scores['skill'].value_counts()
        colors = plt.cm.get_cmap('tab10', len(skill_counts))
        plt.figure(figsize=(3, 3))
        plt.pie(skill_counts, labels=skill_counts.index, autopct='%1.1f%%', colors=colors(range(len(skill_counts))))
        plt.title('Distribution of Skills')
        plt.show()


    # ========= Stratified Train/Val/Test Split by Skill =========
    # Split data into 85% train and 15% test 
    # We split by player so no windows of player performance (which overlap)
    # Can possibly be shared across splits. Necessary to avoid data leakage.
    players_train, players_test = train_test_split(players, 
                                                   test_size=test_size, 
                                                   shuffle=True,
                                                   stratify=skills)

    train_skills = list(players_scores[players_scores['name'].isin(players_train)]['skill'])
    # Further split 30% of training data for validation
    players_train, players_val = train_test_split(players_train, 
                                                  test_size=val_size, 
                                                  shuffle=True,
                                                  stratify=train_skills)
    
    # ================ Preprocess Data ==================
    df = preprocess_cnn_data(df, 
                            features_df, 
                            train_players = players_train, 
                            standardize = standardize,
                            num_features = num_features, 
                            cat_features = cat_features,
                            verbose = verbose)

    # ================ Generate train, val, test sets ====================
    train_data = df[df['name'].isin(players_train)]
    val_data = df[df['name'].isin(players_val)]
    test_data = df[df['name'].isin(players_test)]
    
    # =============== Assert no Data Leakage ==================
    assert len(set(train_data['name']).intersection(val_data['name'])) == 0, "Overlap between train and validation sets"
    assert len(set(train_data['name']).intersection(test_data['name'])) == 0, "Overlap between train and test sets"
    assert len(set(val_data['name']).intersection(test_data['name'])) == 0, "Overlap between validation and test sets"

    X_train = np.array(train_data['features'].tolist())
    X_val = np.array(val_data['features'].tolist())
    X_test = np.array(test_data['features'].tolist())

    y_train = np.array(train_data['target'])
    y_val = np.array(val_data['target'])
    y_test = np.array(test_data['target'])

    if verbose: 
        print(f"========== Done Splitting CNN Data ==========\n")

    return X_train, y_train, X_val, y_val, X_test, y_test

