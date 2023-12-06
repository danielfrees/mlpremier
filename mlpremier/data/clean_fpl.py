"""
Clean scraped FPL Data for Modeling. General purpose cleaning. 

See combine_csv.py for Traditional ML model data prep and cnn/preprocess.py for
CNN model data prep.
"""


import os
import pandas as pd
from fuzzywuzzy import process
import warnings
from urllib.parse import unquote   #for cleanup of bad folder player names from encoded URLS
import re
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
import re

# ================== HELPERS ====================
def decode_url(url):
    # Define a regular expression pattern to match URL-encoded sequences
    pattern = re.compile('%[0-9a-fA-F]{2}')

    # Use a lambda function to replace each match with its decoded equivalent
    decoded_url = pattern.sub(lambda x: unquote(x.group(0)), url)

    return decoded_url
# ================== END HELPERS ====================

# ========== DATA CLEANING ===========
def create_player_meta_csv(season:str, 
                           save_dir:str=os.path.join('clean_data', 'meta')):
    """ 
    Create a csv of player metadata.

    :param str season: Which season of EPL data to process metadata for. Should 
        follow the format 20XX-X(X+1).
    :param str save_dir: path within repo to desired data folder for the metadata
      /path/to/data

    :returns: Nothing
    :rtype: None
    """

    print("======= Creating CSV of Player Metadata ========")
    RAW_DATA_DIR = 'raw_data'
    meta_data_list = []

    # Check if raw_data/season/gws exists
    season_dir = os.path.join(RAW_DATA_DIR, season, 'gws')
    if not os.path.exists(season_dir):
        print(f"No data found for {season}/gws. Exiting.")
        return

    # Iterate through all CSV files in gws folder
    for filename in tqdm(os.listdir(season_dir)):
        if filename.endswith(".csv"):
            csv_path = os.path.join(season_dir, filename)
            try:
                df = pd.read_csv(csv_path)
                df = df.loc[:,['name', 'position', 'team']]
                meta_data_list.append(df)
            except UnicodeDecodeError as e:
                df = pd.read_csv(csv_path, encoding='latin-1')
                df = df.loc[:,['name', 'position', 'team']]
                meta_data_list.append(df)
            except: 
                print(df)
            

    # Concatenate all gw metadata (we do this because players may transfer
    # in and out of the league)
    meta_data_df = pd.concat(meta_data_list, axis=0, ignore_index=True)
    meta_data_df = meta_data_df.drop_duplicates(subset=['name'], keep='first')
    
    # Check if 'name', 'position', 'team' are consistent for each 'name'
    #assert meta_data_df.groupby('name')['position'].nunique().eq(1).all()
    #assert meta_data_df.groupby('name')['team'].nunique().eq(1).all()

    # Write the cleaned dataframe to a CSV file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'player_meta_{season}.csv')
    meta_data_df.to_csv(save_path, index=False)
    print(f"Player metadata saved to {save_path}")

    return

def reorder_and_limit_gwdata_cols(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Reorders and limits columns of the provided dataframe to the following 
    format:

    cols = name, position, team, opponent_team, minutes, goals_scored, assists, 
    goals_conceded, clean_sheets, bps, yellow_cards, red_cards, own_goals, 
    saves, penalties_missed, penalties_saved, ict_index, influence, creativity, 
    threat, was_home, total_points 
    """
    df = df.copy()
    # Define the desired column order
    desired_cols = ['name', 'position', 'team', 'opponent_team',  'minutes', 
                    'goals_scored', 'assists', 'goals_conceded', 'clean_sheets', 
                    'bps', 'yellow_cards', 'red_cards', 'own_goals', 'saves', 
                    'penalties_missed', 'penalties_saved', 'ict_index', 
                    'influence', 'creativity', 'threat', 'was_home', 
                    'total_points']

    # Reorder and limit the columns
    df = df[desired_cols]

    return df

def add_opponent_team_and_matchup_difficulty(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """ 
    Fills in the actual opponent team name from the team ID.

    The `master_team_list.csv` file must be downloaded (see `get_master_team_list`
    in `scrape_fpl_data`).
    """
    df = df.copy() 
    
    CLEAN_DATA_DIR = 'clean_data'
    MASTER_TEAM_LIST_PATH = os.path.join(CLEAN_DATA_DIR, 'master_team_list.csv')

    # Load the master_team_list.csv
    master_team_list = pd.read_csv(MASTER_TEAM_LIST_PATH)

    # Create a mapping dictionary using both 'season' and 'team' columns
    mapping_df = master_team_list[(master_team_list['season'] == season)][['team', 'team_name', 'team_difficulty']]
    #print(mapping_df)
    team_mapping = dict(list(zip(mapping_df['team'], mapping_df['team_name'])))
    #print(team_mapping)
    diff_mapping = dict(list(zip(mapping_df['team_name'], mapping_df['team_difficulty'])))
    #print(diff_mapping)
    # Map the opponent_team column using the created mapping
    df['opponent_team'] = df['opponent_team'].apply(lambda team: team_mapping[team])

    df['team_difficulty'] = df['team'].apply(lambda team: diff_mapping[team])
    df['opponent_difficulty'] = df['opponent_team'].apply(lambda team: diff_mapping[team])
    df['matchup_difficulty'] = df['team_difficulty'] - df['opponent_difficulty']
    
    return df


def clean_player_data(season: str, 
                    save_dir: str = os.path.join('clean_data'),
                    verbose:bool = False):
    """
    Appends player metadata into player gw-by-gw CSVs. Also cleans player data 
    cols.

    :param str season: Which season of EPL data to process metadata for. Should 
        follow the format 20XX-X(X+1).
    :param str save_dir: path within repo to desired data folder for the metadata
        /path/to/data

    :returns: Nothing
    :rtype: None
    """
    print("======= Cleaning Player Data ===========")
    RAW_DATA_DIR = 'raw_data'
    PLAYER_META_CSV_PATH = os.path.join('clean_data', 'meta', f'player_meta_{season}.csv')

    # Load player metadata CSV
    player_meta_df = pd.read_csv(PLAYER_META_CSV_PATH)

    # Iterate through all player_name_folders in the raw_data/players directory
    player_dirs = [f for f in os.listdir(os.path.join(RAW_DATA_DIR, season, 'players')) 
                   if os.path.isdir(os.path.join(RAW_DATA_DIR, season, 'players', f))]
    
    used_names = []

    for player_folder in tqdm(player_dirs):
        player_name = decode_url(player_folder)
        
        if verbose:
            print(f"Appending metadata via fuzzy match for player: \n{player_folder}\n")
        player_folder_path = os.path.join(RAW_DATA_DIR, season, 'players', player_folder)
        player_csv_path = os.path.join(player_folder_path, 'gw.csv')

        # Load player gw CSV
        if verbose:
            print(f"Reading player data for {player_csv_path}")
        player_gw_df = pd.read_csv(player_csv_path)

        # Fuzzy match based on the closest name in PLAYER_META_CSV
        # with player name extracted via raw data player directory name. 
        # column name to match on is 'name'.
        matches = process.extractOne(player_name.replace('_', ' '), 
                                     player_meta_df['name'])
        top_match = matches[0]

        if top_match in used_names:
            warnings.warn(("Attempted to assign an already matched" 
                          f"player name. Player Folder: {player_folder}." 
                          f"Top Match: {matches[0]}. Skipping this player."))
            continue
        else:
            player_metadata = player_meta_df[player_meta_df['name'] == matches[0]]

            # Append player_metadata columns to player_gw_df
            for col in player_metadata.columns:
                player_gw_df[col] = player_metadata[col].values[0]

            #Clean up player_gw_df columns
            player_gw_df = reorder_and_limit_gwdata_cols(player_gw_df)

            #Fill in Opponent Team
            player_gw_df = add_opponent_team_and_matchup_difficulty(player_gw_df, season)

            # Save the updated player_gw_df to the appropriate path
            full_save_dir = os.path.join(save_dir, season, 'players', player_name)
            if not os.path.exists(full_save_dir):
                os.makedirs(full_save_dir, exist_ok=True)


            if verbose: 
                print(f"Saving appended player data to: {full_save_dir}")
            player_gw_df.to_csv(os.path.join(full_save_dir, 'gw.csv'), 
                                index=False)
            
            # Mark the player name as used
            used_names.append(top_match)

    return

def organize_data_by_pos(data_dir: str,
                         verbose: bool = False):
    """
    Organizes player data by position, creating new folders for GKP, DEF, MID, FWD.

    :param str data_dir: Path to the top-level directory containing season and player data.

    :returns: None
    :rtype: None
    """

    print("======= Organizing Clean Data by Position =======")
    POSITIONS = ['GK', 'DEF', 'MID', 'FWD']

    unique_positions = set()

    # Iterate through all seasons (top-level folders)
    top_level_folders = os.listdir(data_dir)
    season_pattern = re.compile(r'^20\d{2}-\d{2}$')
    season_folders = [folder for folder in top_level_folders if season_pattern.match(folder)]
    for season_folder in season_folders:
        season_path = os.path.join(data_dir, season_folder)

        if os.path.isdir(season_path):
            # Iterate through all players within each season
            players_path = os.path.join(season_path, 'players')
            for player_folder in tqdm(os.listdir(players_path)):
                player_path = os.path.join(season_path, 'players', player_folder)

                if os.path.isdir(player_path):
                    # Read in the .csv file in the player's folder
                    player_csv = os.path.join(player_path, "gw.csv")

                    if os.path.isfile(player_csv):
                        player_data = pd.read_csv(player_csv)
                        position = player_data.loc[0, 'position']

                        # Place each player file back into position/season/playername/gw.csv
                        player_pos_folder = os.path.join(season_path, position)
                        os.makedirs(os.path.join(player_pos_folder, player_folder), exist_ok=True)
                        new_player_csv = os.path.join(player_pos_folder, player_folder, "gw.csv")

                        if verbose: 
                            print(f'Copying {player_csv} into {new_player_csv}')
                        player_data.to_csv(new_player_csv, index=False)

    return

MASTER_TEAM_LIST_PATH = os.path.join('clean_data', 'master_team_list.csv')
FIXTURE_PATH = os.path.join('fixtures', 'full_fixture_difficulty.csv')

def append_team_difficulty():
    """ 
    Add team difficulty to master team list.
    """
    # Load DataFrames
    team_list_df = pd.read_csv(MASTER_TEAM_LIST_PATH)
    fixture_df = pd.read_csv(FIXTURE_PATH)

    # Remove '20' prefix from team_list_df['season']
    fixture_df['season'] = fixture_df['season'].apply(lambda x: "20" + x.replace('_', '-'))

    merged_df = pd.merge(team_list_df, fixture_df, left_on=['season', 'team'], right_on=['season', 'team_a'], how='left')
    merged_df = merged_df.drop(columns=['team_a', 'team_h', 'team_h_difficulty'])
    merged_df = merged_df.rename(columns={'team_a_difficulty': 'team_difficulty'})

    merged_df = merged_df.drop_duplicates()
    merged_df.tail(20)

    merged_df.to_csv(MASTER_TEAM_LIST_PATH, index=False)

    return


def main():
    """
    Clean and organize raw FPL Data.
    """
    parser = argparse.ArgumentParser(description='General preprocessing and Cleaning of FPL Data.')
    parser.add_argument('-s', '--season', type=str, help='Comma-separated list of seasons')
    parser.add_argument('-o', '--organize', action='store_true', default=False, help='Organize by positions')
    parser.add_argument('-d', '--difficulty', action='store_true', default=False, help='Add team difficulties to master team list.')

    args = parser.parse_args()

    # Load .env file from dir where script is called
    dotenv_path = os.path.join(os.getcwd(), '.env')
    dotenv_path = os.path.join(os.getcwd(), '.env')
    load_dotenv(dotenv_path)

    # Access the value of the 'season' option
    if args.season:
        seasons = args.season.split(',')
        for season in seasons:
            create_player_meta_csv(season)
            clean_player_data(season)
    else:    
        print("No seasons requested.")
    
    if args.organize:
        organize_data_by_pos('clean_data', verbose=True)
    if args.difficulty:
        append_team_difficulty()

    print("Done. Quitting.")

if __name__ == '__main__':
    main()
