import pandas as pd
import numpy as np
import os

# First 6 rows here represent recent performance via a rolling average with offset 1 (to avoid leakage).
# The last row is this week's information known before the game happens.
# 'total points' is the target variable.
columns_csv = ['assists', 'clean_sheets', 'creativity', 'bps',
               'goals_conceded', 'goals_scored', 'ict_index', 'influence',
               'minutes', 'own_goals', 'penalties_missed',
               'penalties_saved', 'red_cards', 'saves',
               'threat', 'yellow_cards', 
               'total_points','kickoff_time',
               'was_home', 'team', 'opponent_team', 'name']

def combine_csv_position(position, window):
    """Combine the CSVs for a particular position by averaging the last window games for metrics"""
    frames = []
    for player_dir in os.listdir(position):
        # Avoid hidden files
        if player_dir.startswith('.'):
            continue
        player_path = os.path.join(position, player_dir)
        for filename in os.listdir(player_path):
            # Avoid hidden files
            if filename.startswith('.'):
                continue
            filepath = os.path.join(player_path, filename)

            dataset = pd.read_csv(
                filepath, sep=",")
            # Manage columns to drop
            drop_columns = ['position']
            dataset = dataset.drop(
                columns=drop_columns)

            if len(dataset) >= window:
                temp_df = pd.DataFrame(columns=columns_csv)
                # Rolling average for columns [..., total_points]
                for col in columns_csv[:-5]:
                    # Exclude the current row from calculation by setting closed='left'
                    rl_col = dataset[col].rolling(
                        window+1, min_periods=2, closed= "left").mean()
                    temp_df[col] = rl_col
                # Other columns, kept as is
                for col in columns_csv[-6:]:
                    tail_col = dataset[col].tail(-window)
                    if col == 'total_points':
                        temp_df['Target_Output'] = tail_col
                    else:
                        temp_df[col] = tail_col
                # To avoid losing as much as 20% of the data (when window=9) here, set min_periods=2 above
                temp_df = temp_df.dropna()
                frames.append(temp_df)
    combined_df = pd.concat(frames)
    combined_df.reset_index(drop=True)
    return combined_df


def main():
    # Step into the "clean_data" directory
    os.chdir("clean_data_TL")
    
    try:
        os.makedirs("windowed_data")
    except FileExistsError:
        pass
    save_dir = "windowed_data"
    windows = [6]
    for w in windows:
        season = [f for f in os.listdir() if (os.path.isdir(f) and f[:2] == '20')]
        for s in season:
            try:
                os.makedirs(os.path.join(save_dir, s))
            except FileExistsError:
                pass
            pos = [os.path.join(s, f) for f in os.listdir(s) if (os.path.isdir(os.path.join(s, f)) and f != 'players')]
            for p in pos:
                print('Combining %s, window size %i' % (p, w))
                df = combine_csv_position(p, w)
                df.to_csv(os.path.join(save_dir,"{}_{}.csv".format(p, w)), index=False)


if __name__ == '__main__':
    main()
