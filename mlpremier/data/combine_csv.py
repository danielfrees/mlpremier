import pandas as pd
import numpy as np
import os

columns_csv = ['xP', 'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
               'element', 'goals_conceded', 'goals_scored', 'ict_index', 'influence',
               'minutes', 'own_goals', 'penalties_missed',
               'penalties_saved', 'red_cards', 'saves', 'team_a_score', 'team_h_score',
               'threat', 'total_points', 'value', 'yellow_cards', 'was_home', 'opponent_team',
               'Target_Output']


def combine_csv_position(position, window):
    """Combine the CSVs for a paticular position by averaging the last window games for metrics"""
    player_path = os.path.join(position)
    frames = []
    for filename in os.listdir(player_path):
        filepath = os.path.join(player_path, filename)
        dataset = pd.read_csv(
            filepath, sep=",")
        drop_columns = ['name', 'GW', 'transfers_in', 'transfers_out', 'transfers_balance', 'selected',
                        'round', 'kickoff_time', 'fixture', 'position', 'team']
        dataset = dataset.drop(
            columns=drop_columns)
        if len(dataset) >= window:
            temp_df = pd.DataFrame(columns=columns_csv)
            for col in columns_csv[:-3]:
                rl_col = dataset[col].rolling(
                    window, min_periods=window).mean()
                temp_df[col] = rl_col
            for col in columns_csv[-3:]:
                tail_col = dataset[col].tail(-window)
                temp_df[col] = tail_col
            temp_df = temp_df.dropna()
            frames.append(temp_df)
    combined_df = pd.concat(frames)
    combined_df.reset_index(drop=True)
    return combined_df


def main():
    windows = [3, 6, 9]
    for w in windows:
        pos = [f for f in os.listdir() if os.path.isdir(f)]
        for p in pos:
            if p != "raw":
                df = combine_csv_position(p, w)
                df.to_csv("{}_{}.csv".format(p, w))


if __name__ == '__main__':
    main()
