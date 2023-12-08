import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
datapath = './clean_data_TL/windowed_data/'
window_size = '6'
season = '2021-22/'

team_name_df = pd.read_csv('./clean_data_TL/master_team_list.csv')
team_name_df = team_name_df[team_name_df.season == '2020-21']
fixture_df = pd.read_csv('./fixtures/full_fixture_difficulty.csv')
fixture_df = fixture_df[fixture_df.season == '21_22']
fixture_df = fixture_df.merge(team_name_df, how='left', left_on='team_a', right_on='team')[['team_a_difficulty', 'team_h_difficulty', 'team_h', 'team_name']]
fixture_df.rename(columns={'team_name':'away_team_name'}, inplace=True)
fixture_df = fixture_df.merge(team_name_df, how='left', left_on='team_h', right_on='team')[['team_a_difficulty', 'team_h_difficulty', 'away_team_name', 'team_name']]
fixture_df.rename(columns={'team_name':'home_team_name'}, inplace=True)

def attach_difficulty(row):
    try:
        if row['was_home']:
            team = fixture_df[(fixture_df.home_team_name == row.team) & (fixture_df.away_team_name == row.opponent_team)].reset_index(drop=True).loc[0, 'team_h_difficulty']
            opponent = fixture_df[(fixture_df.home_team_name == row.team) & (fixture_df.away_team_name == row.opponent_team)].reset_index(drop=True).loc[0, 'team_a_difficulty']
            return team - opponent
        else:
            team = fixture_df[(fixture_df.home_team_name == row.opponent_team) & (fixture_df.away_team_name == row.team)].reset_index(drop=True).loc[0, 'team_a_difficulty']
            opponent = fixture_df[(fixture_df.home_team_name == row.opponent_team) & (fixture_df.away_team_name == row.team)].reset_index(drop=True).loc[0, 'team_h_difficulty']
            return team - opponent
    except KeyError:
        return 0

def read_data(pos, datapath, window_size, season, minimum_minutes=0):
    path = os.path.join(datapath, season)
    df = pd.read_csv(path + pos + window_size + '.csv')
    df = df[df.minutes >= minimum_minutes]
    df['difficulty_gap'] = df.apply(lambda r: attach_difficulty(r), axis=1, result_type='expand')
    df.drop(['team', 'opponent_team'], axis=1, inplace=True)
    
    df['total_points_bin'] = pd.cut(df['total_points'], 5, labels=False)
    return df

dfs = [read_data(p, datapath, window_size, season, minimum_minutes=1) for p in ['GK_', 'DEF_', 'FWD_', 'MID_']]
try:
    os.makedirs("clean_data_TL/windowed_diff_data")
except FileExistsError:
    pass
for i in range(4):
    dfs[i].to_csv(os.path.join("clean_data_TL/windowed_diff_data","{}_{}.csv".format(['GK', 'DEF', 'FWD', 'MID'][i], window_size)), index=False)
