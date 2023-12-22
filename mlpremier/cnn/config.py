"""
Global configs for CNN experimentation
"""

STANDARD_NUM_FEATURES = ['minutes', 'goals_scored', 'assists', 'goals_conceded',
                          'clean_sheets', 'bps', 'yellow_cards', 'red_cards', 
                          'own_goals', 'saves', 'penalties_missed', 'penalties_saved',
                          'ict_index', 'total_points']
STANDARD_CAT_FEATURES = []

NUM_FEATURES_DICT = {
        'GK': {
            'ptsonly': ['total_points'],
            'pts_ict': ['total_points', 'ict_index'],
            'small': ['total_points', 'saves', 'clean_sheets', 'red_cards'],
            'medium': ['total_points', 'minutes', 'saves', 'bps', 'goals_conceded', 'red_cards'],
            'large': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'yellow_cards', 'red_cards']
        },
        'DEF': {
            'ptsonly': ['total_points'],
            'pts_ict': ['total_points', 'ict_index'],
            'small': ['total_points', 'clean_sheets', 'red_cards'],
            'medium': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'red_cards'],
            'large': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'yellow_cards', 'red_cards', 'own_goals', 'penalties_saved']
        },
        'MID': {
            'ptsonly': ['total_points'],
            'pts_ict': ['total_points', 'ict_index'],
            'small': ['total_points', 'goals_scored', 'assists', 'red_cards'],
            'medium': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'red_cards'],
            'large': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'yellow_cards', 'red_cards']
        },
        'FWD': {
            'ptsonly': ['total_points'],
            'pts_ict': ['total_points', 'ict_index'],
            'small': ['total_points', 'goals_scored', 'assists', 'red_cards'],
            'medium': ['total_points', 'minutes', 'goals_scored', 'assists', 'bps', 'red_cards'],
            'large': ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'bps', 'yellow_cards', 'red_cards']
        }
    }