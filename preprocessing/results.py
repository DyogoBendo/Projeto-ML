import pandas as pd
import numpy as np

# ==========================================
# TARGET VARIABLE
# ==========================================

def get_target_class(df):
    """Converts FTR (H, D, A) into integers (0, 1, 2) for ML training."""
    mapping = {'H': 0, 'D': 1, 'A': 2}
    return df['FTR'].map(mapping)

# ==========================================
# HISTORICAL POINTS FORM (Overall Momentum)
# ==========================================

def _build_points_timeline(df):
    """Calculates points earned per match to build a general form timeline."""
    
    # Home Team Points
    home = df[['Date', 'HomeTeam', 'FTR']].copy()
    home.columns = ['Date', 'Team', 'Result']
    home['Points'] = np.where(home['Result'] == 'H', 1, np.where(home['Result'] == 'D', 0, -1))
    
    # Away Team Points
    away = df[['Date', 'AwayTeam', 'FTR']].copy()
    away.columns = ['Date', 'Team', 'Result']
    away['Points'] = np.where(away['Result'] == 'A', 1, np.where(away['Result'] == 'D', 0, -1))
    
    # Drop the text result, keep just points
    home = home.drop('Result', axis=1)
    away = away.drop('Result', axis=1)
    
    return pd.concat([home, away]).sort_values('Date').reset_index(drop=True)

def get_home_team_points_avg(df, window=5):
    """Average points accumulated per game by the Home team over their last N matches."""
    timeline = _build_points_timeline(df)
    timeline['Points_Avg'] = timeline.groupby('Team')['Points'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    merged = df.merge(timeline[['Date', 'Team', 'Points_Avg']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    return merged['Points_Avg']

def get_away_team_points_avg(df, window=5):
    """Average points accumulated per game by the Away team over their last N matches."""
    timeline = _build_points_timeline(df)
    timeline['Points_Avg'] = timeline.groupby('Team')['Points'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    merged = df.merge(timeline[['Date', 'Team', 'Points_Avg']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    return merged['Points_Avg']