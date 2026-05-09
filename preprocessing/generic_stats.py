import pandas as pd

# ---------------------------------------------------------
# 1. CONTEXT-SPECIFIC FORM (Home matches only / Away matches only)
# ---------------------------------------------------------
def get_home_rolling_avg(df, stat_col, window=5):
    return df.groupby('HomeTeam')[stat_col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

def get_away_rolling_avg(df, stat_col, window=5):
    return df.groupby('AwayTeam')[stat_col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

# ---------------------------------------------------------
# 2. GENERAL OVERALL FORM (Home and Away combined)
# ---------------------------------------------------------
def get_general_rolling_avg(df, target_team, home_stat_col, away_stat_col, window=5):
    """
    Calculates the overall rolling average across a team's entire timeline.
    
    Parameters:
    - target_team: 'Home' (if attaching to df['HomeTeam']) or 'Away' (if attaching to df['AwayTeam'])
    - home_stat_col: The column name when the team is playing AT HOME.
    - away_stat_col: The column name when the team is playing AWAY.
    """
    # 1. Isolate when teams play at home, keeping the original index
    home_df = df[['Date', 'HomeTeam', home_stat_col]].copy()
    home_df.columns = ['Date', 'Team', 'Stat']
    home_df['Original_Index'] = home_df.index
    home_df['Team_Type'] = 'Home'

    # 2. Isolate when teams play away, keeping the original index
    away_df = df[['Date', 'AwayTeam', away_stat_col]].copy()
    away_df.columns = ['Date', 'Team', 'Stat']
    away_df['Original_Index'] = away_df.index
    away_df['Team_Type'] = 'Away'

    # 3. Combine both into a single timeline and sort chronologically
    timeline = pd.concat([home_df, away_df]).sort_values('Date')

    # 4. Calculate the rolling average per team
    timeline['Rolling_Avg'] = timeline.groupby('Team')['Stat'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    # 5. Filter the timeline back to the exact team type we are calculating for
    result = timeline[timeline['Team_Type'] == target_team]
    
    # 6. Re-align with the original dataframe using the index
    result = result.set_index('Original_Index').sort_index()

    return result['Rolling_Avg']