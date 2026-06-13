import pandas as pd
import numpy as np

def get_specific_rolling_stats(df, team_col, stat_col, stat_prefix, sma_windows=[3, 5, 7, 10], ema_windows=[3, 5, 7, 10], min_matches=10):
    """
    Calculates specific form (Home playing at Home, or Away playing Away).
    The min_matches counter strictly counts games played in this specific context.
    """
    # 1. Removed 'Season' from the column selection
    temp_df = df[['Date', team_col, stat_col]].copy()
    
    # 2. Shift strictly within the specific team (Season isolation is now handled upstream)
    temp_df['Shifted'] = temp_df.groupby(team_col)[stat_col].shift(1)
    
    # 3. This counter ticks up when they play at this specific venue
    temp_df['Prev_Count'] = temp_df.groupby(team_col).cumcount()

    result_df = pd.DataFrame(index=temp_df.index)

    # Calculate SMAs and EMAs for all requested windows
    for w in sma_windows:
        result_df[f'{team_col}_Spec_SMA_{w}_{stat_prefix}'] = temp_df.groupby(team_col)['Shifted'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    for w in ema_windows:
        result_df[f'{team_col}_Spec_EMA_{w}_{stat_prefix}'] = temp_df.groupby(team_col)['Shifted'].transform(
            lambda x: x.ewm(span=w, min_periods=1).mean()
        )

    # Apply the strict purge rule
    cols_to_mask = [c for c in result_df.columns if 'SMA_' in c or 'EMA_' in c]
    result_df.loc[temp_df['Prev_Count'] < min_matches, cols_to_mask] = np.nan

    return result_df[cols_to_mask]

def get_advanced_rolling_stats(df, target_team, stat_prefix, home_stat_col, away_stat_col, sma_windows=[3, 5, 7, 10], ema_windows=[3, 5, 7, 10], min_matches=10):
    """
    Calculates General Form: SMAs and EMAs across all matches (Home + Away).
    """
    # 1. Isolate Home matches (Removed 'Season')
    home_df = df[['Date', 'HomeTeam', home_stat_col]].copy()
    home_df.columns = ['Date', 'Team', 'Stat']
    home_df['Original_Index'] = home_df.index
    home_df['Team_Type'] = 'Home'

    # 2. Isolate Away matches (Removed 'Season')
    away_df = df[['Date', 'AwayTeam', away_stat_col]].copy()
    away_df.columns = ['Date', 'Team', 'Stat']
    away_df['Original_Index'] = away_df.index
    away_df['Team_Type'] = 'Away'

    # 3. Combine into a unified timeline
    timeline = pd.concat([home_df, away_df]).sort_values('Date')

    # 4. Shift the stat strictly within the Team group
    timeline['Shifted_Stat'] = timeline.groupby('Team')['Stat'].shift(1)
    
    # 5. Count how many previous matches have been played
    timeline['Previous_Matches_Count'] = timeline.groupby('Team').cumcount()

    # 6. Initialize the results dataframe
    result_df = pd.DataFrame(index=timeline.index)
    result_df['Original_Index'] = timeline['Original_Index']
    result_df['Team_Type'] = timeline['Team_Type']
    result_df['Previous_Matches_Count'] = timeline['Previous_Matches_Count']

    # 7. Calculate all SMAs and EMAs dynamically
    for w in sma_windows:
        # Simple Moving Average
        result_df[f'{target_team}_SMA_{w}_{stat_prefix}'] = timeline.groupby('Team')['Shifted_Stat'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    for w in ema_windows:
        # Exponential Moving Average
        result_df[f'{target_team}_EMA_{w}_{stat_prefix}'] = timeline.groupby('Team')['Shifted_Stat'].transform(
            lambda x: x.ewm(span=w, min_periods=1).mean()
        )

    # 8. Filter back to just the Home or Away rows we are calculating for
    result_df = result_df[result_df['Team_Type'] == target_team]

    # 9. ENFORCE THE 10 MATCH RULE
    cols_to_mask = [col for col in result_df.columns if 'SMA_' in col or 'EMA_' in col]
    result_df.loc[result_df['Previous_Matches_Count'] < min_matches, cols_to_mask] = np.nan

    # 10. Re-align perfectly with the original dataframe
    result_df = result_df.set_index('Original_Index').sort_index()

    return result_df[cols_to_mask]