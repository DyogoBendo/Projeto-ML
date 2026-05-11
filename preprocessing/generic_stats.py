import pandas as pd
import numpy as np

def get_specific_rolling_stats(df, team_col, stat_col, stat_prefix, min_matches=10):
    """
    Calculates specific form (Home playing at Home, or Away playing Away).
    The min_matches counter strictly counts games played in this specific context.
    """
    temp_df = df[['Date', 'Season', team_col, stat_col]].copy()
    
    # Shift strictly within the specific team and season
    temp_df['Shifted'] = temp_df.groupby([team_col, 'Season'])[stat_col].shift(1)
    
    # This counter now only ticks up when they play at this specific venue
    temp_df['Prev_Count'] = temp_df.groupby([team_col, 'Season']).cumcount()

    result_df = pd.DataFrame(index=temp_df.index)

    
    result_df[f'{team_col}_Spec_SMA_{10}_{stat_prefix}'] = temp_df.groupby([team_col, 'Season'])['Shifted'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    result_df[f'{team_col}_Spec_EMA_{5}_{stat_prefix}'] = temp_df.groupby([team_col, 'Season'])['Shifted'].transform(
        lambda x: x.ewm(span=5, min_periods=1).mean()
    )

    # Apply the strict purge rule
    cols_to_mask = [c for c in result_df.columns if 'SMA_' in c or 'EMA_' in c]
    result_df.loc[temp_df['Prev_Count'] < min_matches, cols_to_mask] = np.nan

    return result_df[cols_to_mask]

def get_advanced_rolling_stats(df, target_team, stat_prefix, home_stat_col, away_stat_col, min_matches=10):
    """
    Calculates multiple SMA and EMA windows, strictly contained within individual seasons.
    Forces all outputs to NaN until the team has played at least 'min_matches' in that season.
    """
    # 1. Isolate Home matches
    home_df = df[['Date', 'Season', 'HomeTeam', home_stat_col]].copy()
    home_df.columns = ['Date', 'Season', 'Team', 'Stat']
    home_df['Original_Index'] = home_df.index
    home_df['Team_Type'] = 'Home'

    # 2. Isolate Away matches
    away_df = df[['Date', 'Season', 'AwayTeam', away_stat_col]].copy()
    away_df.columns = ['Date', 'Season', 'Team', 'Stat']
    away_df['Original_Index'] = away_df.index
    away_df['Team_Type'] = 'Away'

    # 3. Combine into a unified timeline
    timeline = pd.concat([home_df, away_df]).sort_values('Date')

    # 4. Shift the stat strictly within the Team/Season group!
    # This ensures Match 1 of Season 03/04 does NOT get the stat from Match 38 of Season 02/03
    timeline['Shifted_Stat'] = timeline.groupby(['Team', 'Season'])['Stat'].shift(1)
    
    # 5. Count how many previous matches have been played in THIS season
    timeline['Previous_Matches_Count'] = timeline.groupby(['Team', 'Season']).cumcount()

    # 6. Initialize the results dataframe
    result_df = pd.DataFrame(index=timeline.index)
    result_df['Original_Index'] = timeline['Original_Index']
    result_df['Team_Type'] = timeline['Team_Type']
    result_df['Previous_Matches_Count'] = timeline['Previous_Matches_Count']
    
    # Simple Moving Average
    result_df[f'{target_team}_SMA_{10}_{stat_prefix}'] = timeline.groupby(['Team', 'Season'])['Shifted_Stat'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    # Exponential Moving Average
    result_df[f'{target_team}_EMA_{5}_{stat_prefix}'] = timeline.groupby(['Team', 'Season'])['Shifted_Stat'].transform(
        lambda x: x.ewm(span=5, min_periods=1).mean()
    )
    
    # 8. Filter back to just the Home or Away rows we are calculating for
    result_df = result_df[result_df['Team_Type'] == target_team]

    # 9. ENFORCE THE 10 MATCH RULE: 
    # If the team has fewer than 10 previous matches in this season, overwrite everything with NaN
    cols_to_mask = [col for col in result_df.columns if 'SMA_' in col or 'EMA_' in col]
    result_df.loc[result_df['Previous_Matches_Count'] < min_matches, cols_to_mask] = np.nan

    # 10. Re-align perfectly with the original dataframe
    result_df = result_df.set_index('Original_Index').sort_index()

    return result_df[cols_to_mask]