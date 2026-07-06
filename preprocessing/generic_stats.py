import pandas as pd
import numpy as np

def get_specific_rolling_stats(df, team_col, stat_col, stat_prefix, sma_windows=[3, 5, 7, 10], ema_windows=[3, 5, 7, 10], min_matches=10):
    """
    Calculates specific form (Home playing at Home, or Away playing Away).
    The min_matches counter strictly counts games played in this specific context.
    If the stat_col represents points (1, 0, -1), it calculates the count of Wins, Draws, Losses and Streaks.
    """
    # 1. Isolate relevant columns
    temp_df = df[['Date', team_col, stat_col]].copy()
    
    # 2. Shift strictly within the specific team (Season isolation is handled upstream)
    temp_df['Shifted'] = temp_df.groupby(team_col)[stat_col].shift(1)
    
    # 3. This counter ticks up when they play at this specific venue
    temp_df['Prev_Count'] = temp_df.groupby(team_col).cumcount()

    result_df = pd.DataFrame(index=temp_df.index)

    # Check if we are calculating Form (which means we should also calculate W/D/L counts and streaks)
    is_form = stat_col in ['Home_Pts', 'Away_Pts']

    # STREAKS (Calculated efficiently on the shifted timeline using cumsum blocks)
    if is_form:
        # Win Streak
        is_win = temp_df['Shifted'] == 1
        win_blocks = (~is_win).groupby(temp_df[team_col]).cumsum()
        result_df[f'{team_col}_Spec_WinStreak_{stat_prefix}'] = is_win.groupby([temp_df[team_col], win_blocks]).cumcount()
        
        # Unbeaten Streak
        is_unbeaten = temp_df['Shifted'] >= 0
        unbeaten_blocks = (~is_unbeaten).groupby(temp_df[team_col]).cumsum()
        result_df[f'{team_col}_Spec_UnbeatenStreak_{stat_prefix}'] = is_unbeaten.groupby([temp_df[team_col], unbeaten_blocks]).cumcount()

        # Losing Streak
        is_loss = temp_df['Shifted'] == -1
        loss_blocks = (~is_loss).groupby(temp_df[team_col]).cumsum()
        result_df[f'{team_col}_Spec_LossStreak_{stat_prefix}'] = is_loss.groupby([temp_df[team_col], loss_blocks]).cumcount()

    # Calculate SMAs and EMAs for all requested windows
    for w in sma_windows:
        result_df[f'{team_col}_Spec_SMA_{w}_{stat_prefix}'] = temp_df.groupby(team_col)['Shifted'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        
        # If this is a form calculation, also get the strict counts of W/D/L in this window
        if is_form:
            result_df[f'{team_col}_Spec_Wins_{w}_{stat_prefix}'] = temp_df.groupby(team_col)['Shifted'].transform(
                lambda x: (x == 1).rolling(window=w, min_periods=1).sum()
            )
            result_df[f'{team_col}_Spec_Losses_{w}_{stat_prefix}'] = temp_df.groupby(team_col)['Shifted'].transform(
                lambda x: (x == -1).rolling(window=w, min_periods=1).sum()
            )
            result_df[f'{team_col}_Spec_Draws_{w}_{stat_prefix}'] = temp_df.groupby(team_col)['Shifted'].transform(
                lambda x: (x == 0).rolling(window=w, min_periods=1).sum()
            )

    for w in ema_windows:
        result_df[f'{team_col}_Spec_EMA_{w}_{stat_prefix}'] = temp_df.groupby(team_col)['Shifted'].transform(
            lambda x: x.ewm(span=w, min_periods=1).mean()
        )

    # Apply the strict purge rule
    cols_to_mask = [c for c in result_df.columns if 'SMA_' in c or 'EMA_' in c or 'Wins_' in c or 'Losses_' in c or 'Draws_' in c or 'Streak_' in c]
    result_df.loc[temp_df['Prev_Count'] < min_matches, cols_to_mask] = np.nan

    return result_df[cols_to_mask]


def get_advanced_rolling_stats(df, target_team, stat_prefix, home_stat_col, away_stat_col, sma_windows=[3, 5, 7, 10], ema_windows=[3, 5, 7, 10], min_matches=10):
    """
    Calculates General Form: SMAs and EMAs across all matches (Home + Away).
    If the stat represents points, calculates the count of Wins, Draws, Losses and Streaks.
    """
    # 1. Isolate Home matches
    home_df = df[['Date', 'HomeTeam', home_stat_col]].copy()
    home_df.columns = ['Date', 'Team', 'Stat']
    home_df['Original_Index'] = home_df.index
    home_df['Team_Type'] = 'Home'

    # 2. Isolate Away matches
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

    is_form = home_stat_col == 'Home_Pts' and away_stat_col == 'Away_Pts'

    # STREAKS (Calculated efficiently on the unified shifted timeline using cumsum blocks)
    if is_form:
        # Win Streak
        is_win = timeline['Shifted_Stat'] == 1
        win_blocks = (~is_win).groupby(timeline['Team']).cumsum()
        result_df[f'{target_team}_WinStreak_{stat_prefix}'] = is_win.groupby([timeline['Team'], win_blocks]).cumcount()
        
        # Unbeaten Streak
        is_unbeaten = timeline['Shifted_Stat'] >= 0
        unbeaten_blocks = (~is_unbeaten).groupby(timeline['Team']).cumsum()
        result_df[f'{target_team}_UnbeatenStreak_{stat_prefix}'] = is_unbeaten.groupby([timeline['Team'], unbeaten_blocks]).cumcount()

        # Losing Streak
        is_loss = timeline['Shifted_Stat'] == -1
        loss_blocks = (~is_loss).groupby(timeline['Team']).cumsum()
        result_df[f'{target_team}_LossStreak_{stat_prefix}'] = is_loss.groupby([timeline['Team'], loss_blocks]).cumcount()

    # 7. Calculate all SMAs and EMAs dynamically
    for w in sma_windows:
        # Simple Moving Average
        result_df[f'{target_team}_SMA_{w}_{stat_prefix}'] = timeline.groupby('Team')['Shifted_Stat'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        
        # Win/Draw/Loss Counts
        if is_form:
             result_df[f'{target_team}_Wins_{w}_{stat_prefix}'] = timeline.groupby('Team')['Shifted_Stat'].transform(
                 lambda x: (x == 1).rolling(window=w, min_periods=1).sum()
             )
             result_df[f'{target_team}_Losses_{w}_{stat_prefix}'] = timeline.groupby('Team')['Shifted_Stat'].transform(
                 lambda x: (x == -1).rolling(window=w, min_periods=1).sum()
             )
             result_df[f'{target_team}_Draws_{w}_{stat_prefix}'] = timeline.groupby('Team')['Shifted_Stat'].transform(
                 lambda x: (x == 0).rolling(window=w, min_periods=1).sum()
             )

    for w in ema_windows:
        # Exponential Moving Average
        result_df[f'{target_team}_EMA_{w}_{stat_prefix}'] = timeline.groupby('Team')['Shifted_Stat'].transform(
            lambda x: x.ewm(span=w, min_periods=1).mean()
        )

    # 8. Filter back to just the Home or Away rows we are calculating for
    result_df = result_df[result_df['Team_Type'] == target_team]

    # 9. ENFORCE THE 10 MATCH RULE
    cols_to_mask = [col for col in result_df.columns if 'SMA_' in col or 'EMA_' in col or 'Wins_' in col or 'Losses_' in col or 'Draws_' in col or 'Streak_' in col]
    result_df.loc[result_df['Previous_Matches_Count'] < min_matches, cols_to_mask] = np.nan

    # 10. Re-align perfectly with the original dataframe
    result_df = result_df.set_index('Original_Index').sort_index()

    return result_df[cols_to_mask]