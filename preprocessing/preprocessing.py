import pandas as pd
import generic_stats
import utils
import results

# 1. Load and prep raw data
df = utils.read_data()
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
df = df.sort_values(by='Date').reset_index(drop=True)

df_train = pd.DataFrame()
df_train['Date'] = df['Date']
df_train['HomeTeam'] = df['HomeTeam']
df_train['AwayTeam'] = df['AwayTeam']

window = 5

# =========================================================
# FEATURE GENERATION (SPECIFIC & GENERAL)
# =========================================================

# Helper lists to loop through to keep code clean
# Format: (Stat Name, Home Column, Away Column)
stats_to_calculate = [
    ('Goals_Pro', 'FTHG', 'FTAG'),
    ('HT_Goals_Pro', 'HTHG', 'HTAG'),
    ('HT_Goals_Against', 'HTAG', 'HTHG'),
    ('Goals_Suffered', 'FTAG', 'FTHG'), # Full time conceded
    ('Corners_Pro', 'HC', 'AC'),
    ('Corners_Against', 'AC', 'HC'),
    ('Fouls_Pro', 'HF', 'AF'),
    ('Fouls_Against', 'AF', 'HF'),
    ('Offsides_Pro', 'HO', 'AO'),
    ('Offsides_Against', 'AO', 'HO'),
    ('Yellows_Pro', 'HY', 'AY'),
    ('Yellows_Against', 'AY', 'HY'),
    ('Reds_Pro', 'HR', 'AR'),
    ('Reds_Against', 'AR', 'HR')
]

for stat_name, h_col, a_col in stats_to_calculate:
    # 1. Specific Form (Home playing Home / Away playing Away)
    df_train[f'Home_Spec_{stat_name}'] = generic_stats.get_home_rolling_avg(df, h_col, window)
    df_train[f'Away_Spec_{stat_name}'] = generic_stats.get_away_rolling_avg(df, a_col, window)
    
    # 2. General Form (Overall)
    df_train[f'Home_Gen_{stat_name}'] = generic_stats.get_general_rolling_avg(df, 'Home', h_col, a_col, window)
    df_train[f'Away_Gen_{stat_name}'] = generic_stats.get_general_rolling_avg(df, 'Away', h_col, a_col, window)

# --- Results Features ---
df_train['Home_Spec_Points'] = results.get_home_team_points_avg(df, window)
df_train['Away_Spec_Points'] = results.get_away_team_points_avg(df, window)
# Assuming you added get_general_points to results.py too:
# df_train['Home_Gen_Points'] = results.get_home_team_general_points(df, window)
# df_train['Away_Gen_Points'] = results.get_away_team_general_points(df, window)

# ---------------------------------------------------------
# TARGET AND EXPORT
# ---------------------------------------------------------
df_train['Target'] = results.get_target_class(df)

df_train = df_train.dropna().reset_index(drop=True)
df_train.to_csv('../processed-data/training_data.csv', index=False)