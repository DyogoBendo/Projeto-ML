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
df_train['Season'] = df['Season']

# =========================================================
# CREATE PERSPECTIVE POINTS (Win=1, Draw=0, Loss=-1)
# =========================================================

# For the Home Team: H is a win (1), A is a loss (-1)
df['Home_Pts'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})

# For the Away Team: A is a win (1), H is a loss (-1)
df['Away_Pts'] = df['FTR'].map({'H': -1, 'D': 0, 'A': 1})

# =========================================================
# FEATURE GENERATION 
# =========================================================
print("Calculating General and Specific features...")
new_feature_blocks = []

# --- A. CALCULATE MATCH FORM (RESULTS) ---
# 1. General Form (Stitching Home and Away timelines together - Requires 10 overall games)
gen_home_form = generic_stats.get_advanced_rolling_stats(df, 'Home', 'Gen_Form', 'Home_Pts', 'Away_Pts', min_matches=10)
gen_away_form = generic_stats.get_advanced_rolling_stats(df, 'Away', 'Gen_Form', 'Home_Pts', 'Away_Pts', min_matches=10)

# 2. Specific Form (Requires 10 Home games for Home, 10 Away games for Away)
spec_home_form = generic_stats.get_specific_rolling_stats(df, 'HomeTeam', 'Home_Pts', 'Form', min_matches=10)
spec_away_form = generic_stats.get_specific_rolling_stats(df, 'AwayTeam', 'Away_Pts', 'Form', min_matches=10)

new_feature_blocks.extend([gen_home_form, gen_away_form, spec_home_form, spec_away_form])

# --- B. CALCULATE STATISTICAL FORM (GOALS, CORNERS, ETC.) ---
stats_to_calculate = [
    ('Goals_Pro', 'FTHG', 'FTAG'),
    ('Goals_Suffered', 'FTAG', 'FTHG'), 
    ('Corners_Pro', 'HC', 'AC'),
    ('Corners_Against', 'AC', 'HC'),
    ('Fouls_Pro', 'HF', 'AF'),
    ('Fouls_Against', 'AF', 'HF'),       
    ('Yellows_Pro', 'HY', 'AY'),
    ('Yellows_Against', 'AY', 'HY'),     
    ('ShotsTarget_Pro', 'HST', 'AST'), 
    ('ShotsTarget_Against', 'AST', 'HST'),       
    ('Shots_Pro', 'HS', 'AS'), 
    ('Shots_Against', 'AS', 'HS'),       
]

for stat_prefix, h_col, a_col in stats_to_calculate:
    # 1. GENERAL STATS (Total season momentum)
    h_gen = generic_stats.get_advanced_rolling_stats(df, 'Home', f'Gen_{stat_prefix}', h_col, a_col, min_matches=10)
    a_gen = generic_stats.get_advanced_rolling_stats(df, 'Away', f'Gen_{stat_prefix}', h_col, a_col, min_matches=10)
    
    # 2. SPECIFIC STATS (Stadium specific momentum)
    h_spec = generic_stats.get_specific_rolling_stats(df, 'HomeTeam', h_col, stat_prefix, min_matches=10)
    a_spec = generic_stats.get_specific_rolling_stats(df, 'AwayTeam', a_col, stat_prefix, min_matches=10)
    
    new_feature_blocks.extend([h_gen, a_gen, h_spec, a_spec])

# Concatenate everything at once
df_train = pd.concat([df_train] + new_feature_blocks, axis=1)
df_train['FTR'] = results.get_target_class(df)

# =========================================================
# THE PURGE
# =========================================================
print(f"Rows before purge: {len(df_train)}")
df_train = df_train.dropna().reset_index(drop=True)
print(f"Rows after purge: {len(df_train)}")

df_train.to_csv('../processed-data/training_data_advanced_reduced.csv', index=False)
print("Advanced Training Data Exported!")