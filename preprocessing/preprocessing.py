import pandas as pd
import glob
import os
import generic_stats
import results
ema_windows = [3, 5, 7, 10]
sma_windows = [3, 5, 7, 10]

# =========================================================
# 1. SINGLE SEASON PROCESSING ENGINE
# =========================================================
def process_single_season(df):
    """
    Takes a raw dataframe for a SINGLE season, calculates all rolling stats,
    formats dates, applies bookmaker odds, drops identifiers, and purges NaNs.
    """
    # Ensure Date is sorted chronologically within the season
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df.sort_values(by='Date').reset_index(drop=True)

    df_season = pd.DataFrame()
    df_season['Date'] = df['Date']
    df_season['HomeTeam'] = df['HomeTeam']
    df_season['AwayTeam'] = df['AwayTeam']

    # --- CREATE PERSPECTIVE POINTS ---
    df['Home_Pts'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})
    df['Away_Pts'] = df['FTR'].map({'H': -1, 'D': 0, 'A': 1})
    
    # 1. Goal Difference
    df['Home_GD'] = df['FTHG'] - df['FTAG']
    df['Away_GD'] = df['FTAG'] - df['FTHG']

    # 2. Clean Sheets & Failed To Score (FTS)
    df['Home_Clean_Sheet'] = (df['FTAG'] == 0).astype(int)
    df['Away_Clean_Sheet'] = (df['FTHG'] == 0).astype(int)
    df['Home_FTS'] = (df['FTHG'] == 0).astype(int)
    df['Away_FTS'] = (df['FTAG'] == 0).astype(int)

    # 3. Traditional Points (3 for Win, 1 for Draw, 0 for Loss)
    df['Home_Trad_Pts'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['Away_Trad_Pts'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 3})

    # 4. Mental Resilience (Comebacks from losing at Half-Time)
    # Check if HTR exists in the dataset first to prevent crashes on older datasets
    if 'HTR' in df.columns:
        df['Home_Comeback'] = ((df['HTR'] == 'A') & (df['FTR'].isin(['H', 'D']))).astype(int)
        df['Away_Comeback'] = ((df['HTR'] == 'H') & (df['FTR'].isin(['A', 'D']))).astype(int)

    new_feature_blocks = []

    # --- A. CALCULATE MATCH FORM (RESULTS) ---
    gen_home_form = generic_stats.get_advanced_rolling_stats(df, 'Home', 'Gen_Form', 'Home_Pts', 'Away_Pts', ema_windows=ema_windows, sma_windows=sma_windows, min_matches=10)
    gen_away_form = generic_stats.get_advanced_rolling_stats(df, 'Away', 'Gen_Form', 'Home_Pts', 'Away_Pts', ema_windows=ema_windows, sma_windows=sma_windows, min_matches=10)

    spec_home_form = generic_stats.get_specific_rolling_stats(df, 'HomeTeam', 'Home_Pts', 'Form', ema_windows=ema_windows, sma_windows=sma_windows, min_matches=10)
    spec_away_form = generic_stats.get_specific_rolling_stats(df, 'AwayTeam', 'Away_Pts', 'Form', ema_windows=ema_windows, sma_windows=sma_windows, min_matches=10)

    new_feature_blocks.extend([gen_home_form, gen_away_form, spec_home_form, spec_away_form])

    # --- B. CALCULATE STATISTICAL FORM ---
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
        ('GD', 'Home_GD', 'Away_GD'),
        ('Clean_Sheet', 'Home_Clean_Sheet', 'Away_Clean_Sheet'),
        ('FTS', 'Home_FTS', 'Away_FTS'),
        ('Trad_Pts', 'Home_Trad_Pts', 'Away_Trad_Pts'),
        ('Comeback', 'Home_Comeback', 'Away_Comeback')
    ]

    for stat_prefix, h_col, a_col in stats_to_calculate:
        h_gen = generic_stats.get_advanced_rolling_stats(df, 'Home', f'Gen_{stat_prefix}', h_col, a_col, ema_windows=ema_windows, sma_windows=sma_windows, min_matches=10)
        a_gen = generic_stats.get_advanced_rolling_stats(df, 'Away', f'Gen_{stat_prefix}', h_col, a_col, ema_windows=ema_windows, sma_windows=sma_windows, min_matches=10)
        
        h_spec = generic_stats.get_specific_rolling_stats(df, 'HomeTeam', h_col, stat_prefix, ema_windows=ema_windows, sma_windows=sma_windows, min_matches=10)
        a_spec = generic_stats.get_specific_rolling_stats(df, 'AwayTeam', a_col, stat_prefix, ema_windows=ema_windows, sma_windows=sma_windows, min_matches=10)
        
        new_feature_blocks.extend([h_gen, a_gen, h_spec, a_spec])

    # Concatenate everything for this specific season
    df_season = pd.concat([df_season] + new_feature_blocks, axis=1)
    df_season['FTR'] = results.get_target_class(df)

    # --- C. TIME AND DATE LOGIC ---
    df_season['Week_of_Year'] = df['Date'].dt.isocalendar().week
    df_season['Day_of_Year'] = df['Date'].dt.dayofyear

    # --- D. BOOKMAKER ODDS ---
    home_bookies = ['1XBH', 'B365H', 'BFH', 'BFDH', 'BMGMH', 'BVH', 'BSH', 'BWH', 'CLH', 'GBH', 'IWH', 'LBH', 'PSH', 'PH', 'SOH', 'SBH', 'SJH', 'SYH', 'VCH', 'WHH']
    draw_bookies = ['1XBD', 'B365D', 'BFD', 'BFDD', 'BMGMD', 'BVD', 'BSD', 'BWD', 'CLD', 'GBD', 'IWD', 'LBD', 'PSD', 'PD', 'SOD', 'SBD', 'SJD', 'SYD', 'VCD', 'WHD']
    away_bookies = ['1XBA', 'B365A', 'BFA', 'BFDA', 'BMGMA', 'BVA', 'BSA', 'BWA', 'CLA', 'GBA', 'IWA', 'LBA', 'PSA', 'PA', 'SOA', 'SBA', 'SJA', 'SYA', 'VCA', 'WHA']

    valid_home = [c for c in home_bookies if c in df.columns]
    valid_draw = [c for c in draw_bookies if c in df.columns]
    valid_away = [c for c in away_bookies if c in df.columns]

    df_season['Avg_Odds_Home'] = df[valid_home].mean(axis=1)
    df_season['Avg_Odds_Draw'] = df[valid_draw].mean(axis=1)
    df_season['Avg_Odds_Away'] = df[valid_away].mean(axis=1)

    # --- E. FINAL GENERALIZATION & PURGE ---
    df_season = df_season.drop(columns=['Date', 'HomeTeam', 'AwayTeam'])
    
    # Drop NaNs purely for this season's 10-game warmup
    df_season = df_season.dropna().reset_index(drop=True)
    
    return df_season


# =========================================================
# 2. THE MAIN LOOP (FILE BY FILE)
# =========================================================
if __name__ == "__main__":
    folder_path = "../raw-data/data/*.csv"     
    all_files = glob.glob(folder_path)
    
    processed_seasons_list = []

    for filename in sorted(all_files):
        try:
            season_name = os.path.basename(filename).replace('.csv', '')
            print(f"Processing Season: {season_name}...")
            
            # Read the raw file
            temp_df = pd.read_csv(filename, encoding='unicode_escape')
            
            # Process the entire season in isolation
            season_processed_df = process_single_season(temp_df)
            
            # Add to our master list
            processed_seasons_list.append(season_processed_df)
            
        except pd.errors.ParserError as e:
            print(f"\n[!] PARSER ERROR IN FILE: {filename}")
            print(f"Error details: {e}\n")
            raise 

    # =========================================================
    # 3. MASTER CONCATENATION & EXPORT
    # =========================================================
    print("\nStitching all processed seasons together...")
    df_train_final = pd.concat(processed_seasons_list, axis=0, ignore_index=True)

    print(f"Final Dataset Ready! Total rows: {len(df_train_final)}")
    df_train_final.to_csv('../processed-data/training_data.csv', index=False)
    print("Advanced Training Data Exported Successfully!")