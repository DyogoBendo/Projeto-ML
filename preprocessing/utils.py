import pandas as pd
import glob
import os

def read_data():    
    folder_path = "../raw-data/data/*.csv"     
    all_files = glob.glob(folder_path)
    df_list = []

    for filename in all_files:
        try:
            # Try to read the file normally
            temp_df = pd.read_csv(filename, encoding='unicode_escape')
            
            season_name = os.path.basename(filename).replace('.csv', '')
            temp_df['Season'] = season_name 
            
            df_list.append(temp_df)
            
        except pd.errors.ParserError as e:
            # If it fails, print the exact file causing the issue
            print(f"\n[!] PARSER ERROR IN FILE: {filename}")
            print(f"Error details: {e}\n")
            # You can either 'continue' to skip the whole file, or let it crash so you can go fix it.
            raise 

    return pd.concat(df_list, axis=0, ignore_index=True)