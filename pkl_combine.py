import os
import os.path as osp
import pickle
import pandas as pd

def combine_pickle_files(directory_path, output_file):
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to store the merged data

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pkl'):
            file_path = osp.join(directory_path, file_name)
            with open(file_path, 'rb') as f:
                content = pd.compat.pickle_compat.load(f)
                if isinstance(content, pd.DataFrame):
                    combined_df = pd.concat([combined_df, content], ignore_index=True)
    
    with open(output_file, 'wb') as out:
        pickle.dump(combined_df, out, protocol=pickle.HIGHEST_PROTOCOL)

# Example usage:
directory_path = 'pkl_files/Ultimate_Data'
output_file = 'pkl_files/data_combined_2020_03_19_to_2025_01_01.pkl'
combine_pickle_files(directory_path, output_file)