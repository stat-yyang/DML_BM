import pandas as pd
import os
import glob
import pickle
from sklearn.preprocessing import StandardScaler

def create_predictor_dict(snp_info_path = '../snp_info.txt'):
    """
    Creates a dictionary mapping each gene to its associated SNPs.

    Parameters:
    - snp_info_path (str): Path to the snp_info.txt file.

    Returns:
    - dict: Dictionary with genes as keys and lists of SNPs as values.
    """
    snp_info = pd.read_csv(snp_info_path, sep='\t')  # Adjust separator if needed
    predictor_dict = {}
    for _, row in snp_info.iterrows():
        gene = row['gene']
        snp = row['SNP']
        if gene not in predictor_dict:
            predictor_dict[gene] = []
        predictor_dict[gene].append(snp)
    return predictor_dict

def read_modalities(features_dir = '../features', modality_files = ['protein_csv', 'hipp_pc.csv', 'hipp_subfields.csv']):
    """
    Reads specified modality files from the features directory.

    Parameters:
    - features_dir (str): Directory containing feature files.
    - modality_files (list of str): List of filenames for modalities.

    Returns:
    - list of pd.DataFrame: List containing each modality DataFrame.
    """
    modalities = []
    for file in modality_files:
        file_path = os.path.join(features_dir, file)
        df = pd.read_csv(file_path, sep='\t')  # Adjust separator if needed
        modalities.append(df)
    return modalities


def main():
    # Paths (update these paths as necessary)
    snp_info_path = ".."
    geno_file = ".."
    features_dir = ".."
    y_file_path = ".." 
    modality_files = ['protein_csv', 'hipp_pc.csv', 'hipp_subfields.csv']

    # Step 1: Create predictor_dict
    predictor_dict = create_predictor_dict(snp_info_path)
    print("Predictor dictionary created.")

    # Step 2: Create covariates_df by combining genotype files
    covariates_df = pd.read_csv(geno_file, sep=',') 
    print("Covariates DataFrame created.")

    # Step 3: Read df_modalities
    df_modalities = read_modalities(features_dir, modality_files)
    print("Modalities DataFrames created.")

    # Step 4: Read y_df
    y_df = pd.read_csv(y_file_path, sep=',') 
    if 'eid' not in y_df.columns:
        raise ValueError("'eid' column is required in y_df.")
    print("Outcome DataFrame (y_df) created.")

    # Step 5: Define modality_names
    modality_names = [os.path.splitext(file)[0] for file in modality_files]
    print("Modality names defined.")

    # Step 6: Initialize DoubleMLData
    double_ml_data = DoubleMLData(
        y_df=y_df,
        df_modalities=df_modalities,
        covariates_df=covariates_df,
        predictor_dict=predictor_dict,
        modality_names=modality_names,
        scale=True  # Set to False if scaling is not desired
    )

    with open(os.path.join(sim_folder_name, '../ukb_dml_data.pkl'), 'wb') as f:
        pickle.dump(double_ml_data, f)
    
    print("DoubleMLData object initialized.")

    # Optional: Inspect the DoubleMLData object
    print("Y DataFrame:")
    print(double_ml_data.y_df.head())
    print("\nCovariates DataFrame:")
    print(double_ml_data.covariates_df.head())
    for i, modality in enumerate(double_ml_data.df_modalities):
        print(f"\nModality '{double_ml_data.modality_names[i]}' DataFrame:")
        print(modality.head())
    print("\nPredictor Dictionary:")
    print(predictor_dict)

if __name__ == "__main__":
    main()
