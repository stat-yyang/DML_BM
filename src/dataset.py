import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error

class DoubleMLData:
    def __init__(self, y_df, df_modalities, covariates_df, predictor_dict, modality_names, scale=False, complete_case=True, impute=True):
        """
        Initialize the DoubleMLData object.

        Parameters:
        - y_df (pd.DataFrame): DataFrame containing the outcome variable with 'eid' as subject identifier.
        - df_modalities (list of pd.DataFrame): List of DataFrames for each omic modality, each starting with 'eid'.
        - covariates_df (pd.DataFrame): DataFrame containing covariate features with 'eid' as subject identifier.
        - predictor_dict (dict): Dictionary mapping each feature in y and modalities to their predictors in covariates.
        - modality_names (list of str): List of names for each modality.
        - scale (bool): Whether to scale the dataframes.
        - complete_case (bool): Whether to use only the intersection of all data frames (complete cases).
        - impute (bool): Whether to impute the modalities.
        """
        self._validate_inputs(y_df, df_modalities, covariates_df, predictor_dict, modality_names)

        if scale:
            self._scale_dataframes()

        self._y_df = y_df.reset_index(drop=True)
        self._df_modalities = [df.reset_index(drop=True) for df in df_modalities]
        self._covariates_df = covariates_df.reset_index(drop=True)
        self._predictor_dict = predictor_dict.copy()
        self._modality_names = modality_names

        # self._ensure_complete_case()

        print(self.calculate_r2_for_modalities())
        
        # Set of unique eids across all datasets
        self._subject_ids = self._get_all_subject_ids()

        # Make sure each dataframe contains the same number of rows with matched eid
        self._check_subject(complete_case)

        # Ensure each modality has proper feature names and check for overlap
        self._check_feature_name_overlap()

        self._df_modalities_long = self._combine_modalities()

        if impute:
            # Impute missing modalities
            self._impute_modalities()


    def _validate_inputs(self, y_df, df_modalities, covariates_df, predictor_dict, modality_names):
        """Validate inputs to the class."""
        if 'eid' not in y_df.columns:
            raise ValueError("y_df must have an 'eid' column for subject identifiers.")
        if 'eid' not in covariates_df.columns:
            raise ValueError("covariates_df must have an 'eid' column for subject identifiers.")
        for df in df_modalities:
            if 'eid' not in df.columns:
                raise ValueError("Each modality DataFrame must have an 'eid' column for subject identifiers.")

        if not isinstance(y_df, pd.DataFrame):
            raise TypeError("y_df must be a pandas DataFrame.")
        if not isinstance(df_modalities, list) or not all(isinstance(df, pd.DataFrame) for df in df_modalities):
            raise TypeError("df_modalities must be a list of pandas DataFrames.")
        if not isinstance(covariates_df, pd.DataFrame):
            raise TypeError("covariates_df must be a pandas DataFrame.")
        if not isinstance(predictor_dict, dict):
            raise TypeError("predictor_dict must be a dictionary.")
        if not isinstance(modality_names, list) or len(modality_names) != len(df_modalities):
            raise ValueError("modality_names must be a list with the same length as df_modalities.")

    def _get_complete_case(self, y_df, df_modalities, covariates_df):
        """
        Get the intersection of all subject IDs across y, modalities, and covariates to ensure complete cases.

        Parameters:
        - y_df (pd.DataFrame): Outcome DataFrame.
        - df_modalities (list of pd.DataFrame): List of modality DataFrames.
        - covariates_df (pd.DataFrame): Covariates DataFrame.

        Returns:
        - y_df, df_modalities, covariates_df: DataFrames containing only the complete cases.
        """
        common_eids = set(y_df['eid']).intersection(set(covariates_df['eid']))
        for df in df_modalities:
            common_eids = common_eids.intersection(set(df['eid']))

        y_df = y_df[y_df['eid'].isin(common_eids)].reset_index(drop=True)
        df_modalities = [df[df['eid'].isin(common_eids)].reset_index(drop=True) for df in df_modalities]
        covariates_df = covariates_df[covariates_df['eid'].isin(common_eids)].reset_index(drop=True)

        return y_df, df_modalities, covariates_df

    def _check_feature_name_overlap(self):
        """Check that there is no overlap between feature names in each modality DataFrame."""
        all_features = set()
        for modality_df in self._df_modalities:
            features = set(modality_df.columns[1:])
            if not all_features.isdisjoint(features):
                raise ValueError("Feature names overlap between modalities.")
            all_features.update(features)

    def _get_all_subject_ids(self):
        """Get the union of all subject IDs across y, modalities, and covariates."""
        all_eids = set(self._y_df['eid']).union(set(self._covariates_df['eid']))
        for df in self._df_modalities:
            all_eids = all_eids.union(set(df['eid']))
        return all_eids

    def _check_subject(self, complete_case):
        """
        Ensure all dataframes have the same number of rows with matching subjects.
        If complete_case is True, use the intersection of all 'eid' columns and retain only common subjects.
        If complete_case is False, use the union of all 'eid' columns, with missing subjects having NaN values.
        """
        if complete_case:
            # Get the intersection of all unique 'eid' values across all dataframes
            common_eids = set(self._y_df['eid']).intersection(set(self._covariates_df['eid']))
            for df in self._df_modalities:
                common_eids = common_eids.intersection(set(df['eid']))
            all_eids = common_eids

            # Retain only common subjects across all dataframes
            self._y_df = self._y_df[self._y_df['eid'].isin(all_eids)].reset_index(drop=True)
            self._df_modalities = [df[df['eid'].isin(all_eids)].reset_index(drop=True) for df in self._df_modalities]
            self._covariates_df = self._covariates_df[self._covariates_df['eid'].isin(all_eids)].reset_index(drop=True)
        else:
            # Get the union of all unique 'eid' values across all dataframes
            all_eids = self._subject_ids

            # Reindex all dataframes to ensure consistency across subjects
            self._y_df = self._y_df.set_index('eid').reindex(all_eids).reset_index()
            self._df_modalities = [df.set_index('eid').reindex(all_eids).reset_index() for df in self._df_modalities]
            self._covariates_df = self._covariates_df.set_index('eid').reindex(all_eids).reset_index()
   
   
    def _ensure_complete_case(self):
        """Ensure that the dataset is a complete case, if not apply mean imputation."""
        imputer = SimpleImputer(strategy='mean')

        # Impute y_df (excluding 'eid')
        if self._y_df.isnull().any().any():
            self._y_df.iloc[:, 1:] = imputer.fit_transform(self._y_df.iloc[:, 1:])

        # Impute each modality DataFrame (excluding 'eid')
        for df in self._df_modalities:
            if df.isnull().any().any():
                df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

        # Impute covariates_df (excluding 'eid')
        if self._covariates_df.isnull().any().any():
            self._covariates_df.iloc[:, 1:] = imputer.fit_transform(self._covariates_df.iloc[:, 1:])
   
    def _combine_modalities(self):
        """
        Combine all dataframes in modalities together on 'eid'.
        
        Returns:
        - combined_df (pd.DataFrame): A DataFrame containing all modalities combined, indexed by 'eid'.
        """
        combined_df = self._df_modalities[0].set_index('eid')
        for df in self._df_modalities[1:]:
            combined_df = combined_df.join(df.set_index('eid'), how='outer', rsuffix='_modality')
        combined_df = combined_df.reset_index()
        return combined_df

    def compute_missing_mask(self):
        """
        Compute a mask to determine missing values across all modalities.

        Returns:
        - pd.DataFrame: A boolean mask with the same dimensions as the combined modalities DataFrame, indicating missing values.
        """
        missing_mask = self._df_modalities_long.iloc[:, 1:].isna()
        return missing_mask

    def _scale_dataframes(self):
        """
        Scale each DataFrame (y, modalities, covariates) separately.
        """
        scaler = StandardScaler()

        # Scale y_df (excluding 'eid')
        self._y_df.iloc[:, 1:] = scaler.fit_transform(self._y_df.iloc[:, 1:])

        # Scale each modality DataFrame (excluding 'eid')
        for df in self._df_modalities:
            df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

        # Scale covariates_df (excluding 'eid')
        self._covariates_df.iloc[:, 1:] = scaler.fit_transform(self._covariates_df.iloc[:, 1:])

    def _impute_modalities(self, model=LinearRegression()):
        """
        Impute missing values in modalities using the given model.

        Parameters:
        - model: An estimator object implementing 'fit' and 'predict' methods.
        """
        for idx, feature in enumerate(self._df_modalities_long.columns[1:]):
            predictors = self._predictor_dict.get(feature, [])
            # Filter rows where modality data is missing
            missing_rows = self._df_modalities_long[feature].isna()
            if missing_rows.any():
                X_train = self._covariates_df.loc[~missing_rows, predictors]
                y_train = self._df_modalities_long.loc[~missing_rows, feature]
                X_test = self._covariates_df.loc[missing_rows, predictors]
                model.fit(X_train, y_train)
                self._df_modalities_long.loc[missing_rows, feature] = model.predict(X_test)

        print("Impute all modalities successfully!")
            
    
    def calculate_r2_for_modalities(self):
        """
        Calculate R^2 explained by the predictors for all features in the modalities.

        Returns:
        - r2_results (dict): Dictionary with modality names as keys and DataFrames containing R2, MSE, and Mean as values.
        """
        r2_results = {}
        for modality_name, modality_df in zip(self._modality_names, self._df_modalities):
            modality_r2 = []
            modality_mse = []
            modality_mean = []
            
            # Merge modality dataframe with covariates dataframe on 'eid'
            merged_df = pd.merge(modality_df, self._covariates_df, on='eid', how='inner')
            
            for feature in modality_df.columns[1:]:  # Skip 'eid'
                predictors = self._predictor_dict.get(feature, [])
                if not predictors:
                    continue
                
                X = merged_df[predictors].values
                y = merged_df[feature].values
                
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate R2, MSE, and Mean
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                mean_value = np.mean(y)
                
                modality_r2.append(r2)
                modality_mse.append(mse)
                modality_mean.append(mean_value)
            
            # Store results in a DataFrame
            results_df = pd.DataFrame({
                'Feature': modality_df.columns[1:],
                'R2': modality_r2,
                'MSE': modality_mse,
                'Mean': modality_mean
            })
            
            r2_results[modality_name] = results_df
        
        return r2_results

    
    @property
    def y(self):
        """Return the outcome variable as a DataFrame."""
        return self._y_df.copy()

    @property
    def modality_names(self):
        """Return a list of modality names."""
        return self._modality_names

    @property
    def modalities(self):
        """Return a list of modality DataFrames."""
        return [df.copy() for df in self._df_modalities]

    @property
    def covariates(self):
        """Return the covariates DataFrame."""
        return self._covariates_df.copy()

    @property
    def predictor_dict(self):
        """Return the predictor dictionary."""
        return self._predictor_dict.copy()

    def summary(self):
        """Print a summary of the dataset."""
        print("=== DoubleMLData Summary ===")
        print(f"Number of subjects: {len(self._subject_ids)}")
        print(f"Number of outcome variables: {self._y_df.shape[1] - 1}")
        print(f"Number of omic modalities: {len(self._df_modalities)}")
        for idx, modality_df in enumerate(self._df_modalities):
            print(f"  Modality {self._modality_names[idx]}: {modality_df.shape[1] - 1} features")
        print(f"Number of covariates: {self._covariates_df.shape[1] - 1}")
        print("============================")

# Example Usage
if __name__ == "__main__":
    # Simulate some data
    np.random.seed(42)
    n_samples = 1000

    # Outcome variable with 'eid'
    y_df = pd.DataFrame({
        'eid': range(n_samples),
        'y': np.random.randn(n_samples)
    })

    # Multi-omic modalities with 'eid'
    omic1 = pd.DataFrame({
        'eid': range(0, n_samples, 2),  # Half the subjects
        'gene_expr1': np.random.randn(n_samples // 2),
        'gene_expr2': np.random.randn(n_samples // 2)
    })

    omic2 = pd.DataFrame({
        'eid': range(1, n_samples, 2),  # The other half of the subjects
        'metabolite1': np.random.randn(n_samples // 2),
        'metabolite2': np.random.randn(n_samples // 2)
    })

    df_modalities = [omic1, omic2]
    modality_names = ['omic1', 'omic2']

    # Covariates with 'eid'
    covariates_df = pd.DataFrame({
        'eid': range(n_samples),
        'age': np.random.randint(20, 70, size=n_samples),
        'BMI': np.random.randn(n_samples)
    })

    # Predictor dictionary
    predictor_dict = {
        'Y': ['age', 'BMI'],
        'gene_expr1': ['age'],
        'gene_expr2': ['BMI'],
        'metabolite1': ['age'],
        'metabolite2': ['BMI']
    }

    # Initialize the DoubleMLData object
    dml_data = DoubleMLData(
        y_df=y_df,
        df_modalities=df_modalities,
        covariates_df=covariates_df,
        predictor_dict=predictor_dict,
        modality_names=modality_names
    )

    # Display summary
    dml_data.summary()

    # Compute missing mask
    missing_mask = dml_data.compute_missing_mask()
    print("\nMissing Mask:", missing_mask)

    # Scale dataframes
    dml_data.scale_dataframes()
    print("\nScaled Outcome (y):", dml_data.y.head())
