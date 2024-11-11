import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from dataset import DoubleMLData
from score import LinearScoreMixin

class DML_BM:
    def __init__(self, obj_dml_data, ml_g=None, ml_m=None, n_folds=5):
        """
        Initialize the DML_BM object.

        Parameters:
        - obj_dml_data (DoubleMLData): An instance of DoubleMLData.
        - ml_g: Model to be used for g estimation (default: LinearRegression).
        - ml_m: Model to be used for m estimation (default: LinearRegression).
        - n_folds (int): Number of folds for cross-fitting.
        """
        if ml_g is None:
            ml_g = LinearRegression()
        if ml_m is None:
            ml_m = LinearRegression()
        
        self._dml_data = obj_dml_data
        self._ml_g = ml_g
        self._ml_m = ml_m
        self._n_folds = n_folds
        self._modality_names = list(self._dml_data.modality_names)
        self._summary = None

    def _nuisance_est(self, smpls):
        g_models = []
        g_hat = np.zeros((self._dml_data.y.shape[0], 1))

        # Fit nuisance model for g across all modalities
        for idx, (train_index, test_index) in enumerate(smpls):
            g_model = clone(self._ml_g)
            predictors = self._dml_data.predictor_dict['y']
            
            X_train = self._dml_data.covariates.iloc[train_index][predictors]
            Y_train = self._dml_data.y.iloc[train_index]['y']

            # Fit the model on training data
            g_model.fit(X_train, Y_train)

            # Predict on the test data
            X_test = self._dml_data.covariates.iloc[test_index][predictors]
            g_hat[test_index] = g_model.predict(X_test).reshape(-1, 1)

            # Store the model for each fold
            g_models.append(g_model)

        return g_models, g_hat

    def _nuisance_est_modalites(self, smpls):
        m_models = []
        m_hat = np.full((self._dml_data._df_modalities_long.shape[0], self._dml_data._df_modalities_long.shape[1] - 1), np.nan)

        for idx, (train_index, test_index) in enumerate(smpls):
            fold_m_models = []
            fold_m_hat = np.full((len(test_index), self._dml_data._df_modalities_long.shape[1] - 1), np.nan)

            # Loop through features in self._dml_data._df_modalities_long excluding 'eid'
            for feature_idx, feature in enumerate(self._dml_data._df_modalities_long.columns[1:]):
                predictors = self._dml_data.predictor_dict.get(feature, [])

                # Check if predictors list is empty
                if not predictors:
                    raise ValueError(f"No predictors found for feature '{feature}'. Please ensure predictors are correctly specified.")

                m_model = clone(self._ml_m)

                # Extract training data for the current feature, drop rows with missing values in X_train or D_train
                X_train = self._dml_data.covariates.iloc[train_index][predictors]
                D_train = self._dml_data._df_modalities_long.iloc[train_index][feature]

                # Drop rows with NaN values in either X_train or D_train
                valid_indices = X_train.index.intersection(D_train.dropna().index)
                X_train = X_train.loc[valid_indices].dropna()
                D_train = D_train.loc[valid_indices]

                # Debugging to verify training data
                print(f"Fold {idx}, Feature {feature}: X_train shape: {X_train.shape}, D_train shape: {D_train.shape}")

                # Ensure no empty data before fitting
                if X_train.empty or D_train.empty:
                    raise ValueError(f"X_train or D_train is empty in fold {idx} for feature {feature}. Ensure valid predictors and complete data.")

                # Fit the model on training data for the current feature
                m_model.fit(X_train, D_train)

                # Predict on the test data
                X_test = self._dml_data.covariates.iloc[test_index][predictors]
                fold_m_hat[:, feature_idx] = m_model.predict(X_test)

                # Store the model for each feature in the current fold
                fold_m_models.append(m_model)

            # Store predictions for the current fold
            m_hat[test_index] = fold_m_hat
            # Store the models for each fold
            m_models.append(fold_m_models)

        return m_models, m_hat

    def _obtain_psi(self, smpls):
        psi_a_folds = []
        psi_b_folds = []

        g_models, g_hat = self._nuisance_est(smpls)

        # Residuals for outcome variable
        res_y = self._dml_data.y.iloc[:, 1:] - g_hat

        m_models, m_hat = self._nuisance_est_modalites(smpls)
        
        missing_mask = self._dml_data.compute_missing_mask()
        df_modalities_long_copy = self._dml_data._df_modalities_long.iloc[:, 1:].copy()

        # Ensure the missing_mask and m_hat have the same shape
        if missing_mask.shape != m_hat.shape:
            raise ValueError("The shapes of missing_mask and m_hat do not match. Please check the dimensions.")

        # Ensure the missing_mask and df_modalities_long_copy have the same shape
        if missing_mask.shape != df_modalities_long_copy.shape:
            raise ValueError("The shapes of missing_mask and m_hat do not match. Please check the dimensions.")

        # Update df_modalities_long_copy with m_hat values where missing_mask is True
        df_modalities_long_copy = df_modalities_long_copy.mask(missing_mask, m_hat)
        # df_modalities_long_copy[missing_mask] = m_hat[missing_mask]
        # df_modalities_long_copy[missing_mask == 1] = m_hat[missing_mask == 1]
        res_d = df_modalities_long_copy - m_hat

        # Ensure res_d and res_y are aligned for dot product
        if res_d.shape[0] != len(res_y):
            raise ValueError("The number of rows in res_d and res_y do not match for matrix multiplication.")

        # Ensure res_y has the same shape as res_d for Hadamard product
        res_y_identity = np.ones_like(res_d) * res_y.values.reshape(-1, 1)

        # Compute psi_a and psi_b
        psi_a = np.einsum('ij,ik->ijk', res_d, df_modalities_long_copy)
        psi_b = -res_d * res_y_identity

        psi_a_folds.append(psi_a)
        psi_b_folds.append(psi_b)

        # Concatenate psi_a and psi_b across folds
        psi_a = np.concatenate(psi_a_folds, axis=0)
        psi_b = np.concatenate(psi_b_folds, axis=0)

        # Return psi_elements as a dictionary to be used in LinearScoreMixin
        psi_elements = {'psi_a': psi_a, 'psi_b': psi_b}
        return psi_elements


    def fit(self):
        """
        Fit the DML_BM model.
        """
        self.coef_ = {}
        self.se_ = {}
        
        kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=42)
        smpls = list(kf.split(np.arange(self._dml_data.y.shape[0])))
        psi_elements = self._obtain_psi(smpls)
        score = LinearScoreMixin()
        coef = score._est_coef(psi_elements)
        se = score._est_sd(psi_elements, coef)
        self.coef_ = coef
        self.se_ = se

        print(coef.shape)
        print(se.shape)
        
        self._calc_summary_measures()
        return self

    @property
    def summary(self):
        return self._summary

    def _calc_summary_measures(self):
        from scipy.stats import norm
        self._summary = {
            'coef': self.coef_,
            'se': self.se_,
            't': {k: self.coef_[k] / self.se_[k] for k in range(len(self._modality_names))},
            'pval': {k: 2 * (1 - norm.cdf(np.abs(self.coef_[k] / self.se_[k]))) for k in range(len(self._modality_names))}
        }
