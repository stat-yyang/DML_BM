import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from src.dataset import DoubleMLData
from matplotlib.lines import Line2D

def generate_simulated_data_for_doubleml(n_obs, p, p_extra, modalities, covariate_effects, outcome_effects, outcome_effects_covariates, missing_prob=0.2, missing_mechanism='MCAR'):
    if not (0 <= missing_prob <= 1):
        raise ValueError("missing_prob must be between 0 and 1")

    np.random.seed(42)
    
    # Generate covariates (X)
    X = np.random.normal(0, 1, (n_obs, p + p_extra))
    covariates_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(p + p_extra)])
    covariates_df['eid'] = range(n_obs)
    
    # Generate modalities (Z) and apply missingness
    df_modalities = []
    Z = {}
    beta = np.random.normal(0, 1, (p + p_extra, len(modalities)))
    M = generate_missing_indicator(n_obs, len(modalities), missing_mechanism, covariates=X, beta=beta)
    
    for idx, (name, n_vars) in enumerate(modalities.items()):
        Z[name] = np.zeros((n_obs, n_vars))
        for i in range(n_vars):
            Z[name][:, i] = X[:, :p] @ covariate_effects[name][:, i] + np.random.normal(0, 0.1, n_obs)
        Z[name][M[:, idx] == 0, :] = np.nan
        modality_df = pd.DataFrame(Z[name], columns=[f'{name}_{i}' for i in range(n_vars)])
        modality_df['eid'] = range(n_obs)
        modality_df = modality_df.set_index('eid').reindex(covariates_df['eid']).reset_index()
        df_modalities.append(modality_df)
    
    # Generate outcome (Y)
    Y = np.zeros(n_obs)
    for name, effects in outcome_effects.items():
        Y += Z[name] @ effects
    Y += X[:, :p] @ outcome_effects_covariates + np.random.normal(0, 0.1, n_obs)
    y_df = pd.DataFrame({'eid': range(n_obs), 'Y': Y})
    
    # Create predictor dictionary
    predictor_dict = {'Y': [f'X{i}' for i in range(p)]}
    for name in modalities.keys():
        predictor_dict[name] = [f'X{i}' for i in range(p)]
    
    # Create DoubleMLData object
    dml_data = DoubleMLData(y_df, df_modalities, covariates_df, predictor_dict, modality_names=list(modalities.keys()))
    
    return dml_data


def generate_missing_indicator(n_obs, n_modalities, mechanism='MCAR', missing_prob=0.2, covariates=None, beta=None):
    if mechanism == 'MCAR':
        M = np.random.binomial(1, missing_prob, (n_obs, n_modalities))
    elif mechanism == 'MAR':
        if covariates is None or beta is None:
            raise ValueError("Covariates and beta must be provided for MAR mechanism")
        logits = covariates @ beta
        prob = 1 / (1 + np.exp(-logits))
        M = np.random.binomial(1, prob)
    elif mechanism == 'MNAR':
        if covariates is None or beta is None:
            raise ValueError("Covariates and beta must be provided for MNAR mechanism")
        logits = covariates @ beta
        prob = 1 / (1 + np.exp(-logits))
        M = np.random.binomial(1, prob)
    else:
        raise ValueError("Invalid missing mechanism specified")
    return M

if __name__ == '__main__':
    # Example usage
    n_obs = 1000
    p = 5
    p_extra = 2
    modalities = {'mod1': 2, 'mod2': 3}
    covariate_effects = {
        'mod1': np.random.normal(0, 1, (p, 2)),
        'mod2': np.random.normal(0, 1, (p, 3))
    }
    outcome_effects = {'mod1': np.array([0.5, 0.7]), 'mod2': np.array([0.6, 0.8, 0.9])}
    outcome_effects_covariates = np.random.normal(0, 1, p)

    # Generate data
    simulated_data = generate_simulated_data_for_doubleml(n_obs, p, p_extra, modalities, covariate_effects, outcome_effects, outcome_effects_covariates)
    print(simulated_data)
