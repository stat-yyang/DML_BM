import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from src.dataset import DoubleMLData
from matplotlib.lines import Line2D
import copy

def generate_simulated_data_for_doubleml(n_obs, p, modalities, covariate_effects, outcome_effects, outcome_effects_covariates, missing_prob=0.2, missing_mechanism='MCAR', scale=False, predictor_covariates_num = 200, feature_covariates_num = [10, 101]):
    if not (0 <= missing_prob <= 1):
        raise ValueError("missing_prob must be between 0 and 1")

    np.random.seed(42)
    
    # Generate covariates (X)
    # X = np.random.normal(0, 1, (n_obs, p))
    X = np.random.randint(0, 3, size=(n_obs, p))
    covariates_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(p)])
    covariates_df.insert(0, 'eid', range(n_obs))
    
    # Generate modalities (Z) and apply missingness
    df_modalities = []
    Z = {}
    Z_masked = {}

    M = generate_missing_indicator(n_obs, len(modalities), missing_mechanism, covariates=X)

    predictor_dict = {'Y': [f'X{i}' for i in range(min(p, predictor_covariates_num))]} 

    for idx, (name, n_vars) in enumerate(modalities.items()):
        Z[name] = np.zeros((n_obs, n_vars))
        for i in range(n_vars):
            # Select a random number of covariates (between feature_covariates_num[0] and feature_covariates_num[1]) for each feature
            n_selected_covariates = np.random.randint(feature_covariates_num[0], feature_covariates_num[1])
            selected_covariates = np.random.choice(p, n_selected_covariates, replace=False)
            mask = np.zeros((p,))
            mask[selected_covariates] = 1
            masked_effects = covariate_effects[name][:, i] * mask
            print('masked_effects')
            print(masked_effects)
            Z[name][:, i] = X @ masked_effects + np.random.normal(0, 0.1, n_obs)
            predictor_dict[f'{name}_{i}'] = [f'X{j}' for j in selected_covariates]

        Z_masked = copy.deepcopy(Z)
        # apply block missing 
        Z_masked[name][M[:, idx] == 0, :] = np.nan
        modality_df = pd.DataFrame(Z_masked[name], columns=[f'{name}_{i}' for i in range(n_vars)])
        modality_df.insert(0, 'eid', range(n_obs))
        modality_df = modality_df.set_index('eid').reindex(covariates_df['eid']).reset_index()
        modality_df = modality_df.dropna().reset_index(drop=True)
        df_modalities.append(modality_df)


    # Apply mask on outcome_effects_covariates using predictor_dict for Y
    y_covariates_mask = np.zeros((p,))
    y_covariates_indices = [int(covariate[1:]) for covariate in predictor_dict['Y']]
    y_covariates_mask[y_covariates_indices] = 1
    masked_outcome_effects_covariates = outcome_effects_covariates * y_covariates_mask
    
    # Generate outcome (Y)
    Y = np.zeros(n_obs)
    print(Y)
    print(Z[name])
    for name, effects in outcome_effects.items():
        Y += Z[name] @ effects
    Y += X @ masked_outcome_effects_covariates + np.random.normal(0, 0.1, n_obs)
    y_df = pd.DataFrame({'eid': range(n_obs), 'Y': Y})

    # Generate outcome (Y)
    Y = np.zeros(n_obs)
    for name, effects in outcome_effects.items():
        Y += Z[name] @ effects
    Y += X @ outcome_effects_covariates + np.random.normal(0, 0.1, n_obs)
    y_df = pd.DataFrame({'eid': range(n_obs), 'Y': Y})



    # Remove duplicate entries in predictor_dict
    for key in predictor_dict:
        predictor_dict[key] = list(set(predictor_dict[key]))

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
    p = 300
    modalities = {'mod1': 2, 'mod2': 3}
    covariate_effects = {
        'mod1': np.random.normal(0, 1, (p, 5)),
        'mod2': np.random.normal(0, 1, (p, 5))
    }
    outcome_effects = {'mod1': np.array([0.5, 0.7]), 'mod2': np.array([0.6, 0.8, 0.9])}
    outcome_effects_covariates = np.random.normal(0, 5, p)

    # Generate data
    simulated_data = generate_simulated_data_for_doubleml(n_obs, p, modalities, covariate_effects, outcome_effects, outcome_effects_covariates)
    print(simulated_data)

