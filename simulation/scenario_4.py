import numpy as np
from src.dataset import DoubleMLData
from src.DML_BM import DML_BM
from generate_data import generate_simulated_data_for_doubleml

# In this senerior, we generated a simulated dataset comprising 100,000 subjects to evaluate the performance of our proposed model in a high-dimensional setting. The dataset included three distinct modalities, each containing 200 features. Additionally, 10,000 covariates were generated for each subject to serve as predictors for the modalities and outcome. The outcome effects for each modality were structured such that half of the features had no effect (coefficients set to zero), while the remaining features had effects gradually increasing from 0 to 1. Missing data was introduced under a Missing Completely At Random (MCAR) mechanism, with a missing probability of 0.5 for each modality. This complex simulation scenario allowed us to assess the robustness and effectiveness of our model in the presence of high-dimensional data and incomplete information.
# Representative configurations for testing
representative_configs = [
    {
        'n_obs': 100000,
        'p': 10000,
        'modalities': {'mod1': 200, 'mod2': 200, 'mod3': 200},
        'covariate_effects': {
            'mod1': np.random.normal(0, 1, (10000, 200)),
            'mod2': np.random.normal(0, 1, (10000, 200)),
            'mod3': np.random.normal(0, 1, (10000, 200))
        },
        'outcome_effects': {
            'mod1': np.concatenate((np.zeros(100), np.linspace(0, 1, 100))),
            'mod2': np.concatenate((np.zeros(100), np.linspace(0, 1, 100))),
            'mod3': np.concatenate((np.zeros(100), np.linspace(0, 1, 100)))
        },
        'outcome_effects_covariates': np.random.normal(0, 1, 10000),
        'missing_prob': 0.5,
        'missing_mechanism': 'MCAR'
    }
]

# Generate datasets for each configuration
for config in representative_configs:
    simulated_data = generate_simulated_data_for_doubleml(
        config['n_obs'],
        config['p'],
        config['modalities'],
        config['covariate_effects'],
        config['outcome_effects'],
        config['outcome_effects_covariates'],
        config['missing_prob'],
        config['missing_mechanism']
    )
    simulated_data.summary()


# Train Double Machine Learning model using DML_BM
dml_bm = DML_BM(obj_dml_data=simulated_data, n_folds=5)
dml_bm.fit()
print(f"\nSummary:\n{dml_bm.summary}")

