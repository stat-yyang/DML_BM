import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import DoubleMLData
from src.DML_BM import DML_BM
from simulation.generate_data import generate_simulated_data_for_doubleml
from simulation.visualization import visualize_fitting_results_with_true_effects
import pickle
import os
import uuid

# In this scenario, we generate a simulated dataset comprising 10,000 subjects to evaluate the performance of our proposed model in a high-dimensional setting. 
# The dataset included three distinct modalities, each containing 50 features. Additionally, 10,000 covariates were generated for each subject to serve as predictors for the modalities and outcome. 
# Missing data was introduced under a Missing Completely At Random (MCAR) mechanism, with a missing probability of 0.5 for each modality. 
# This complex simulation scenario allowed us to assess the robustness and effectiveness of our model in the presence of high-dimensional data and incomplete information.

# sbatch --ntasks=1 --time=12:00:00 --mem=16G --wrap="python scenario_1_batch.py" 

# Representative configurations for testing
true_effects = {
    'mod1': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1))),
    'mod2': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1))),
    'mod3': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1)))
}

representative_configs = [
    {
        'n_obs': 10000,
        'p': 20,
        'modalities': {'mod1': 5, 'mod2': 5, 'mod3': 5},
        'covariate_effects': {
            'mod1': np.random.normal(1, 1, (20, 5)),
            'mod2': np.random.normal(1, 1, (20, 5)),
            'mod3': np.random.normal(1, 1, (20, 5))
        },
        'outcome_effects': true_effects,
        'outcome_effects_covariates': np.random.normal(1, 1, 20),
        'missing_prob': 1,
        'missing_mechanism': 'MCAR',
        'feature_covariates_num': [10, 15],
        'predictor_covariates_num': 10
    },
    {
        'n_obs': 10000,
        'p': 20,
        'modalities': {'mod1': 5, 'mod2': 5, 'mod3': 5},
        'covariate_effects': {
            'mod1': np.random.normal(1, 1, (20, 5)),
            'mod2': np.random.normal(1, 1, (20, 5)),
            'mod3': np.random.normal(1, 1, (20, 5))
        },
        'outcome_effects': true_effects,
        'outcome_effects_covariates': np.random.normal(1, 1, 20),
        'missing_prob': 0.8,
        'missing_mechanism': 'MCAR',
        'feature_covariates_num': [10, 15],
        'predictor_covariates_num': 10
    },
    {
        'n_obs': 10000,
        'p': 20,
        'modalities': {'mod1': 5, 'mod2': 5, 'mod3': 5},
        'covariate_effects': {
            'mod1': np.random.normal(1, 1, (20, 5)),
            'mod2': np.random.normal(1, 1, (20, 5)),
            'mod3': np.random.normal(1, 1, (20, 5))
        },
        'outcome_effects': true_effects,
        'outcome_effects_covariates': np.random.normal(1, 1, 20),
        'missing_prob': 0.5,
        'missing_mechanism': 'MCAR',
        'feature_covariates_num': [10, 15],
        'predictor_covariates_num': 10
    },
    {
        'n_obs': 10000,
        'p': 20,
        'modalities': {'mod1': 5, 'mod2': 5, 'mod3': 5},
        'covariate_effects': {
            'mod1': np.random.normal(1, 1, (20, 5)),
            'mod2': np.random.normal(1, 1, (20, 5)),
            'mod3': np.random.normal(1, 1, (20, 5))
        },
        'outcome_effects': true_effects,
        'outcome_effects_covariates': np.random.normal(1, 1, 20),
        'missing_prob': 0.2,
        'missing_mechanism': 'MCAR',
        'feature_covariates_num': [10, 15],
        'predictor_covariates_num': 10
    }

]

# representative_configs=representative_configs[1:]

# Define root directory for storing results
root_dir = "/work/users/y/y/yyang96/Project2/results/simulation/scenario_1"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# Generate datasets for each configuration
for idx, config in enumerate(representative_configs):
    # Create a folder for each configuration based on n_obs, p, and missing_prob
    folder_name = os.path.join(root_dir, f"n_obs_{config['n_obs']}_missing_mechanism_{config['missing_mechanism']}_missing_prob_{config['missing_prob']}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Run multiple simulations for each configuration
    for sim_num in range(1000):  # Running 1000 simulations for each configuration
        # Create a sub-folder for each simulation run
        sim_folder_name = os.path.join(folder_name, f"simulation_{sim_num}")
        if not os.path.exists(sim_folder_name):
            os.makedirs(sim_folder_name)

        # Generate the dataset
        simulated_data = generate_simulated_data_for_doubleml(
            config['n_obs'],
            config['p'],
            config['modalities'],
            config['covariate_effects'],
            config['outcome_effects'],
            config['outcome_effects_covariates'],
            config['missing_prob'],
            config['missing_mechanism'],
            predictor_covariates_num=config['predictor_covariates_num'],
            feature_covariates_num=config['feature_covariates_num'],
            random_seed = sim_num
        )
        simulated_data.summary()

        # Train Double Machine Learning model using DML_BM
        dml_bm = DML_BM(obj_dml_data=simulated_data, n_folds=5)
        dml_bm.fit(plot=True, fig_dir=os.path.join(sim_folder_name, 'figures'))
        print(f"\nSummary:\n{dml_bm.summary}")

        # Save DML results to a file within the simulation folder
        with open(os.path.join(sim_folder_name, 'dml_summary.pkl'), 'wb') as f:
            pickle.dump(dml_bm.summary, f)
        print(f"\nSummary saved in {sim_folder_name}/dml_summary.pkl")

        fig_file = f"{sim_folder_name}/effect.png"
        visualize_fitting_results_with_true_effects(dml_bm.summary, true_effects, fig_file)


