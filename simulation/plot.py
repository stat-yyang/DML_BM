import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import DoubleMLData
from src.DML_BM import DML_BM
from simulation.generate_data import generate_simulated_data_for_doubleml
from simulation.visualization import visualize_multiple_fitting_results_with_true_effects
import pickle
import os
import uuid


dml_summaries = []
for root, dirs, files in os.walk('/work/users/y/y/yyang96/Project2/results/simulation/scenario_1/n_obs_10000_missing_mechanism_MCAR_missing_prob_1'):
    for file in files:
        if file == 'dml_summary.pkl':
            with open(os.path.join(root, file), 'rb') as f:
                dml_summary = pickle.load(f)
                dml_summaries.append(dml_summary)

true_effects = {
    'mod1': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1))),
    'mod2': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1))),
    'mod3': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1)))
}

results = visualize_multiple_fitting_results_with_true_effects(dml_summaries, true_effects, '/work/users/y/y/yyang96/DML_BM/simulation/n_obs_10000_missing_mechanism_MCAR_missing_prob_1.png')



dml_summaries = []
for root, dirs, files in os.walk('/work/users/y/y/yyang96/Project2/results/simulation/scenario_1/n_obs_10000_missing_mechanism_MCAR_missing_prob_0.8'):
    for file in files:
        if file == 'dml_summary.pkl':
            with open(os.path.join(root, file), 'rb') as f:
                dml_summary = pickle.load(f)
                dml_summaries.append(dml_summary)

true_effects = {
    'mod1': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1))),
    'mod2': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1))),
    'mod3': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1)))
}

results = visualize_multiple_fitting_results_with_true_effects(dml_summaries, true_effects, '/work/users/y/y/yyang96/DML_BM/simulation/n_obs_10000_missing_mechanism_MCAR_missing_prob_0.8.png')



dml_summaries = []
for root, dirs, files in os.walk('/work/users/y/y/yyang96/Project2/results/simulation/scenario_1/n_obs_10000_missing_mechanism_MCAR_missing_prob_0.5'):
    for file in files:
        if file == 'dml_summary.pkl':
            with open(os.path.join(root, file), 'rb') as f:
                dml_summary = pickle.load(f)
                dml_summaries.append(dml_summary)

true_effects = {
    'mod1': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1))),
    'mod2': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1))),
    'mod3': np.concatenate((-np.ones(1), np.zeros(3), np.ones(1)))
}

results = visualize_multiple_fitting_results_with_true_effects(dml_summaries, true_effects, '/work/users/y/y/yyang96/DML_BM/simulation/n_obs_10000_missing_mechanism_MCAR_missing_prob_0.5.png')
