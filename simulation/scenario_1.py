import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import DoubleMLData
from src.DML_BM import DML_BM
from generate_data import generate_simulated_data_for_doubleml
import pickle
import os

# In this senerior, we generated a simulated dataset comprising 10,000 subjects to evaluate the performance of our proposed model in a high-dimensional setting. The dataset included three distinct modalities, each containing 50 features. Additionally, 10,00 covariates were generated for each subject to serve as predictors for the modalities and outcome. The outcome effects for each modality were structured such that half of the features had no effect (coefficients set to zero), while the remaining features had effects gradually increasing from 0 to 1. Missing data was introduced under a Missing Completely At Random (MCAR) mechanism, with a missing probability of 0.5 for each modality. This complex simulation scenario allowed us to assess the robustness and effectiveness of our model in the presence of high-dimensional data and incomplete information.
# Representative configurations for testing


# true_effects = {
#             'mod1': np.concatenate((np.linspace(-1, 0, 10), np.zeros(30), np.linspace(0, 1, 10))),
#             'mod2': np.concatenate((np.linspace(-1, 0, 10), np.zeros(30), np.linspace(0, 1, 10))),
#             'mod3': np.concatenate((np.linspace(-1, 0, 10), np.zeros(30), np.linspace(0, 1, 10)))
# }

true_effects = {
            'mod1': np.concatenate((-np.ones(10), np.zeros(30), np.ones(10))),
            'mod2': np.concatenate((-np.ones(10), np.zeros(30), np.ones(10))),
            'mod3': np.concatenate((-np.ones(10), np.zeros(30), np.ones(10)))
}

representative_configs = [
    {
        'n_obs': 80000,
        'p': 100,
        'modalities': {'mod1': 50, 'mod2': 50, 'mod3': 50},
        'covariate_effects': {
            'mod1': np.random.normal(1, 1, (100, 50)),
            'mod2': np.random.normal(1, 1, (100, 50)),
            'mod3': np.random.normal(1, 1, (100, 50))
        },
        'outcome_effects': true_effects,
        'outcome_effects_covariates': np.random.normal(1, 1, 100),
        'missing_prob': 0.2,
        'missing_mechanism': 'MCAR',
        'feature_covariates_num': [10, 15],
        'predictor_covariates_num': 20
    }
]

# true_effects = {
#             'mod1': np.array(1),
#             'mod2': np.array(1),
#             'mod3': np.array(1)
# }

# representative_configs = [
#     {
#         'n_obs': 3,
#         'p': 2,
#         'modalities': {'mod1': 1, 'mod2': 1, 'mod3': 1},
#         'covariate_effects': {
#             'mod1': np.random.normal(1, 1, (2, 1)),
#             'mod2': np.random.normal(1, 1, (2, 1)),
#             'mod3': np.random.normal(1, 1, (2, 1))
#         },
#         'outcome_effects': true_effects,
#         'outcome_effects_covariates': np.random.normal(1, 1, 1),
#         'missing_prob': 0.2,
#         'missing_mechanism': 'MCAR',
#         'feature_covariates_num': [1, 2],
#         'predictor_covariates_num': 1
#     }
# ]
# print(representative_configs)


# true_effects = {
#             'mod1': np.array([-1, 0, 0, 0, 1]),
#             'mod2': np.array([-1, 0, 0, 0, 1]),
#             'mod3': np.array([-1, 0, 0, 0, 1]),
# }
# representative_configs = [
#     {
#         'n_obs': 100,
#         'p': 10,
#         'modalities': {'mod1': 5, 'mod2': 5, 'mod3': 5},
#         'covariate_effects': {
#             'mod1': np.random.normal(0, 1, (10, 5)),
#             'mod2': np.random.normal(0, 1, (10, 5)),
#             'mod3': np.random.normal(0, 1, (10, 5))
#         },
#         'outcome_effects': true_effects,
#         # 'outcome_effects_covariates': np.random.normal(0, 1, 100),
#         'outcome_effects_covariates': np.zeros(10),
#         'missing_prob': 0.2,
#         'missing_mechanism': 'MCAR',
#         'feature_covariates_num': [2, 3],
#         'predictor_covariates_num': 2
#     }
# ]

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
        config['missing_mechanism'],
        predictor_covariates_num = config['predictor_covariates_num'],
        feature_covariates_num = config['feature_covariates_num']
    )
    simulated_data.summary()


# # Load DML results
# if os.path.exists('scenario_1.pkl'):
#     with open('scenario_1.pkl', 'rb') as f:
#         dml_summary = pickle.load(f)
#         print(f"Loaded Summary: {dml_summary}")
if True:
    # Train Double Machine Learning model using DML_BM
    dml_bm = DML_BM(obj_dml_data=simulated_data, n_folds=5)
    dml_bm.fit()
    print(f"\nSummary:\n{dml_bm.summary}")

    # Save DML results to a file
    with open('scenario_1.pkl', 'wb') as f:
        pickle.dump(dml_bm.summary, f)
    print(f"\nSummary:\n{dml_bm.summary}")
    dml_summary = dml_bm.summary

def visualize_fitting_results_with_true_effects(dml_summary, true_effects):
    # Convert summary and true effects into a DataFrame for easy visualization
    modalities = list(true_effects.keys())
    coef = dml_summary['coef']
    se = dml_summary['se']

    data = []
    for idx, modality in enumerate(modalities):
        n_features = len(true_effects[modality])
        for i in range(n_features):
            data.append({
                'Modality': modality,
                'Feature': f'{modality}_{i}',
                'True Effect': true_effects[modality][i],
                'Estimated Effect': coef[idx*50 + i],
                'SE': se[i]
            })

    results_df = pd.DataFrame(data)

    print(results_df)

    # Plotting using seaborn
    plt.figure(figsize=(16, 10))
    sns.set(style="whitegrid")

    # True Effects
    sns.lineplot(
        data=results_df,
        x='Feature',
        y='True Effect',
        hue='Modality',
        linestyle='--',
        marker='o',
        label='True Effect',
        legend=True
    )

    # Estimated Effects with SE
    for modality, color in zip(modalities, sns.color_palette('tab10', len(modalities))):
        subset_df = results_df[results_df['Modality'] == modality]
        plt.errorbar(
            x=subset_df['Feature'],
            y=subset_df['Estimated Effect'],
            yerr=subset_df['SE'],
            fmt='o',
            capsize=4,
            color=color,
            label=f'Estimated Effect with SE ({modality})'
        )

    plt.xlabel('Feature')
    plt.ylabel('Effect')
    plt.title('True Effects vs. Estimated Effects with Standard Errors')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.savefig('scenario_1.png')
    plt.show()



# Example usage of visualization with true effects
visualize_fitting_results_with_true_effects(dml_summary, true_effects)
