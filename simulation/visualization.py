import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_fitting_results_with_true_effects(dml_summary, true_effects, fig_file):
    # Convert summary and true effects into a DataFrame for easy visualization
    modalities = list(true_effects.keys())
    coef = dml_summary['coef']
    se = dml_summary['se']

    data = []
    for idx, modality in enumerate(modalities):
        n_features = len(true_effects[modality])
        added = 0
        for i in range(n_features):
            data.append({
                'Modality': modality,
                'Feature': f'{modality}_{i}',
                'True Effect': true_effects[modality][i],
                'Estimated Effect': coef[added + i],
                'SE': se[i]
            })
        added += true_effects[modality].shape[0]

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
    plt.savefig(fig_file)
    plt.show()




def visualize_multiple_fitting_results_with_true_effects(dml_summaries, true_effects, fig_file):
    # Convert summary and true effects into a DataFrame for easy visualization
    modalities = list(true_effects.keys())
    data = []

    for idx, modality in enumerate(modalities):
        n_features = len(true_effects[modality])
        for i in range(n_features):
            data.append({ 
                'Modality': modality,
                'Feature': f'{modality}_{i}',
                'True Effect': true_effects[modality][i],
                'Estimated Effect': true_effects[modality][i],
                'Type': 'True'
            })

    for dml_summary in dml_summaries:
        coef = dml_summary['coef']

        for idx, modality in enumerate(modalities):
            n_features = len(true_effects[modality])
            added = 0
            for i in range(n_features):
                data.append({
                    'Modality': modality,
                    'Feature': f'{modality}_{i}',
                    'True Effect': true_effects[modality][i],
                    'Estimated Effect': coef[added + i],
                    'Type': 'Estimated'
                })
                
            added += true_effects[modality].shape[0]
            added += true_effects[modality].shape[0]

    results_df = pd.DataFrame(data)
    print(results_df)

    # Plotting using seaborn
    plt.figure(figsize=(16, 10))
    sns.set(style="whitegrid")

    # Plotting True Effects
    sns.scatterplot(
        data=results_df[results_df['Type'] == 'True'],
        x='Feature',
        y='Estimated Effect',
        hue='Modality',
        marker='o',
        s=100,
        label='True Effect',
        legend=True,
        zorder=3  # Ensure true effect points are on top
    )

    # Plotting Estimated Effects with Violin Plot
    sns.violinplot(
        data=results_df[results_df['Type'] == 'Estimated'],
        x='Feature',
        y='Estimated Effect',
        hue='Modality',
        inner="stick",
        palette='tab10',
        dodge=False,
        cut=0
    )

    plt.xlabel('Feature')
    plt.ylabel('Effect')
    plt.title('True Effects vs. Estimated Effects')
    plt.xticks(rotation=90)
    plt.ylim([-5, 5])
    plt.tight_layout()
    plt.legend()
    plt.savefig(fig_file)
    plt.show()

    return results_df