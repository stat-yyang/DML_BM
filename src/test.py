import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from dataset import DoubleMLData
from DML_BM import DML_BM

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
        'gene_expr2': np.random.randn(n_samples // 2),
        'gene_expr3': np.random.randn(n_samples // 2)
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
        'y': ['age', 'BMI'],
        'gene_expr1': ['age'],
        'gene_expr2': ['BMI'],
        'gene_expr3': ['BMI', 'age'],
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

    # Train Double Machine Learning model using DML_BM
    dml_bm = DML_BM(obj_dml_data=dml_data, n_folds=5)
    dml_bm.fit()
    print(f"\nSummary:\n{dml_bm.summary}")
