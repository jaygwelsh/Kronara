# tutorials/breast_cancer_data_demo.py
from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

if __name__ == "__main__":
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    df = X.copy()
    df['label'] = y
    output_path = os.path.join(os.path.dirname(__file__), "breast_cancer_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Breast cancer dataset saved as {output_path}")
