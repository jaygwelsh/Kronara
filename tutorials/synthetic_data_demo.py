# tutorials/synthetic_data_demo.py
import numpy as np
import pandas as pd

def generate_synthetic_data(n=10000, d=200, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d)
    score = X[:,:10].sum(axis=1) + np.sin(X[:,10:20]).sum(axis=1) + np.random.randn(n)*0.5
    y = (score>0).astype('float32')
    flip_mask = np.random.rand(n)<0.005
    y[flip_mask] = 1 - y[flip_mask]
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(d)])
    df["label"] = y
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("synthetic_data.csv", index=False)
    print("Synthetic dataset generated and saved to synthetic_data.csv")
