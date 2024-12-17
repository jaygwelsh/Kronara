# Kronara

Kronara is a production-ready machine learning training pipeline originally demonstrated with synthetic data, now fully enhanced with real-world data validation, calibration, interpretability, and robust logging. It uses PyTorch Lightning, Hydra, MLflow, and a variety of data science libraries.

## Key Features

- **Name & Identity:** Project is now named "Kronara," a unique, primordial-sounding name representing foundational strength.
- **Data Handling:** Supports both synthetic and real-world (e.g., Breast Cancer Wisconsin) datasets.
- **Robust Model:** Large MLP with advanced regularization, early stopping, and OneCycleLR scheduling.
- **Calibration & Threshold Optimization:** Automated threshold selection based on F1. Reliability diagrams for calibration checks.
- **Interpretability:** SHAP-based feature importance analysis.
- **Cross-Validation & Ensembling:** K-fold splits and ensemble predictions for improved generalization.
- **Artifacts Directory:** All `.pt` prediction files, reliability diagrams, and other outputs saved to `artifacts/` at the project root.
- **Logging & Monitoring:** Structured logging with Loguru, MLflow experiment tracking, and CI-ready testing framework.

## Setup Instructions

1. **Create Virtual Environment & Install:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Create Artifacts Directory:**
   ```bash
   mkdir artifacts
   ```

3. **Run Synthetic Data (Default):**
   ```bash
   python scripts/train.py
   ```
   This trains using synthetic data and saves results to artifacts/.

4. **Real-World Data (Breast Cancer):**
   ```bash
   python tutorials/breast_cancer_data_demo.py
   python scripts/train.py data.path=./tutorials/breast_cancer_data.csv data.fallback_to_synthetic=false
   ```

5. **Cross-Validation & Ensembling:**
   ```bash
   python scripts/benchmark.py
   python scripts/ensemble.py
   ```
   Results and `.pt` files appear in artifacts/.

6. **Calibration & Evaluation:**
   ```bash
   python -m kronara.evaluate
   ```

7. **Feature Importance:**
   ```bash
   python tutorials/feature_importance_demo.py
   ```

8. **Hyperparameter Tuning:**
   ```bash
   python scripts/tune_hparams.py
   ```

9. **Tests:**
   ```bash
   pytest tests
   ```

## Usage Examples

### Single Fold Training with Synthetic Data:
```bash
python scripts/train.py
```

### Training on Real-World Data:
```bash
python tutorials/breast_cancer_data_demo.py
python scripts/train.py data.path=./tutorials/breast_cancer_data.csv data.fallback_to_synthetic=false
```

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes with clear commit messages.
4. Submit a pull request with a detailed explanation of changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 📊 Performance Metrics

The Kronara model was trained and evaluated on a synthetic dataset designed to simulate real-world scenarios. Below are the details of the dataset and the resulting performance metrics:

### 🗃️ Dataset Details
- **Number of Samples:** 1,000  
- **Number of Features:** 20  
- **Dataset Type:** Synthetic Binary Classification  

### 🧠 Model Architecture
- **Model Type:** Multi-Layer Perceptron (MLP)  
- **Number of Parameters:** 51.2 Million  
- **Model Size:** Approximately 204.8 MB  
- **Training Framework:** PyTorch Lightning  

### 📈 Training Results

| Metric       | Value    |
|--------------|----------|
| Accuracy     | 91.08%   |
| AUC          | 0.9714   |
| F1 Score     | 0.9094   |
| Precision    | 92.34%   |
| Recall       | 89.58%   |
| Loss         | 0.2256   |

### 🔍 Interpretation

- **Accuracy (91.08%)**: The model correctly classified approximately 91% of the instances in the synthetic dataset.  
- **AUC (0.9714)**: An Area Under the Receiver Operating Characteristic Curve (AUC) of 0.9714 indicates excellent discriminative ability, meaning the model effectively distinguishes between the two classes.  
- **F1 Score (0.9094)**: The F1 score balances precision and recall, reflecting a high level of performance in identifying true positives while minimizing false positives and false negatives.  
- **Precision (92.34%)**: This metric shows that when the model predicts a positive class, it is correct 92.34% of the time.  
- **Recall (89.58%)**: The model successfully identifies 89.58% of all actual positive instances, demonstrating its effectiveness in capturing relevant data points.  
- **Loss (0.2256)**: The Binary Cross-Entropy loss value indicates the error between the predicted probabilities and the actual labels. A lower loss signifies better model performance.  
