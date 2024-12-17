# tests/test_metrics.py
import pytest
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix

def test_additional_metrics():
    y_true = np.array([0,1,1,0,1])
    y_pred = np.array([0,1,0,0,1])
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2,2)
    assert -1.0 <= mcc <= 1.0
