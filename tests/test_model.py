# tests/test_model.py
import pytest
import torch
from kronara.models.mlp import MLP

def test_mlp_forward():
    model = MLP(input_dim=10, hidden_layers=[64,64], output_dim=1)
    x = torch.randn(32, 10)
    y_hat = model(x)
    assert y_hat.shape == (32,1)
