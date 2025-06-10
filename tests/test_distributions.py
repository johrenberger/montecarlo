import importlib.util
import numpy as np
import pathlib

# Dynamically load module from file path
module_path = pathlib.Path(__file__).resolve().parent.parent / 'montecarlo' / 'montecarlo-selection.py'
spec = importlib.util.spec_from_file_location("montecarlo_selection", module_path)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

def test_normal_distribution_weights_sum_to_one():
    data = [10, 20, 30, 40, 50]
    weights, _ = mc.calculate_normal_distribution_weight(data)
    assert np.isclose(np.sum(weights), 1.0)

def test_lognormal_distribution_weights_sum_to_one():
    data = [10, 20, 30, 40, 50]
    weights, _ = mc.calculate_lognormal_distribution_weight(data)
    assert np.isclose(np.sum(weights), 1.0)

def test_gamma_distribution_weights_sum_to_one():
    data = [10, 20, 30, 40, 50]
    weights, _ = mc.calculate_gamma_distribution_weight(data)
    assert np.isclose(np.sum(weights), 1.0)

def test_beta_distribution_weights_sum_to_one():
    data = [10, 20, 30, 40, 50]
    weights, _ = mc.calculate_beta_distribution_weight(data)
    assert np.isclose(np.sum(weights), 1.0)
