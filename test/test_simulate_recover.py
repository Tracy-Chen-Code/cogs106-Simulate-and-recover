# with help of Chatgpt
import unittest
import numpy as np
import pandas as pd

import sys
import os

# Add the parent directory of src to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from simulate_recover import simulate_ez_diffusion, recover_parameters, simulate_and_recover, analyze_results
#from src.simulate_recover import simulate_ez_diffusion, recover_parameters, simulate_and_recover, analyze_results


class TestEZDiffusionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Runs once before all tests. """
        np.random.seed(42)  # Ensure reproducibility

    def test_simulation(self):
        """ Test if simulated values are within expected ranges. """
        R, M, V = simulate_ez_diffusion(1.0, 1.0, 0.2, 100)
        self.assertTrue(0 <= R <= 1, "Simulated accuracy is out of bounds.")
        self.assertGreater(M, 0, "Mean reaction time should be positive.")
        self.assertGreater(V, 0, "Variance should be positive.")

    def test_recovery(self):
        """ Test if estimated parameters are finite and reasonable. """
        a_est, v_est, t0_est = recover_parameters(0.7, 0.5, 0.02)
        self.assertTrue(np.isfinite([a_est, v_est, t0_est]).all(), "Recovered parameters should be finite.")
        self.assertGreater(a_est, 0, "Recovered boundary separation should be positive.")
        self.assertGreaterEqual(t0_est, 0, "Recovered non-decision time should be non-negative.")

    def test_large_N_simulation(self):
        """ Test if simulation runs efficiently for large N values. """
        N = 4000  # Large trial number
        iterations = 5  # Keep it small for speed
        df = simulate_and_recover(N, iterations)
        
        self.assertEqual(len(df), iterations, "Simulated data should have the correct number of rows.")
        self.assertTrue(df["a_true"].between(0.5, 2).all(), "Boundary separation (a) out of range.")
        self.assertTrue(df["v_true"].between(0.5, 2).all(), "Drift rate (v) out of range.")
        self.assertTrue(df["t0_true"].between(0.1, 0.5).all(), "Non-decision time (t0) out of range.")

    def test_bias_mse_computation(self):
        """ Test if bias and mean squared error (MSE) calculations are reasonable. """
        N = 10
        iterations = 5
        df_simulated = simulate_and_recover(N, iterations)
        df_results = analyze_results(df_simulated)

        self.assertEqual(len(df_results), 1, "Bias analysis should have one row per N.")
        self.assertFalse(df_results[["Bias_a", "Bias_v", "Bias_t0", "MSE_a", "MSE_v", "MSE_t0"]].isnull().values.any(), "Bias or MSE contains NaN.")
        self.assertTrue((df_results[["MSE_a", "MSE_v", "MSE_t0"]] >= 0).all().all(), "Mean squared error (MSE) should be non-negative.")

    def test_edge_cases(self):
        """ Test handling of extreme accuracy values (near 0 or 1). """
        a_est, v_est, t0_est = recover_parameters(0.99, 0.5, 0.02)  # Very high accuracy
        self.assertTrue(np.isfinite([a_est, v_est, t0_est]).all(), "Recovered parameters should be finite.")
        self.assertGreater(a_est, 0, "Recovered boundary separation should be positive.")

        a_est, v_est, t0_est = recover_parameters(0.01, 0.5, 0.02)  # Very low accuracy
        self.assertTrue(np.isfinite([a_est, v_est, t0_est]).all(), "Recovered parameters should be finite.")

if __name__ == "__main__":
    unittest.main()

