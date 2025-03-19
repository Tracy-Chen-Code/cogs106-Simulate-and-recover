# with help of Chatgpt
import numpy as np
import pandas as pd
import argparse

def simulate_ez_diffusion(a, v, t0, N):
    """ Simulates reaction times and accuracy from the EZ diffusion model. """
    y = np.exp(-a * v)

    # Forward EZ Equations (from slide 13)
    R_pred = 1 / (y + 1)
    M_pred = t0 + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (y + 1)**2)

    # Generate observed summary statistics
    R_obs = np.random.binomial(N, R_pred) / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    V_obs = np.random.gamma((N - 1) / 2, (2 * V_pred) / (N - 1))

    return R_obs, M_obs, V_obs


def recover_parameters(R_obs, M_obs, V_obs):
    """ Recovers model parameters using inverse EZ equations. """
    # Prevent R_obs from being exactly 0 or 1
    R_obs = np.clip(R_obs, 0.001, 0.999)
    
    # Compute log odds (L)
    L = np.log(R_obs / (1 - R_obs))

    # Ensure V_obs is not too small to avoid divide-by-zero errors
    V_obs = max(V_obs, 1e-6)  # Set a lower bound for variance

    # Compute estimated drift rate (v)
    try:
        v_est = np.sign(R_obs - 0.5) * ((L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs)**0.25)
    except ValueError:
        v_est = np.nan  # If sqrt goes negative, assign NaN
        #raise("valueError occur, please do not have Nan")
    
    # Avoid division by zero in a_est
    if np.isnan(v_est) or v_est == 0:
        a_est = np.nan
        #raise("valueError occur, please avoide division by zero")
    else:
        a_est = L / v_est

    # Compute estimated non-decision time (t0)
    if np.isnan(a_est) or np.isnan(v_est):
        t0_est = np.nan
    else:
        t0_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
    #print(a_est, v_est, t0_est)
    return a_est, v_est, t0_est


def simulate_and_recover(N, iterations):
    """ Runs the full simulate-and-recover process for a given N. """
    results = []
    for _ in range(iterations):
        a_true = np.random.uniform(0.5, 2)
        v_true = np.random.uniform(0.5, 2)
        t0_true = np.random.uniform(0.1, 0.5)

        R_obs, M_obs, V_obs = simulate_ez_diffusion(a_true, v_true, t0_true, N)
        a_est, v_est, t0_est = recover_parameters(R_obs, M_obs, V_obs)

        results.append([N, a_true, v_true, t0_true, a_est, v_est, t0_est])

    return pd.DataFrame(results, columns=["N", "a_true", "v_true", "t0_true", "a_est", "v_est", "t0_est"])

def analyze_results(df):
    """ Computes bias and mean squared error (MSE) for recovered parameters. """
    results = []
    for N in df["N"].unique():
        df_N = df[df["N"] == N]

        bias_a = (df_N["a_est"] - df_N["a_true"]).mean()
        bias_v = (df_N["v_est"] - df_N["v_true"]).mean()
        bias_t0 = (df_N["t0_est"] - df_N["t0_true"]).mean()

        mse_a = ((df_N["a_est"] - df_N["a_true"]) ** 2).mean()
        mse_v = ((df_N["v_est"] - df_N["v_true"]) ** 2).mean()
        mse_t0 = ((df_N["t0_est"] - df_N["t0_true"]) ** 2).mean()


        results.append([N, bias_a, bias_v, bias_t0, mse_a, mse_v, mse_t0])

    return pd.DataFrame(results, columns=["N", "Bias_a", "Bias_v", "Bias_t0", "MSE_a", "MSE_v", "MSE_t0"])

def run_simulation(N, iterations=1000):
    """Runs the simulate-and-recover process for a given N and returns averaged results."""
    biases, squared_errors = [], []

    for _ in range(iterations):
        a_true = np.random.uniform(0.5, 2)
        v_true = np.random.uniform(0.5, 2)
        t0_true = np.random.uniform(0.1, 0.5)

        R_obs, M_obs, V_obs = simulate_ez_diffusion(a_true, v_true, t0_true, N)
        a_est, v_est, t0_est = recover_parameters(R_obs, M_obs, V_obs)
        
        bias = np.array([v_true - v_est, a_true - a_est, t0_true - t0_est])
        squared_error = bias ** 2

        biases.append(bias)
        squared_errors.append(squared_error)

    avg_bias = np.nanmean(biases, axis=0)
    avg_squared_error = np.nanmean(squared_errors, axis=0)
    print()

    return avg_bias, avg_squared_error



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()

    df = simulate_and_recover(args.n, args.iterations)
    df.to_csv(f"results/simulated_N{args.n}.csv", index=False)

    df_results = analyze_results(df)
    df_results.to_csv(f"results/summary_N{args.n}.csv", index=False)

    #print(df_results)
    print(df_results.to_string(index=False))


 
    

    

