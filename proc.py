import pandas as pd
import numpy as np
import sys, os
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest
from utils import first_n_digits

PATH = r"data"
class text_colors:
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'

def process_data():
    used_seeds = set()
    datasets = []
    for i, f in enumerate(os.listdir(rf"{PATH}/")):
        if f == ".dataignore":
            continue

        gpu = "gpu" in f
        r = pd.read_pickle(rf"{PATH}/{f}")
        r["gpu"] = gpu
        datasets.append(r)

    mass = pd.concat(datasets).reset_index(drop=True)
    local_cpu = mass.loc[~mass["gpu"]]
    local_gpu = mass.loc[mass["gpu"]]
    
    viliam_data = pd.DataFrame([[2_374_692_793, -1, 6_250_000_000, False]], columns=[*mass.columns])
    mass = pd.concat([mass, viliam_data])

    successes, trials = np.sum(mass["regular"]), np.sum(mass["N"])
    s_cpu, t_cpu = np.sum(local_cpu["regular"]), np.sum(local_cpu["N"])
    s_gpu, t_gpu = np.sum(local_gpu["regular"]), np.sum(local_gpu["N"])
    p = successes / trials
    print(f"total {successes=:,} local: (CPU: {s_cpu:,}, GPU: {s_gpu:,})\ntotal {trials=:,} local: (CPU: {t_cpu:,}, GPU: {t_gpu:,})")

    alpha = 0.05
    sm_CI = proportion_confint(successes, trials, alpha=alpha, method='beta')
    scp_CI = binomtest(successes, trials).proportion_ci(confidence_level=1-alpha)._asdict()

    agreement = first_n_digits(*sm_CI) 
    width = sm_CI[1] - sm_CI[0]

    print(f"statsmodels.stats.proportion.proportion_confint: {sm_CI}")
    print(f"scipy.stats.binomtest: ({alpha=})              ({scp_CI["low"]}, {scp_CI["high"]})")
    print(f"CI Width: {width:.16E}")
    print(f"Standard Error: {np.sqrt(p * (1 - p) / trials)}")

    str_point_estimate = str(float(p))
    certain_estimate, uncertain_estimate = str_point_estimate[:agreement], str_point_estimate[agreement:]

    print(f"Point Estimate: {text_colors.OKGREEN}{certain_estimate}{text_colors.ENDC}{uncertain_estimate}")

    return mass
    

if __name__ == "__main__":
    process_data()