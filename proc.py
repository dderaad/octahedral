import pandas as pd
import numpy as np
import sys, os
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest

PATH = r"data"

def process_data():
    used_seeds = set()
    datasets = []
    for i, f in enumerate(os.listdir(rf"{PATH}/")):
        if f == ".dataignore":
            continue
        r = pd.read_pickle(rf"{PATH}/{f}")
        datasets.append(r)

    mass = pd.concat(datasets).reset_index(drop=True)
    
    viliam_data = (1139883093, 3_000_000_000)
    mass.loc[len(mass)] = (viliam_data[0], -1, viliam_data[1])
    #viliam_data = (0, 0)

    successes, trials = np.sum(mass["regular"]), np.sum(mass["N"])
    p = successes / trials
    print(f"{successes=:,}\n{trials=:,}")

    alpha = 0.05
    print(proportion_confint(successes, trials, alpha=alpha, method='beta'))
    print(binomtest(successes, trials).proportion_ci(confidence_level=1-alpha))

    print(f"{p=}")

    return mass
    

if __name__ == "__main__":
    process_data()