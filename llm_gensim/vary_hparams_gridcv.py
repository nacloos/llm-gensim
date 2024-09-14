from collections import defaultdict
import copy
from functools import partial
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin




class PersonalitySimulator(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline, personality_keys, beta=5, num_steps=500, rng=None):
        # make a copy of pipeline
        self.pipeline = pipeline
        self.beta = beta
        self.num_steps = num_steps
        self.rng = rng
        self.personality_keys = personality_keys

    def _simulate(self, x):
        assert x.ndim == 1, "Input must be 1D"
        personality = {k: float(v) for k, v in zip(self.personality_keys, x)}
        self.pipeline.policy = partial(self.pipeline.policy, beta=self.beta, rng=self.rng)
        res = simulate(self.pipeline, personality, num_steps=self.num_steps)
        scores = self.pipeline.eval_score(res.actions)
        recovered_personality = interpret_hexaco_personality(scores)
        
        y = np.array(list(recovered_personality.values()))
        return y

    def fit(self, X, y):
        Y = []
        for x in X:
            y = self._simulate(x)
            Y.append(y)
        Y = np.array(Y)

        # compute the mean on train data
        self.X_mean = X.mean(axis=0)
        self.Y_mean = Y.mean(axis=0)
        return self

    def score(self, X, y=None):
        correlations = []
        for x in X:
            y = self._simulate(x)

            x = x - self.X_mean
            y = y - self.Y_mean
            corr = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))
            correlations.append(corr)
        return np.mean(correlations)


def hparams_grid_search(pipeline_path, num_runs=10, seed=0, save_dir=None):
    pipeline = load_pipeline(pipeline_path)
    rng = np.random.default_rng(seed) if seed is not None else None

    param_grid = {
        'beta': [1, 5, 10, 15, 20],
        'num_steps': [20, 100, 300, 500, 800, 1000]
    }

    personality_keys = list(sample_hexaco_personality().keys())
    estimator = PersonalitySimulator(pipeline, personality_keys=personality_keys, rng=rng)

    grid_search = GridSearchCV(estimator, param_grid, cv=num_runs, n_jobs=-1, verbose=4)

    X = [sample_hexaco_personality() for _ in range(num_runs)]
    X = np.array([list(x.values()) for x in X])
    y = [0] * num_runs  # Dummy y values

    grid_search.fit(X, y)

    results = grid_search.cv_results_

    # Plot results
    plt.figure(figsize=(5, 3), dpi=300)
    for i, beta in enumerate(param_grid['beta']):
        mask = results['param_beta'] == beta
        plt.errorbar(results['param_num_steps'][mask], 
                     results['mean_test_score'][mask],
                     yerr=results['std_test_score'][mask],
                     marker='.', label=f'beta={beta}')

    plt.xlabel('Number of simulation steps')
    plt.ylabel('Correlation between\n original and recovered personality')
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "grid_search_results.png")
    plt.close()

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    return grid_search.best_params_, grid_search.best_score_


def plot_vary_hparams(pipeline_path, num_runs=10, seed=0):
    pipeline = load_pipeline(pipeline_path)

    rng = np.random.default_rng(seed) if seed is not None else None
    pipeline.policy = partial(pipeline.policy, beta=5, rng=rng)

    personality = sample_hexaco_personality()

    plt.figure()
    plt.plot(personality.keys(), personality.values(), label="Original", color="black", linewidth=2)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for i, num_steps in enumerate([10, 100, 500, 1000]):
        recoverred_personalities = []

        for run in range(num_runs):
            res = simulate(pipeline, personality, num_steps=num_steps)
            scores = pipeline.eval_score(res.actions)
            recoverred_presonality = interpret_hexaco_personality(scores)
            recoverred_personalities.append(recoverred_presonality)

        avg_recoverred_personality = {k: np.mean([p[k] for p in recoverred_personalities]) for k in recoverred_personalities[0]}
        std_recoverred_personality = {k: np.std([p[k] for p in recoverred_personalities]) for k in recoverred_personalities[0]} 

        plt.plot(avg_recoverred_personality.keys(), avg_recoverred_personality.values(), label=f"Recovered, {num_steps} steps", color=colors[i])
        plt.fill_between(avg_recoverred_personality.keys(), 
                            [avg_recoverred_personality[k] - std_recoverred_personality[k] for k in avg_recoverred_personality.keys()],
                            [avg_recoverred_personality[k] + std_recoverred_personality[k] for k in avg_recoverred_personality.keys()],
                            alpha=0.2, color=colors[i])

    plt.xticks(rotation=90, fontsize=6, ha="right")
    plt.legend()
    plt.show()

    # same but vary beta
    num_steps = 500
    personality = sample_hexaco_personality()

    plt.figure()
    plt.plot(personality.keys(), personality.values(), label="Original", color="black", linewidth=2)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for i, beta in enumerate([0.1, 1, 5, 10, 15]):
        recoverred_personalities = []
        pipeline.policy = partial(pipeline.policy, beta=beta, rng=rng)

        for run in range(num_runs):
            res = simulate(pipeline, personality, num_steps=num_steps)
            scores = pipeline.eval_score(res.actions)
            recoverred_presonality = interpret_hexaco_personality(scores)
            recoverred_personalities.append(recoverred_presonality)

        avg_recoverred_personality = {k: np.mean([p[k] for p in recoverred_personalities]) for k in recoverred_personalities[0]}
        std_recoverred_personality = {k: np.std([p[k] for p in recoverred_personalities]) for k in recoverred_personalities[0]} 

        plt.plot(avg_recoverred_personality.keys(), avg_recoverred_personality.values(), label=f"Recovered, beta={beta}", color=colors[i])
        plt.fill_between(avg_recoverred_personality.keys(), 
                            [avg_recoverred_personality[k] - std_recoverred_personality[k] for k in avg_recoverred_personality.keys()],
                            [avg_recoverred_personality[k] + std_recoverred_personality[k] for k in avg_recoverred_personality.keys()],
                            alpha=0.2, color=colors[i])

    plt.xticks(rotation=90, fontsize=6, ha="right")
    plt.legend()
    plt.show()
