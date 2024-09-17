from collections import defaultdict
import copy
from functools import partial
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_gensim.analysis import interpret_hexaco_personality
from llm_gensim.generate_sim import sample_hexaco_personality, simulate
from llm_gensim.separate_questions import load_pipeline


save_dir = Path(__file__).parent / "results" / "vary_hparams"
save_dir.mkdir(parents=True, exist_ok=True)



def plot_correlations_num_steps_beta(results, save_dir):
    betas = results["beta"]
    num_steps = results["num_steps"]
    means = results["mean_correlation"]
    stds = results["std_correlation"]
    
    # Create a scatter plot with error bars
    plt.figure(figsize=(4, 3), dpi=300)
    for i, beta in enumerate(set(betas)):
        mask = np.array(betas) == beta
        plt.errorbar(np.array(num_steps)[mask], np.array(means)[mask], 
                     yerr=np.array(stds)[mask], marker='o', label=f'beta={beta}')
    
    plt.xlabel('Number of simulation steps')
    plt.ylabel('Correlation between original\nand recovered personalities')
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "num_steps_beta_correlations.png")
    plt.savefig(save_dir / "num_steps_beta_correlations.pdf")
    plt.close()


def compute_correlation(original_personalities: list[dict], recovered_personalities: list[dict]):
    X = np.array([list(p.values()) for p in original_personalities])
    Y = np.array([list(p.values()) for p in recovered_personalities])
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    correlations = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        corr = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))

        if np.isnan(corr):
            # TODO: sometimes y is all zeros
            corr = 0

        correlations.append(corr)
    return correlations


def evaluate_pipeline(pipeline, beta, num_steps, num_samples, return_data=False, rng=None):
    pipeline = copy.deepcopy(pipeline)
    pipeline.policy = partial(pipeline.policy, beta=beta, rng=rng)

    personalities = []
    recovered_personalities = []
    score_results = []
    sim_results = []
    for _ in range(num_samples):
        personality = sample_hexaco_personality()
        res = simulate(pipeline, personality, num_steps=num_steps)
        scores = pipeline.eval_score(res.actions)
        recovered_personality = interpret_hexaco_personality(scores)

        personalities.append(personality)
        recovered_personalities.append(recovered_personality)
        score_results.append(scores)
        sim_results.append(res)

    correlations = compute_correlation(personalities, recovered_personalities)

    if not return_data:
        return correlations
    else:
        return {
            "correlations": correlations,
            "personalities": personalities,
            "recovered_personalities": recovered_personalities,
            "scores": score_results,
            "activities": [res.actions for res in sim_results]
        }


def eval_vary_hparams(pipeline_path, num_samples=50, seed=0, save_dir=None):
    save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(pipeline_path)
    rng = np.random.default_rng(seed) if seed is not None else None
    pipeline.policy = partial(pipeline.policy, beta=5, rng=rng)

    # Initialize lists to store results
    # num_steps_list = [10, 100, 300, 500, 800, 1000]
    # beta_list = [0.1, 1, 5, 10, 15, 20]

    num_steps_list = [100, 500, 1000]
    # beta_list = [1, 5, 10, 15]
    beta_list = [1, 5, 15, 30]


    results = defaultdict(list)
    for num_steps in num_steps_list:
        for beta in beta_list:
            correlations = evaluate_pipeline(pipeline, beta=beta, num_steps=num_steps, num_samples=num_samples, rng=rng)

            mean_correlation = np.mean(correlations)
            std_correlation = np.std(correlations)  # This is always non-negative
            results["num_steps"].append(num_steps)
            results["beta"].append(beta)
            results["mean_correlation"].append(mean_correlation)
            results["std_correlation"].append(std_correlation)

    plot_correlations_num_steps_beta(results, save_dir)

    # compute best hparams
    best_hparams = {
        "num_steps": results["num_steps"][np.argmax(results["mean_correlation"])],
        "beta": results["beta"][np.argmax(results["mean_correlation"])]
    }
    return results, best_hparams



if __name__ == "__main__":
    pipeline_dir = Path(__file__).parent / "results" / "separate_questions" / "stoch_policy"

    config_names = [
        "hexaco_state-personality_activities-separate_questions",
        # "hexaco_state-personality_factor_activities-separate_questions",
        "hexaco_state-free_activities-separate_questions",
        "hexaco_state-question_activities-separate_questions"
    ]
    model_name = "claude-3-5-sonnet-20240620"
    batch_names = [
        "batch1",
        "batch2",
        "batch3"
    ]

    num_fit_samples = 10
    num_test_samples = 100
    eval_save_dir = save_dir / "eval_hparams" / "stoch_policy"
    fit_save_dir = save_dir / "fit_hparams" / "stoch_policy"

    results = defaultdict(list)
    for config_name in config_names:
        for batch_name in batch_names:
            _save_dir = fit_save_dir / config_name / model_name / batch_name

            if (_save_dir / "best_values.json").exists():
                print(f"Load best values from {_save_dir / 'best_values.json'}")
                # load the best values
                with open(_save_dir / "best_values.json", "r") as f:
                    best_hparams = json.load(f)
            else:
                pipeline_path = pipeline_dir / config_name / model_name / batch_name / "gen"
                res, best_hparams = eval_vary_hparams(pipeline_path, num_samples=num_fit_samples, seed=0, save_dir=_save_dir)
                with open(_save_dir / "best_values.json", "w") as f:
                    json.dump(best_hparams, f)

            results["config"].append(config_name)
            results["batch"].append(batch_name)
            results["model"].append(model_name)
            results["num_steps"].append(best_hparams["num_steps"])
            results["beta"].append(best_hparams["beta"])
            results["num_samples"].append(num_fit_samples)

    best_hparams_df = pd.DataFrame(results)
    best_hparams_df.to_csv(fit_save_dir / "best_hparams.csv", index=False)



    # evaluate with the best hparams
    eval_results = defaultdict(list)
    for config_name in config_names:
        for batch_name in batch_names:
            _save_dir = eval_save_dir / config_name / model_name / batch_name
            _save_dir.mkdir(parents=True, exist_ok=True)

            if (_save_dir / "sim_results.json").exists():
                print(f"Load eval sim results from {_save_dir / 'sim_results.json'}")
                with open(_save_dir / "sim_results.json", "r") as f:
                    res = json.load(f)

                # load best hparams
                with open(_save_dir / "hparams.json", "r") as f:
                    best_hparams = json.load(f)
                beta = best_hparams["beta"]
                num_steps = best_hparams["num_steps"]
            else:
                # retrieve best hparams
                best_hparams = best_hparams_df[(best_hparams_df["config"] == config_name) & (best_hparams_df["batch"] == batch_name)]
                assert len(best_hparams) == 1
                best_hparams = best_hparams.iloc[0]
                beta = float(best_hparams["beta"])
                num_steps = int(best_hparams["num_steps"])
                
                # simulate
                pipeline = load_pipeline(pipeline_dir / config_name / model_name / batch_name / "gen")
                
                res = evaluate_pipeline(
                    pipeline, beta=beta, num_steps=num_steps, num_samples=num_test_samples, return_data=True
                )
                # save sim results as json
                with open(_save_dir / "sim_results.json", "w") as f:
                    json.dump(res, f, indent=4)

                # save best hparams
                with open(_save_dir / "hparams.json", "w") as f:
                    json.dump({"beta": beta, "num_steps": num_steps}, f, indent=4)

            eval_results["config"].append(config_name)
            eval_results["batch"].append(batch_name)
            eval_results["model"].append(model_name)
            eval_results["num_steps"].append(num_steps)
            eval_results["beta"].append(beta)
            eval_results["mean_correlation"].append(np.mean(res["correlations"]))
            eval_results["std_correlation"].append(np.std(res["correlations"]))
            eval_results["num_samples"].append(num_test_samples)
    eval_results_df = pd.DataFrame(eval_results)
    eval_results_df.to_csv(eval_save_dir / "eval_results.csv", index=False)


    # bar plot
    legend_names = {
        "hexaco_state-personality_activities-separate_questions": "Personality\ntraits",
        "hexaco_state-question_activities-separate_questions": "Questions",
        "hexaco_state-free_activities-separate_questions": "None",
        # "hexaco_state-personality_factor_activities-separate_questions": "Personality\nfactors",
    }
    colors = {
        "hexaco_state-free_activities-separate_questions": "yellowgreen",
        "hexaco_state-question_activities-separate_questions": "coral",
        "hexaco_state-personality_activities-separate_questions": "cornflowerblue",
    }

    plt.figure(figsize=(4, 3), dpi=300)

    for i, config in enumerate(legend_names.keys()):
        # scatter plot with mean and error bar with non-random jitter
        means = eval_results_df[eval_results_df["config"] == config]["mean_correlation"]
        stds = eval_results_df[eval_results_df["config"] == config]["std_correlation"]
        jitter = np.linspace(-0.15, 0.15, len(means))
        plt.errorbar(i + jitter, means, yerr=stds, capsize=5, color=colors[config], ecolor=colors[config], marker="o",  ls="none")
        # plt.scatter(i + jitter, means, label=config, color=colors[config], zorder=10)

    plt.ylabel("Correlation between original\nand recovered personalities")
    x_labels = [legend_names[p] for p in legend_names.keys()]
    plt.xticks(np.arange(len(x_labels)), x_labels)
    plt.xlabel("Contextual information")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(eval_save_dir / "eval_correlations_plot.png")
    plt.savefig(eval_save_dir / "eval_correlations_plot.pdf")
    plt.close()

