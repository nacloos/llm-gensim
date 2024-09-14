from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict



results_dir = Path(__file__).parent / "results" / "regression" / "stoch_policy"

save_dir = Path(__file__).parent / "results" / "eval_results_regression"
save_dir.mkdir(parents=True, exist_ok=True)


legend_names = {
    "hexaco_state-free_activities-separate_questions": "Free\nactivities",
    "hexaco_state-personality_factor_activities-separate_questions": "Personality\nfactors",
    "hexaco_state-question_activities-separate_questions": "Questions",
    "hexaco_state-personality_activities-separate_questions": "Personality\nsubfactors",
}

if __name__ == "__main__":
    model_name = "claude-3-5-sonnet-20240620"
    batch_name = "batch1"
    num_samples = 50

    results = defaultdict(list)
    for p in results_dir.iterdir():
        pipeline_name = p.name
        metrics_path = p / model_name / batch_name / f"num_samples{num_samples}" / "metrics.json"
        print(metrics_path)
        metrics = json.loads(metrics_path.read_text())
        print(metrics["correlation_mean"])

        results["pipeline"].append(pipeline_name)
        results["correlation_mean"].append(metrics["correlation_mean"])
        results["correlation_std"].append(metrics["correlation_std"])

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / "results.csv", index=False)
    
    # order pipelines by correlation mean
    results_df = results_df.sort_values(by="correlation_mean", ascending=False)

    # bar plot
    plt.figure(figsize=(5, 4), dpi=300)
    plt.bar(results_df["pipeline"], results_df["correlation_mean"], yerr=results_df["correlation_std"], capsize=5, color="thistle", ecolor="#636363")
    plt.ylabel("Correlation between\noriginal and recovered personalities")
    x_labels = [legend_names[p] for p in results_df["pipeline"]]
    plt.xticks(results_df["pipeline"], x_labels)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "bar_plot.png")
    plt.savefig(save_dir / "bar_plot.pdf")
    plt.close()