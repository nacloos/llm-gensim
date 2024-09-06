from pathlib import Path
import pandas as pd
import yaml
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt



results_dir = Path(__file__).parent / "results" / "generate_pipeline"
save_dir = Path(__file__).parent / "results" / "eval_results"
save_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    model_name = "claude-3-5-sonnet-20240620"
    batch_name = "batch1"

    legend_names = {
        "pipeline": "LLM-generated state",
        "fixed_hexaco_state": "Hexaco state",
        "fixed_hexaco_state-no_effects": "Hexaco state (no effects)",
        "fixed_hexaco_state-no_effects-action_comments": "Hexaco state (no effects, action comments)",
    }

    results = defaultdict(list)
    # loop over folders in results_dir
    for res in results_dir.iterdir():
        config_name = res.name
        sim_dir = res / model_name / batch_name / "sim"
        # loop over eval folders
        for eval_dir in sim_dir.iterdir():
            eval_name = eval_dir.name
            print(eval_dir)
            metrics_path = eval_dir / "metrics.yaml"
            metrics = yaml.safe_load(metrics_path.read_text())
            print(metrics)

            results["config"].append(config_name)
            results["eval"].append(eval_name)
            results["mae"].append(metrics["mae"])
            results["mse"].append(metrics["mse"])
            results["pearsonr"].append(metrics["pearsonr"])

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / "results.csv", index=False)

    # plot MSE
    plt.figure(figsize=(5, 4), dpi=150)
    sns.lineplot(x="eval", y="mse", hue="config", data=results_df, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Evaluation")
    plt.ylabel("MSE")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # remove legend
    ax.legend().remove()
    plt.ylim(0, None)

    # separate legend
    legend_handles = ax.get_legend_handles_labels()[0]
    legend_labels = [legend_names[h.get_label()] for h in legend_handles]
    plt.tight_layout()
    plt.savefig(save_dir / "mse.png")

    # plot legend
    plt.figure(figsize=(4, 2), dpi=150)
    plt.legend(legend_handles, legend_labels)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_dir / "legend.png")

    # plot MAE
    plt.figure(figsize=(5, 4), dpi=150)
    sns.lineplot(x="eval", y="mae", hue="config", data=results_df, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Evaluation")
    plt.ylabel("MAE")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend().remove()
    plt.ylim(0, None)
    # remove legend
    plt.tight_layout()
    plt.savefig(save_dir / "mae.png")

