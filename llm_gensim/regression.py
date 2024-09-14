import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import seaborn as sns
import matplotlib.pyplot as plt

from llm_gensim.analysis import interpret_hexaco_personality
from llm_gensim.generate_sim import sample_hexaco_personality, simulate


save_dir = Path(__file__).parent / "results" / Path(__file__).stem
save_dir.mkdir(parents=True, exist_ok=True)


def analyze_predictors(predictors, observations):
    # Prepare the data
    X = pd.DataFrame(predictors)
    y = pd.DataFrame(observations)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize results dictionary
    results = {}
    
    # Perform Lasso regression for each activity
    for activity in y.columns:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y[activity], test_size=0.2, random_state=42)
        
        # Perform Lasso regression with cross-validation
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_train, y_train)
        
        # Get the coefficients and their absolute values
        coef = pd.Series(lasso.coef_, index=X.columns)
        abs_coef = abs(coef)
        
        # Sort coefficients by absolute value
        sorted_coef = abs_coef.sort_values(ascending=False)
        
        # Store results
        results[activity] = {
            'top_predictors': sorted_coef.index[:2].tolist(),  # Top 2 predictors
            'coefficients': coef.to_dict(),
            'r2_score': lasso.score(X_test, y_test)
        }
    
    return results


def plot_scatter_regression(personalities_df, activity_frequencies_df, personality_factor, activity, save_dir):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=personalities_df[personality_factor], y=activity_frequencies_df[activity])
    plt.title(f'{personality_factor} vs {activity}')
    plt.xlabel(personality_factor)
    plt.ylabel(f'{activity} Frequency')
    plt.tight_layout()
    plt.savefig(save_dir / f'{personality_factor}_{activity}_scatter.png')
    plt.close()


def plot_corr_heatmap(x_df, y_df, x_label, y_label, figsize=(30, 15), dpi=300, save_path=None):
    # Calculate correlation matrix
    corr_matrix = pd.concat([x_df, y_df], axis=1).corr()

    # Select only the correlations between personality factors and activities
    # x_factors = x_df.columns
    # y_factors = y_df.columns
    # correlation_subset = correlation_matrix.loc[x_factors, y_factors].T
    # Select only the correlations between personality factors and activities
    x_factors = x_df.columns
    y_factors = y_df.columns
    corr_subset = corr_matrix.iloc[len(x_factors):, :len(x_factors)]

    # Create heatmap
    plt.figure(figsize=figsize, dpi=dpi)
    palette = sns.diverging_palette(220, 20, sep=30, as_cmap=True)
    abs_max = np.max(np.abs(corr_subset))
    ax = sns.heatmap(corr_subset, xticklabels=True, yticklabels=True, cmap=palette, center=0, vmin=-abs_max, vmax=abs_max)
    # colorbar title (side)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Correlation")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix(".pdf"))
    else:
        plt.show()
    plt.close()


def plot_tsne(data_df, labels, save_path):
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Initialize TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(data_df)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap='viridis')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    from llm_gensim.separate_questions import make_sim_pipeline

    model_id = "claude-3-5-sonnet-20240620"
    # pipeline_name = "hexaco_state-personality_activities-separate_questions"
    # pipeline_name = "hexaco_state-personality_factor_activities-separate_questions"
    pipeline_name = "hexaco_state-free_activities-separate_questions"
    # pipeline_name = "hexaco_state-question_activities-separate_questions"
    batch_name = "batch3"
    
    # num_samples = 500
    # num_sim_steps = 100
    
    num_samples = 100
    num_sim_steps = 500
    # test_ratio = 0.2
    
    pipe_dir = Path(__file__).parent / "results" / "separate_questions" / "stoch_policy" / pipeline_name / model_id / batch_name / "gen"
    # pipe_dir = Path(__file__).parent / "results" / "separate_questions" / pipeline_name / model_id / batch_name / "gen"

    save_dir = save_dir / "stoch_policy" / pipeline_name / model_id / batch_name
    save_dir /= f"num_samples{num_samples}-num_steps{num_sim_steps}"
    save_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = Path(__file__).parent / "configs" / "separate_questions" / (pipeline_name + ".yaml")
    pipeline_config = yaml.safe_load(pipeline_path.read_text())

    sim_pipeline = make_sim_pipeline(model_id, pipeline_config, pipe_dir)

    if not (save_dir / "personalities.csv").exists():
        personalities = []
        activity_frequencies = []
        question_scores = []
        inferred_personalities = []
        for i in range(num_samples):
            rdm_personality = sample_hexaco_personality()

            res = simulate(sim_pipeline, rdm_personality, num_steps=num_sim_steps)
        
            scores = sim_pipeline.eval_score(res.actions)
            inferred_personality = interpret_hexaco_personality(scores)

            personalities.append(rdm_personality)
            activity_frequencies.append(res.average_actions)
            question_scores.append(scores)
            inferred_personalities.append(inferred_personality)

        # personalities: list of dictionaries, each containing personality factors for a simulation
        # activity_frequencies: list of dictionaries, each containing activity frequencies for a simulation
        # question_scores: list of lists, each containing question scores for a simulation
        # inferred_personalities: list of dictionaries, each containing inferred personality factors for a simulation
        personalities_df = pd.DataFrame(personalities)
        activity_frequencies_df = pd.DataFrame(activity_frequencies)
        question_scores_df = pd.DataFrame(question_scores)
        # question index starts from 1
        question_scores_df.columns = [f"Q{i}" for i in range(1, len(question_scores_df.columns) + 1)]
        inferred_personalities_df = pd.DataFrame(inferred_personalities)

        # save datasets
        personalities_df.to_csv(save_dir / "personalities.csv", index=False)
        activity_frequencies_df.to_csv(save_dir / "activity_frequencies.csv", index=False)
        question_scores_df.to_csv(save_dir / "question_scores.csv", index=False)
        inferred_personalities_df.to_csv(save_dir / "inferred_personalities.csv", index=False)
    else:
        personalities_df = pd.read_csv(save_dir / "personalities.csv")
        activity_frequencies_df = pd.read_csv(save_dir / "activity_frequencies.csv")
        question_scores_df = pd.read_csv(save_dir / "question_scores.csv")
        inferred_personalities_df = pd.read_csv(save_dir / "inferred_personalities.csv")

    # MSE between true and inferred personality
    # mse = ((personalities_df - inferred_personalities_df) ** 2).mean()
    # mse_std = ((personalities_df - inferred_personalities_df) ** 2).std()
    # # plot MSE for each factor
    # plt.figure(figsize=(5, 4), dpi=300)
    # plt.plot(personalities_df.columns, mse, marker=".")
    # # plt.fill_between(personalities_df.columns, mse - mse_std, mse + mse_std, alpha=0.2)
    # plt.xlabel("Personality Factors")
    # plt.xticks(rotation=45, ha="right")
    # plt.ylabel("MSE")
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig(save_dir / "mse.png")
    # plt.close()

    # correlation between true and inferred personality
    X = personalities_df.to_numpy()  # n_samples x n_facets
    Y = inferred_personalities_df.to_numpy()
    
    # substract the mean score of each facet across samples (approx equal to the mean in hexaco_data.yaml)

    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    correlations = []
    for i in range(num_samples):
        x = X[i]
        y = Y[i]
        # calculate correlation
        # don't subtract the mean sample-wise (otherwise not able to predict new samples)
        # x = x - x.mean()
        # y = y - y.mean()
        corr = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))
        # corr_np = np.corrcoef(x, y)[0, 1]
        # print(f"Correlation between true and inferred personality for sample {i}: {corr}")
        # print(f"Correlation between true and inferred personality for sample {i}: {corr_np}")
        # print("-" * 100)
        # breakpoint()
        # plt.figure(figsize=(5, 4), dpi=300)
        # plt.title(corr)
        # plt.plot(x)
        # plt.plot(y)
        # plt.show()

        correlations.append(corr)

    print(f"Correlation: {np.mean(correlations)} +/- {np.std(correlations)}")
    # save correlations in json
    with open(save_dir / "metrics.json", "w") as f:
        json.dump({
            "correlations": correlations,
            "correlation_std": np.std(correlations),
            "correlation_mean": np.mean(correlations),
        }, f, indent=4)

    plot_corr_heatmap(activity_frequencies_df, personalities_df, x_label="Activities", y_label="Personality", save_path=save_dir / "activity_vs_personality_corr.png")
    plot_corr_heatmap(activity_frequencies_df, question_scores_df, x_label="Activities", y_label="Question Scores", save_path=save_dir / "score_vs_activity_corr.png")
    plot_corr_heatmap(personalities_df, inferred_personalities_df, x_label="Original Personality", y_label="Recovered Personality", save_path=save_dir / "true_vs_inferred_personality_corr.png", figsize=(7, 6))
    plot_corr_heatmap(question_scores_df, inferred_personalities_df, x_label="Question Scores", y_label="Recovered Personality", save_path=save_dir / "score_vs_inferred_personality_corr.png")
    plot_corr_heatmap(activity_frequencies_df, activity_frequencies_df, x_label="Activities", y_label="Activities", save_path=save_dir / "activity_vs_activity_corr.png")

    # plot t-SNE of activities
    plot_tsne(activity_frequencies_df, activity_frequencies_df.idxmax(axis=1), save_path=save_dir / "tsne_activities.png")

    # results = analyze_predictors(personalities_df, activity_frequencies_df)
    # # save results
    # with open(save_dir / "results.yaml", "w") as f:
    #     yaml.dump(results, f)

    # # Print results
    # _save_dir = save_dir / "top_predictors"
    # # delete old plots
    # for _file in _save_dir.glob("*.png"):
    #     _file.unlink()
    # _save_dir.mkdir(parents=True, exist_ok=True)

    # for activity, data in results.items():
    #     print(f"\nActivity: {activity}")
    #     print(f"Top predictors: {', '.join(data['top_predictors'])}")
    #     print(f"R2 score: {data['r2_score']:.3f}")
    #     print("Coefficients:")
    #     for factor, coef in data['coefficients'].items():
    #         if abs(coef) > 0.01:  # Only print non-zero coefficients
    #             print(f"  {factor}: {coef:.3f}")

    #     # plot regression activity vs best predictors
    #     plt.figure(figsize=(8, 6))
    #     sns.regplot(x=personalities_df[data['top_predictors'][0]], y=activity_frequencies_df[activity])
    #     plt.title(f'{data["top_predictors"][0]} vs {activity}')
    #     plt.xlabel(data["top_predictors"][0])
    #     plt.ylabel(f'{activity} Frequency')
    #     plt.tight_layout()
    #     plt.savefig(_save_dir / f'{activity}_{data["top_predictors"][0]}_scatter.png')
    #     plt.close()
