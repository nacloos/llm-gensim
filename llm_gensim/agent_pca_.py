from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

from llm_gensim.generate_sim import sample_hexaco_personality, simulate
from llm_gensim.separate_questions import load_pipeline
from llm_gensim.analysis import interpret_hexaco_personality, set_personality_factor


save_dir = Path(__file__).parent / "results" / "agent_pca"
save_dir.mkdir(parents=True, exist_ok=True)

def extraversion_pca(pipeline, num_samples, num_steps):
    def _simulate(personality, num_samples, num_steps):
        action_freqs = []
        for _ in range(num_samples):
            res = simulate(pipeline, personality, num_steps=num_steps)
            action_freq = [res.actions.count(action) / num_steps for action in pipeline.actions]
            action_freqs.append(action_freq)
        return action_freqs

    personality = sample_hexaco_personality()
    personality = {k: 3 for k in personality.keys()}
    personality_low = set_personality_factor(personality, "Extraversion", 1)
    personality_high = set_personality_factor(personality, "Extraversion", 5)

    action_freqs_low = _simulate(personality_low, num_samples, num_steps)
    action_freqs_high = _simulate(personality_high, num_samples, num_steps)
    action_freqs = np.array(action_freqs_low + action_freqs_high)
    print(action_freqs.shape)

    action_freqs = np.array(action_freqs)
    print(action_freqs.shape)

    pca = PCA(n_components=None)
    pca.fit(action_freqs)
    action_freqs_pca = pca.transform(action_freqs)

    plt.figure()
    # color by extraversion
    colors = [1 if i < num_samples else 5 for i in range(num_samples * 2)]
    plt.scatter(action_freqs_pca[:, 0], action_freqs_pca[:, 1], c=colors)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.show()


def personalities_pca(pipeline, num_samples, num_steps, save_dir):
    def _simulate(personality, num_samples, num_steps):
        correlations = []
        action_freqs = []
        for _ in range(num_samples):
            res = simulate(pipeline, personality, num_steps=num_steps)
            action_freq = [res.actions.count(action) / num_steps for action in pipeline.actions]
            action_freqs.append(action_freq)

            scores = pipeline.eval_score(res.actions)
            recoverred_personality = interpret_hexaco_personality(scores)

            x = np.array(list(personality.values()))
            y = np.array(list(recoverred_personality.values()))
            corr = np.corrcoef(x, y)[0, 1]
            correlations.append(corr)
        
        return action_freqs, correlations

    base_personality = {k: 3 for k in sample_hexaco_personality().keys()}
    factors = ["Extraversion", "Conscientiousness", "Openness to Experience", "Agreeableness", "Honesty-Humility"]
    
    all_action_freqs = []
    correlations = []
    colors = []
    labels = []

    for i, factor in enumerate(factors):
        personality_low = set_personality_factor(base_personality.copy(), factor, 1)
        personality_high = set_personality_factor(base_personality.copy(), factor, 5)

        action_freqs_low, correlations_low = _simulate(personality_low, num_samples, num_steps)
        action_freqs_high, correlations_high = _simulate(personality_high, num_samples, num_steps)
        
        all_action_freqs.extend(action_freqs_low + action_freqs_high)
        correlations.append(correlations_low + correlations_high)
        colors.extend([i] * num_samples * 2)  # Use factor index as color
        labels.extend([f"{factor}_low"] * num_samples + [f"{factor}_high"] * num_samples)

    all_action_freqs = np.array(all_action_freqs)
    correlations = np.array(correlations)

    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    base_colors = [mcolors.to_rgba(color) for color in base_colors]
    color_map = {}
    for i, factor in enumerate(factors):
        light_color = mcolors.rgb_to_hsv(base_colors[i][:3])
        light_color[1] *= 0.4  # Reduce saturation for light version
        light_color = mcolors.hsv_to_rgb(light_color)
        
        dark_color = mcolors.rgb_to_hsv(base_colors[i][:3])
        dark_color[1] *= 0.7  # Reduce saturation for dark version
        dark_color[2] *= 0.99  # Reduce value for dark version
        dark_color = mcolors.hsv_to_rgb(dark_color)
        
        color_map[f"{factor}_low"] = light_color
        color_map[f"{factor}_high"] = dark_color

    colors = [color_map[label] for label in labels]

    # PCA
    pca = PCA(n_components=None)
    action_freqs_pca = pca.fit_transform(all_action_freqs)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    action_freqs_tsne = tsne.fit_transform(all_action_freqs)

    # Plot PCA
    plt.figure(figsize=(4, 4), dpi=300)
    for factor in factors:
        mask_low = np.array([label == f"{factor}_low" for label in labels])
        mask_high = np.array([label == f"{factor}_high" for label in labels])
        plt.scatter(action_freqs_pca[mask_low, 0], action_freqs_pca[mask_low, 1], 
                    label=f"{factor} (Low)", facecolors='none', edgecolors=color_map[f"{factor}_low"])
        plt.scatter(action_freqs_pca[mask_high, 0], action_freqs_pca[mask_high, 1], 
                    c=[color_map[f"{factor}_low"]], label=f"{factor} (High)")

    # plt.legend(title="Personality Factors", loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.title("PCA of Action Frequencies for Different Personality Factors")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir / "personality_pca.png")
    plt.savefig(save_dir / "personality_pca.pdf")

    # Plot t-SNE
    plt.figure(figsize=(4, 4), dpi=300)
    for factor in factors:
        mask_low = np.array([label == f"{factor}_low" for label in labels])
        mask_high = np.array([label == f"{factor}_high" for label in labels])
        plt.scatter(action_freqs_tsne[mask_low, 0], action_freqs_tsne[mask_low, 1], 
                    label=f"{factor} (Low)", facecolors='none', edgecolors=color_map[f"{factor}_low"])
        plt.scatter(action_freqs_tsne[mask_high, 0], action_freqs_tsne[mask_high, 1], 
                    c=[color_map[f"{factor}_low"]], label=f"{factor} (High)")

    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    # plt.legend(title="Personality Factors", loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.title("t-SNE of Action Frequencies for Different Personality Factors")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir / "personality_tsne.png")
    plt.savefig(save_dir /"personality_tsne.pdf")

    # legend figure
    plt.figure(figsize=(5, 4), dpi=300)
    plt.legend(legend_handles, legend_labels, title="Personality Factors", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir / "personality_tsne_legend.png")
    plt.savefig(save_dir / "personality_tsne_legend.pdf")

    # Plot explained variance ratio
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("Cumulative Explained Variance Ratio")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.tight_layout()
    plt.savefig(save_dir / "personality_pca_explained_variance.png")


    # plot correlation mean and std for the different factors
    plt.figure(figsize=(3.5, 3.4), dpi=300)
    corr_means = np.mean(correlations, axis=1)
    corr_stds = np.std(correlations, axis=1)
    plt.bar(factors, corr_means, yerr=corr_stds, capsize=5, color=[color_map[f"{factor}_low"] for factor in factors], ecolor="#636363")
    plt.xticks(rotation=45, ha='right')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel("Correlation between original\nand recovered personality")
    plt.tight_layout()
    plt.savefig(save_dir / "personality_correlation.png")
    plt.savefig(save_dir / "personality_correlation.pdf")


if __name__ == "__main__":
    config_name = "hexaco_state-personality_activities-separate_questions"
    # config_name = "hexaco_state-personality_factor_activities-separate_questions"
    # config_name = "hexaco_state-free_activities-separate_questions"
    # config_name = "hexaco_state-question_activities-separate_questions"
    model_name = "claude-3-5-sonnet-20240620"
    batch_name = "batch1"
    # config_name = "hexaco_state-free_activities-separate_questions"
    pipeline_path = Path(__file__).parent / "results" / "separate_questions" / "stoch_policy" / config_name / model_name / batch_name / "gen"

    pipeline = load_pipeline(pipeline_path)
    
    save_dir = save_dir / config_name / model_name / batch_name
    save_dir.mkdir(parents=True, exist_ok=True)

    num_steps = 100
    num_samples = 10
    save_dir = save_dir / f"num_samples_{num_samples}-num_steps_{num_steps}"
    save_dir.mkdir(parents=True, exist_ok=True)
    # extraversion_pca(pipeline, num_samples, num_steps)
    personalities_pca(pipeline, num_samples, num_steps, save_dir)