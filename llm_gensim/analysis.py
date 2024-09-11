import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from llm_gensim.generate_sim import sample_hexaco_personality, simulate, SimPipeline
from llm_gensim.constants import hexaco_data_path, hexaco_scoring_keys_path


def interpret_hexaco_personality(scores, return_details=False):
    if isinstance(scores, dict):
        # make sure order is correct
        scores = list(scores.values())

    # Define the structure of the HEXACO model
    # hexaco_structure = {
    #     'Honesty-Humility': {
    #         'Sincerity': [6, 30, 54, 78],
    #         'Fairness': [12, 36, 60, 84],
    #         'Greed Avoidance': [18, 42, 66, 90],
    #         'Modesty': [24, 48, 72, 96]
    #     },
    #     'Emotionality': {
    #         'Fearfulness': [5, 29, 53, 77],
    #         'Anxiety': [11, 35, 59, 83],
    #         'Dependence': [17, 41, 65, 89],
    #         'Sentimentality': [23, 47, 71, 95]
    #     },
    #     'Extraversion': {
    #         'Social Self-Esteem': [4, 28, 52, 76],
    #         'Social Boldness': [10, 34, 58, 82],
    #         'Sociability': [16, 40, 64, 88],
    #         'Liveliness': [22, 46, 70, 94]
    #     },
    #     'Agreeableness': {
    #         'Forgiveness': [3, 27, 51, 75],
    #         'Gentleness': [9, 33, 57, 81],
    #         'Flexibility': [15, 39, 63, 87],
    #         'Patience': [21, 45, 69, 93]
    #     },
    #     'Conscientiousness': {
    #         'Organization': [2, 26, 50, 74],
    #         'Diligence': [8, 32, 56, 80],
    #         'Perfectionism': [14, 38, 62, 86],
    #         'Prudence': [20, 44, 68, 92]
    #     },
    #     'Openness to Experience': {
    #         'Aesthetic Appreciation': [1, 25, 49, 73],
    #         'Inquisitiveness': [7, 31, 55, 79],
    #         'Creativity': [13, 37, 61, 85],
    #         'Unconventionality': [19, 43, 67, 91]
    #     }
    # }
    
    # # Define reverse-keyed items
    # reverse_keyed = [1, 6, 9, 10, 12, 13, 15, 16, 19, 20, 21, 25, 29, 35, 36, 38, 41, 42, 
    #                  44, 50, 51, 52, 53, 54, 55, 56, 59, 63, 66, 67, 70, 72, 74, 76, 77, 
    #                  79, 80, 82, 84, 85, 87, 89, 90, 91, 92, 94, 95, 96]
    
    with open(hexaco_scoring_keys_path, "r") as f:
        scoring_keys = yaml.safe_load(f)
    hexaco_structure = scoring_keys["hexaco_scoring_keys"]
    reverse_keyed = scoring_keys["reverse_keyed"]

    # Function to reverse score
    def reverse_score(score):
        return 6 - score
    
    # Calculate facet scores
    facet_scores = {}
    for factor, facets in hexaco_structure.items():
        facet_scores[factor] = {}
        for facet, questions in facets.items():
            facet_total = 0
            for q in questions:
                score = scores[q-1]  # Adjust for 0-based indexing
                if q in reverse_keyed:
                    score = reverse_score(score)
                facet_total += score
            facet_scores[factor][facet] = facet_total / len(questions)
    
    # Calculate factor scores
    factor_scores = {}
    for factor, facets in facet_scores.items():
        factor_scores[factor] = sum(facets.values()) / len(facets)
    
    # Calculate Altruism score (interstitial facet scale)
    altruism_questions = [97, 98, 99, 100]
    altruism_score = 0
    for q in altruism_questions:
        score = scores[q-1]  # Adjust for 0-based indexing
        if q in [99, 100]:  # These are reverse-keyed
            score = reverse_score(score)
        altruism_score += score
    altruism_score /= len(altruism_questions)
    
    if return_details:
        return {
            'factor_scores': factor_scores,
            'facet_scores': facet_scores,
            'altruism_score': altruism_score
        }
    else:
        scores = {}
        for factor, facet_scores in facet_scores.items():
            for facet, score in facet_scores.items():
                scores[f"{facet}"] = score
        scores["Altruism"] = altruism_score
        return scores


def hexaco_facets_to_factors(facets: dict | list) -> dict | list:
    with open(hexaco_scoring_keys_path, "r") as f:
        scoring_keys = yaml.safe_load(f)
    hexaco_structure = scoring_keys["hexaco_scoring_keys"]

    if isinstance(facets, dict):
        factor_scores = {}

        for factor, facet_scores in hexaco_structure.items():
            factor_scores[factor] = sum(facets[facet] for facet in facet_scores) / len(facet_scores)
        
        return factor_scores
    elif isinstance(facets, list):
        return list(hexaco_structure.keys())

    else:
        raise ValueError(f"Invalid type for facets: {type(facets)}")


def analyze(pipeline, res, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    personality = res.personality
    init_state = res.init_state
    scores = pipeline.eval_score(res.actions)
    inferred_personality = interpret_hexaco_personality(scores)
    inferred_factors = interpret_hexaco_personality(scores, return_details=True)

    # save original personality
    with open(save_dir / "personality.json", "w") as f:
        json.dump(personality, f, indent=4)

    # save inferred personality
    with open(save_dir / "inferred_personality.json", "w") as f:
        json.dump(inferred_personality, f, indent=4)

    # save eval scores
    with open(save_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)


    plt.figure(figsize=(10, 5), dpi=300)
    for var in pipeline.states:
        plt.plot([state[var] for state in res.states], label=var)
    plt.legend()
    plt.savefig(save_dir / "state_vars.png")
    plt.savefig(save_dir / "state_vars.pdf")
    plt.figure(figsize=(10, 5), dpi=300)
    for activity in pipeline.actions:
        plt.plot([action == activity for action in res.actions], label=activity)
    plt.legend()
    plt.savefig(save_dir / "activities.png")
    plt.savefig(save_dir / "activities.pdf")
    # plot statistics
    plt.figure(figsize=(10, 5), dpi=300)
    avg_stats = {var: np.mean([state[var] for state in res.states]) for var in pipeline.states}
    std_stats = {var: np.std([state[var] for state in res.states]) for var in pipeline.states}
    plt.bar(avg_stats.keys(), avg_stats.values(), yerr=std_stats.values())
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("State variables")
    plt.ylabel("Average value")
    plt.tight_layout()
    plt.savefig(save_dir / "avg_stats.png")
    plt.savefig(save_dir / "avg_stats.pdf")
    plt.figure(figsize=(10, 5), dpi=300)
    avg_actions = {activity: np.mean([action == activity for action in res.actions]) for activity in pipeline.actions}
    plt.bar(avg_actions.keys(), avg_actions.values())
    # align x labels center
    plt.xticks(rotation=90, ha='center', fontsize=5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Activities")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_dir / "avg_actions.png")
    plt.savefig(save_dir / "avg_actions.pdf")


    # plot actions ordered by frequency
    avg_actions_ordered = sorted(avg_actions.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar([x[0] for x in avg_actions_ordered], [x[1] for x in avg_actions_ordered])
    plt.xticks(rotation=90, ha='center', fontsize=5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Activities")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_dir / "avg_actions_ordered.png")
    plt.savefig(save_dir / "avg_actions_ordered.pdf")


    # same but show only to 20
    plt.figure(figsize=(6, 5), dpi=300)
    plt.bar([x[0] for x in avg_actions_ordered[:20]], [x[1] for x in avg_actions_ordered[:20]])
    plt.xticks(rotation=90, ha='center', fontsize=7)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Activities")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_dir / "avg_actions_ordered_top10.png")
    plt.savefig(save_dir / "avg_actions_ordered_top10.pdf")

    # plot personality factors
    facet_names = list(personality.keys())
    original_scores = list(personality.values())
    inferred_scores = [inferred_personality[facet] for facet in facet_names]

    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(facet_names, original_scores, marker=".", label="Original")
    plt.plot(facet_names, inferred_scores, marker=".", label="Inferred")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Personality factors")
    plt.ylabel("Average score")
    plt.ylim(0.9, 5.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "personality.png")
    plt.savefig(save_dir / "personality.pdf")


    factors = hexaco_facets_to_factors(personality)
    inferred_factors = hexaco_facets_to_factors(inferred_personality)
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(list(factors.keys()), list(factors.values()), label="Original")
    plt.plot(list(inferred_factors.keys()), list(inferred_factors.values()), label="Inferred")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Personality factors")
    plt.ylabel("Average score")
    plt.ylim(0.9, 5.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "factors.png")
    plt.savefig(save_dir / "factors.pdf")


    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(list(range(1, len(scores)+1)), scores)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(0.9, 5.1)
    plt.xlabel("Questions")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(save_dir / "scores.png")
    plt.savefig(save_dir / "scores.pdf")


    if init_state is not None:
        with open(save_dir / "init_state.json", "w") as f:
            json.dump(init_state, f, indent=4)

    # radar chart
    from llm_gensim.radar_plot import plot_radar_chart

    data = []
    data_labels = []
    for i, personality in enumerate([personality, inferred_personality]):
        factors = hexaco_facets_to_factors(personality)
        data.append(list(factors.values()).copy())
        data_labels.append(f"Original" if i == 0 else "Inferred")

    labels = factors.keys()
    plot_radar_chart(
        data,
        data_labels,
        labels,
        title="Personality Factors",
        figsize=(4, 4)
    )
    ax = plt.gca()
    ax.set_rgrids([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    plt.savefig(save_dir / "factors_radar.png")
    plt.savefig(save_dir / "factors_radar.pdf")
    plt.close('all')


    # TODO: require some constraints on the eval_scores generated code
    # # plot list of activities used in eval_scores for each personality factor
    # with open(hexaco_scoring_keys_path, "r") as f:
    #     data = yaml.safe_load(f)
    # scoring_keys = data["hexaco_scoring_keys"]
    # reverse_keyed = data["reverse_keyed"]

    # question_activities = eval_question_activities    

    # for factor in scoring_keys.keys():
    #     factor_pos_activities = []
    #     factor_neg_activities = []
    #     factor_pos_questions = []
    #     factor_neg_questions = []
    #     print("Factor:", factor)
    #     for facet, questions in scoring_keys[factor].items():
    #         for question in questions:
    #             a = question_activities[question]
    #             factor_pos_activities.extend(a["positive_activities"])
    #             factor_neg_activities.extend(a["negative_activities"])
    #             factor_pos_questions.extend([
    #                 f"{question}" if question not in reverse_keyed else f"{question} (R)"
    #             ]*len(a["positive_activities"]))
    #             factor_neg_questions.extend([
    #                 f"{question}" if question not in reverse_keyed else f"{question} (R)"
    #             ]*len(a["negative_activities"]))

    #     act_counts = Counter(res.actions)

    #     plt.figure(figsize=(10, 8), dpi=300)
        
    #     # Positive activities subplot
    #     plt.subplot(2, 1, 1)
    #     pos_counts = [act_counts[act] for act in factor_pos_activities]
    #     labels = [f"{act} - {q}" for act, q in zip(factor_pos_activities, factor_pos_questions)]
    #     plt.bar(range(len(factor_pos_activities)), pos_counts)
    #     plt.xticks(range(len(factor_pos_activities)), labels, rotation=45, ha='right', fontsize=5)
    #     plt.title(f"Positive Activities for {factor}")
    #     plt.ylabel("Frequency")
        
    #     # Negative activities subplot
    #     plt.subplot(2, 1, 2)
    #     neg_counts = [act_counts[act] for act in factor_neg_activities]
    #     labels = [f"{act} - {q}" for act, q in zip(factor_neg_activities, factor_neg_questions)]
    #     plt.bar(range(len(factor_neg_activities)), neg_counts)
    #     plt.xticks(range(len(factor_neg_activities)), labels, rotation=45, ha='right', fontsize=5)
    #     plt.title(f"Negative Activities for {factor}")
    #     plt.ylabel("Frequency")
        
    #     plt.tight_layout()
    #     plt.savefig(save_dir / f"{factor}_activities.png")
    #     plt.close()

    return scores, inferred_personality, inferred_factors


def eval_random_personality(sim_pipeline, save_dir, num_runs=10):
    metrics = {
        "pearsonr": [],
        "mae": [],
        "mse": []
    }
    for i in range(num_runs):
        _save_dir = save_dir / f"run{i}"

        personality = sample_hexaco_personality()

        sim_result = simulate(
            sim_pipeline,
            personality,
            num_steps=100
        )
        
        scores, inferred_personality, inferred_factors = analyze(
            sim_pipeline,
            sim_result,
            _save_dir
        )
        
        target = np.array(list(personality.values()))
        pred = np.array([inferred_personality[k] for k in personality.keys()])

        metrics["pearsonr"].append(pearsonr(target, pred))
        metrics["mae"].append(mean_absolute_error(target, pred))
        metrics["mse"].append(mean_squared_error(target, pred))

    metrics["pearsonr"] = float(np.mean(metrics["pearsonr"]))
    metrics["mae"] = float(np.mean(metrics["mae"]))
    metrics["mse"] = float(np.mean(metrics["mse"]))

    # save metrics
    with open(save_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    return metrics


def eval_single_factor(sim_pipeline: SimPipeline, factor_name: str, num_values, save_dir, default_value='mean', num_runs=10):
    """
    Vary one factor at a time while keeping others at their mean value.
    The specified factor varies between 1 and 5 with constant step size, num_values times.
    """
    def _make_personality(factors):
        """
        Initialize personality to mean values except for the factors given in arguments.
        """
        with open(hexaco_data_path, "r") as f:
            data = yaml.safe_load(f)["factors"]

        if default_value == 'mean':
            for name, score in factors.items():
                for item in data:
                    if item["category"] == name:
                        item["mean"] = score
                        item["sd"] = 0
            scores = np.array([factor["mean"] for factor in data])
            return {factor["name"]: scores[i] for i, factor in enumerate(data)}

        elif default_value == 'zero':
            personality = {}
            for item in data:
                if item["category"] == factor_name:
                    personality[item["name"]] = factors[factor_name]
                else:
                    personality[item["name"]] = 1.0
            return personality

        else:
            raise ValueError(f"Invalid default value: {default_value}")



    values = np.linspace(1, 5, num_values)
    # values = [5]
    personalities = [_make_personality({factor_name: v}) for v in values]

    metrics = {
        "pearsonr": [],
        "mse": [],
        "mae": []
    }
    results = defaultdict(list)
    for value, personality in zip(values, personalities):
        _save_dir = save_dir / f"{factor_name}-{value}"

        sim_result = simulate(sim_pipeline, personality, num_steps=100)

        scores, inferred_personality, inferred_factors = analyze(
            sim_pipeline,
            sim_result,
            _save_dir
        )
        inferred_personality = {k: inferred_personality[k] for k in personality.keys()}

        target = np.array(list(personality.values()))
        pred = np.array([inferred_personality[k] for k in personality.keys()])

        metrics["pearsonr"].append(pearsonr(target, pred))
        metrics["mae"].append(mean_absolute_error(target, pred))
        metrics["mse"].append(mean_squared_error(target, pred))

        results["scores"].append(scores)
        results["personality"].append(personality)
        results["inferred_personality"].append(inferred_personality)
        results["inferred_factors"].append(inferred_factors)

    metrics["pearsonr"] = float(np.mean(metrics["pearsonr"]))
    metrics["mae"] = float(np.mean(metrics["mae"]))
    metrics["mse"] = float(np.mean(metrics["mse"]))

    # save metrics
    with open(save_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    # plot personalities    
    plt.figure(figsize=(5, 3), dpi=300)
    x_labels = list(results["personality"][0].keys())
    colors = ["cornflowerblue", "coral"]
    for i, inferred_personality in enumerate(results["inferred_personality"]):
        plt.plot(inferred_personality.keys(), inferred_personality.values(), color=colors[i])
    
    for i, personality in enumerate(results["personality"]):
        plt.plot(personality.keys(), personality.values(), linestyle="--", color=colors[i])

    plt.plot([], [], color="cornflowerblue", label=f"{factor_name} = 1")
    plt.plot([], [], color="coral", label=f"{factor_name} = 5")
    plt.plot([], [], color="gray", linestyle='--', label="Groundtruth")
    plt.plot([], [], color="gray", label="Simulation")

    plt.xticks(rotation=45, ha='right', fontsize=5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Hexaco personality inventory")
    plt.ylabel("Score")
    plt.ylim(0.9, 5.1)
    plt.yticks(np.linspace(1, 5, 5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"personality.png")
    plt.close()

    # plot groundtruth factors vs inferred factors
    plt.figure(figsize=(5, 3), dpi=300)
    for i, personality in enumerate(results["personality"]):
        factors = hexaco_facets_to_factors(personality)
        plt.plot(factors.keys(), factors.values(), label="Groundtruth", color=colors[i], linestyle='--')
    for i, personality in enumerate(results["inferred_personality"]):
        factors = hexaco_facets_to_factors(personality)
        plt.plot(factors.keys(), factors.values(), label="Inferred", color=colors[i])
    plt.xlabel("Hexaco personality inventory")
    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.ylabel("Score")
    plt.ylim(0.9, 5.1)
    plt.yticks(np.linspace(1, 5, 5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "factors.png")

    # same but as radar chart
    from llm_gensim.radar_plot import plot_radar_chart

    data = []
    data_labels = []
    for i, personality in enumerate(results["personality"]):
        factors = hexaco_facets_to_factors(personality)
        data.append(list(factors.values()).copy())
        data_labels.append(f"{factor_name} = {values[i]}")
    for i, personality in enumerate(results["inferred_personality"]):
        factors = hexaco_facets_to_factors(personality)
        data.append(list(factors.values()).copy())
        data_labels.append(f"{factor_name} = {values[i]}")
    labels = factors.keys()

    plot_radar_chart(
        data,
        data_labels,
        labels,
        title="Personality Factors"
    )
    ax = plt.gca()
    ax.set_rgrids([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)

    plt.savefig(save_dir / "factors_radar.png")

    return metrics

