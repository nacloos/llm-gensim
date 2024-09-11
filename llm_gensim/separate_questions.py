from collections import defaultdict
import os
from pathlib import Path
import numpy as np
import yaml
import anthropic
import re
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


from llm_gensim.analysis import analyze, eval_single_factor, eval_random_personality, hexaco_facets_to_factors, interpret_hexaco_personality
from llm_gensim.llm_utils import parse_output, llm_model
from llm_gensim.generate_sim import SimResult, gen_pipeline_step, GenPipeline, GenPipelineStep, SimPipeline, Env, sample_hexaco_personality, simulate


save_dir = Path(__file__).parent / "results" / Path(__file__).stem
save_dir.mkdir(parents=True, exist_ok=True)

hexaco_questions_path = Path(__file__).parent / "configs" / "hexaco_form.md"
hexaco_data_path = Path(__file__).parent / "configs" / "hexaco_data.yaml"

# https://github.com/anthropics/prompt-eng-interactive-tutorial/blob/master/Anthropic%201P/01_Basic_Prompt_Structure.ipynb
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


def get_completion(messages: list[dict], model_name, system_prompt="", temperature=1.0):
    # user and assistant messages MUST alternate, and messages MUST start with a user turn.
    message = client.messages.create(
        model=model_name,
        max_tokens=8192,
        temperature=temperature,
        system=system_prompt,
        messages=messages
    )
    return message.content[0].text


def make_eval_code(gen_pipeline: GenPipeline, step: GenPipelineStep, save_dir, num_questions=100):
    save_dir.mkdir(parents=True, exist_ok=True)

    # save prompt if doesn't exist
    if not (save_dir / "prompt.md").exists():
        with open(save_dir / "prompt.md", "w") as f:
            f.write(step.prompt.format(**gen_pipeline.context_vars))

    messages = [
        {"role": "user", "content": step.prompt.format(**gen_pipeline.context_vars)}
    ]

    utils_blocks = []
    question_blocks_dict = {}
    i = 1
    while len(question_blocks_dict) < num_questions:
        if not (save_dir / f"answer{i}.md").exists():
            print(f"Generating answer{i}.md")
            # save messages
            with open(save_dir / f"messages{i}.yaml", "w") as f:
                yaml.dump(messages, f)

            # prompt llm
            answer = get_completion(messages, gen_pipeline.llm_name, system_prompt=gen_pipeline.system_prompt)

            # save answer
            with open(save_dir / f"answer{i}.md", "w") as f:
                f.write(answer)
        else:
            print(f"Loading answer{i}.md")
            answer = open(save_dir / f"answer{i}.md", "r").read()

        # extract question blocks from answer
        blocks = re.findall(r'<q(\d+)>(.*?)</q\d+>', answer, re.DOTALL)
        for num, content in blocks:
            if num not in question_blocks_dict:
                question_blocks_dict[num] = content

        utils_blocks.extend(re.findall(r'<utils>(.*?)</utils>', answer, re.DOTALL))

        if len(question_blocks_dict) < num_questions:
            messages += [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Continue implementing the code. Rewrite the last block of code in your previous answer if it was truncated."}
            ]
            i += 1

    utils_code = "\n\n".join(utils_blocks)
    utils_code = "\n".join([f"    {line}" for line in utils_code.split("\n")])

    question_blocks = [question_blocks_dict[k] for k in sorted(question_blocks_dict.keys())]
    questions_code = "\n\n".join(question_blocks)
    # indent question code
    questions_code = "\n".join([f"    {line}" for line in questions_code.split("\n")])

    score_list = "[" + ", ".join([f"q{i}(activities)" for i in range(1, 101)]) + "]"
    # place the question code inside a function eval_score
    code = f"""
def eval_score(activities: list[str]) -> list[int]:
    # utils
    {utils_code}

    # eval questions
    {questions_code}
    
    return {score_list}
"""
    with open(save_dir / "eval_score.py", "w") as f:
        f.write(code)

    full_answer = f"```python\n{code}\n```"
    # save to full answer to answer.md
    with open(save_dir / "answer.md", "w") as f:
        f.write(full_answer)

    text_output, parsed_output = parse_output(full_answer, output_type=step.output_type, output_name=step.output_name)

    gen_pipeline.context_vars["eval_score"] = text_output
    gen_pipeline.outputs["eval_score"] = parsed_output
    return


def make_policy_code(gen_pipeline: GenPipeline, step: GenPipelineStep, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    # save prompt if doesn't exist
    if not (save_dir / "prompt.md").exists():
        with open(save_dir / "prompt.md", "w") as f:
            f.write(step.prompt.format(**gen_pipeline.context_vars))

    messages = [
        {"role": "user", "content": step.prompt.format(**gen_pipeline.context_vars)}
    ]
    utils_blocks = []
    policy_blocks_dict = {}
    i = 1
    while len(policy_blocks_dict) < len(gen_pipeline.outputs["actions"]):
        if not (save_dir / f"answer{i}.md").exists():
            print(f"Generating answer{i}.md")
            answer = get_completion(messages, gen_pipeline.llm_name, system_prompt=gen_pipeline.system_prompt)
            with open(save_dir / f"answer{i}.md", "w") as f:
                f.write(answer)
        else:
            print(f"Loading answer{i}.md")
            answer = open(save_dir / f"answer{i}.md", "r").read()

        blocks = re.findall(r'<a(\d+)>(.*?)</a\d+>', answer, re.DOTALL)
        for num, content in blocks:
            # make sure don't have duplicate blocks because the code for an activity might be repeated twice in two consecutive answers
            if num not in policy_blocks_dict:
                policy_blocks_dict[num] = content

        utils_blocks.extend(re.findall(r'<utils>(.*?)</utils>', answer, re.DOTALL))

        if len(policy_blocks_dict) < len(gen_pipeline.outputs["actions"]):
            messages += [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Continue implementing the code. Rewrite the last block of code in your previous answer if it was truncated."}
            ]
            i += 1


    utils_code = "\n\n".join(utils_blocks)
    utils_code = "\n".join([f"    {line}" for line in utils_code.split("\n")])

    policy_blocks = [policy_blocks_dict[k] for k in sorted(policy_blocks_dict.keys())]
    policy_code = "\n\n".join(policy_blocks)
    policy_code = "\n".join([f"    {line}" for line in policy_code.split("\n")])

    proba_list = "[" + ", ".join([f"a{i}(state)" for i in range(1, len(gen_pipeline.outputs["actions"])+1)]) + "]"

    code = f"""
def policy(state: dict) -> str:
    # utils
    {utils_code}

    # policy
    {policy_code}

    # TODO: temp fix
    # state = list(state.values())
    activities = {gen_pipeline.outputs["actions"]}

    import numpy as np
    activity_probabilities = np.array({proba_list})
    print(activity_probabilities)
    # softmax
    beta = 5.0
    p = np.exp(activity_probabilities * beta) / np.exp(activity_probabilities * beta).sum()
    # activity_probabilities = activity_probabilities / activity_probabilities.sum()
    # sample activity
    activity = np.random.choice(activities, p=p)
    return activity
"""

    with open(save_dir / "policy.py", "w") as f:
        f.write(code)


    full_answer = f"```python\n{code}\n```"
    # save to full answer to answer.md
    with open(save_dir / "answer.md", "w") as f:
        f.write(full_answer)

    text_output, parsed_output = parse_output(full_answer, output_type=step.output_type, output_name=step.output_name)

    gen_pipeline.context_vars["policy"] = text_output
    gen_pipeline.outputs["policy"] = parsed_output

    return


def make_sim_pipeline(model_id, gen_pipeline_config, save_dir, fix_state=True, no_effects=True) -> SimPipeline:
    llm = llm_model(model_id)

    # load data
    with open(hexaco_questions_path, "r") as f:
        questions = f.read()

    with open(hexaco_data_path, "r") as f:
        hexaco_data = yaml.safe_load(f)

    facets = [d["name"] for d in hexaco_data["factors"]]
    factors = hexaco_facets_to_factors(facets)

    context_vars = {
        "questions": questions,
        "personality_factors": ", ".join(factors),
        "personality_facets": ", ".join(facets)
    }
    outputs = {}

    if fix_state: 
        # agent state = personality
        context_vars['states'] = ", ".join(facets)
        outputs['states'] = facets

    if no_effects:
        outputs['effects'] = {}

    policy_config = gen_pipeline_config["steps"].pop("policy")
    eval_config = gen_pipeline_config["steps"].pop("eval_score")

    # make pipeline
    pipeline = GenPipeline(
        llm_name=model_id,
        llm=llm,
        steps=[
            GenPipelineStep(**step) for step in gen_pipeline_config["steps"].values()
            # GenPipelineStep(**step) for k, step in gen_pipeline_config["steps"].items() if k != "eval_score"
        ],
        system_prompt=gen_pipeline_config["system_prompt"],
        save_dir=save_dir,
        context_vars=context_vars,
        outputs=outputs
    )

    print(f"Generating pipeline with {model_id}")
    print("Save dir: ", save_dir)
    for step in pipeline.steps:
        gen_pipeline_step(pipeline, step)

    step = GenPipelineStep(**policy_config)
    make_policy_code(pipeline, step, save_dir / step.output_name)

    step = GenPipelineStep(**eval_config)
    make_eval_code(pipeline, step, save_dir / step.output_name)

    if fix_state:
        # identity function since agent state = personality (normalized)
        pipeline.outputs["init_state"] = lambda personality: {k: (v-1)/4*100 for k, v in personality.items()}

    pipeline.outputs["effects"] = {}

    return SimPipeline(
        actions=pipeline.outputs["actions"],
        states=pipeline.outputs["states"],
        effects=pipeline.outputs["effects"],
        init_state_fn=pipeline.outputs["init_state"],
        policy=pipeline.outputs["policy"],
        eval_score=pipeline.outputs["eval_score"]
    )


def plot_personalities(
        personality_low,
        personality_high,
        inferred_personalities_low,
        inferred_personalities_high,
        save_dir
):
    # plot personality vs inferred personality
    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(personality_low.keys(), personality_low.values(), label=f"{factor_name}=1", color="cornflowerblue", linestyle='--')
    plt.plot(personality_high.keys(), personality_high.values(), label=f"{factor_name}=5", color="coral", linestyle='--')
    # plot mean and std of inferred personality
    x_labels_low = [k for k in personality_low.keys()]
    x_labels_high = [k for k in personality_high.keys()]
    inferred_personalities_low_arr = np.array([list(v.values()) for v in inferred_personalities_low])
    inferred_personalities_high_arr = np.array([list(v.values()) for v in inferred_personalities_high])
    plt.plot(inferred_personalities_low_arr.mean(axis=0), color="cornflowerblue")
    plt.plot(inferred_personalities_high_arr.mean(axis=0), color="coral")
    plt.fill_between(x_labels_low, inferred_personalities_low_arr.mean(axis=0)-inferred_personalities_low_arr.std(axis=0), inferred_personalities_low_arr.mean(axis=0)+inferred_personalities_low_arr.std(axis=0), color="cornflowerblue", alpha=0.3)
    plt.fill_between(x_labels_high, inferred_personalities_high_arr.mean(axis=0)-inferred_personalities_high_arr.std(axis=0), inferred_personalities_high_arr.mean(axis=0)+inferred_personalities_high_arr.std(axis=0), color="coral", alpha=0.3)

    plt.xlabel("Hexaco personality inventory")
    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.ylabel("Score")
    plt.ylim(0.9, 5.1)
    plt.yticks(np.linspace(1, 5, 5))
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / f"personality_avg_std.png")
    plt.close()

    plt.figure(figsize=(5, 3), dpi=300)
    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(personality_low.keys(), personality_low.values(), label=f"{factor_name}=1", color="cornflowerblue", linestyle='--')
    plt.plot(personality_high.keys(), personality_high.values(), label=f"{factor_name}=5", color="coral", linestyle='--')
    # plot inferred personality
    for inferred_personality_low in inferred_personalities_low:
        plt.plot(inferred_personality_low.keys(), inferred_personality_low.values(), color="cornflowerblue")
    for inferred_personality_high in inferred_personalities_high:
        plt.plot(inferred_personality_high.keys(), inferred_personality_high.values(), color="coral")
    plt.xlabel("Hexaco personality inventory")
    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.ylabel("Score")
    plt.ylim(0.9, 5.1)
    plt.yticks(np.linspace(1, 5, 5))
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / f"personality_all.png")
    plt.close()


def plot_activity_comparison(
        sim_pipeline: SimPipeline,
        res1: SimResult,
        res2: SimResult,
        label1,
        label2,
        save_dir
):
    # plot frequency of activities
    avg_actions1 = {activity: np.mean([action == activity for action in res1.actions]) for activity in sim_pipeline.actions}
    avg_actions2 = {activity: np.mean([action == activity for action in res2.actions]) for activity in sim_pipeline.actions}
    
    avg_actions_ordered1 = sorted(avg_actions1.items(), key=lambda x: x[1], reverse=True)
    avg_actions_ordered2 = sorted(avg_actions2.items(), key=lambda x: x[1], reverse=True)

    k = 3
    top_activities1 = [activity for activity, _ in avg_actions_ordered1[:k]]
    top_activities2 = []
    for activity, _ in avg_actions_ordered2:
        if activity not in top_activities1:
            top_activities2.append(activity)
        if len(top_activities2) == k:
            break

    top_activities = top_activities1 + top_activities2[::-1]

    import seaborn as sns
    import pandas as pd

    data_df = pd.DataFrame({
        "Activity": top_activities + top_activities,
        "Frequency": [avg_actions1[activity] for activity in top_activities] + [avg_actions2[activity] for activity in top_activities],
        "Group": [label1] * 2 * k + [label2] * 2 * k
    })
    # Get the colors from matplotlib
    color_names = ["cornflowerblue", "coral"]
    colors = [plt.cm.colors.to_rgb(color) for color in color_names]

    sns.barplot(x="Activity", y="Frequency", hue="Group", data=data_df, palette=colors)
    # align x labels center
    plt.xticks(rotation=90, ha='center')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Activities")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_dir / "bar_plot_activity_comparison.png")
    plt.savefig(save_dir / "bar_plot_activity_comparison.pdf")

    # same but activities on the y axis (seaborn orient y)
    plt.figure(figsize=(7, 3), dpi=300)
    sns.barplot(x="Frequency", y="Activity", hue="Group", data=data_df, orient="y", palette=colors)
    plt.xlabel("Frequency")
    plt.ylabel("Activities")
    # remove title from legend
    plt.legend(title=None)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "bar_plot_activity_comparison_horiz.png")
    plt.savefig(save_dir / "bar_plot_activity_comparison_horiz.pdf")

    plt.figure(figsize=(5, 3), dpi=300)
    plt.bar(range(len(avg_actions1)), list(avg_actions1.values()), color=colors[0])
    plt.bar(range(len(avg_actions2)), list(avg_actions2.values()), color=colors[1])
    plt.xticks(rotation=90, ha='center')
    plt.xlabel("Activities")
    plt.ylabel("Frequency")
    plt.legend([label1, label2])
    plt.tight_layout()
    plt.savefig(save_dir / "bar_plot_activity_comparison_single.png")
    plt.savefig(save_dir / "bar_plot_activity_comparison_single.pdf")

    plt.figure(figsize=(6, 3), dpi=300)
    y_pos = range(len(top_activities))[::-1]
    a1 = [avg_actions1[activity] for activity in top_activities]
    a2 = [avg_actions2[activity] for activity in top_activities]
    # same as edgecolor but lighter
    facecolors = [plt.cm.colors.to_rgba(color) for color in colors]
    facecolors = [[r, g, b, 0.3] for r, g, b, _ in facecolors]
    plt.barh(y_pos, a2, height=0.35, edgecolor=colors[1], facecolor=facecolors[1], label=label2, align='center')
    plt.barh([y + 0.39 for y in y_pos], a1, height=0.35, edgecolor=colors[0], facecolor=facecolors[0], label=label1, align='center')
    plt.yticks([y + 0.195 for y in y_pos], top_activities)
    plt.xlabel("Frequency")
    plt.ylabel("Activities")
    plt.legend()
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_dir / "bar_plot_activity_comparison_horiz_matplotlib.png")
    plt.savefig(save_dir / "bar_plot_activity_comparison_horiz_matplotlib.pdf")
    plt.close()


def plot_personality_comparison(
        personality1,
        personality2,
        inferred_personality1,
        inferred_personality2,
        save_dir
):
    plt.figure(figsize=(5, 3), dpi=300)
    plt.plot(personality1.keys(), personality1.values(), label=f"{factor_name}=1", color="cornflowerblue", linestyle='--')
    plt.plot(personality2.keys(), personality2.values(), label=f"{factor_name}=5", color="coral", linestyle='--')
    # plot inferred personality
    for inferred_personality1 in inferred_personality1:
        plt.plot(inferred_personality1.keys(), inferred_personality1.values(), color="cornflowerblue")
    for inferred_personality2 in inferred_personality2:
        plt.plot(inferred_personality2.keys(), inferred_personality2.values(), color="coral")
    plt.xlabel("Hexaco personality inventory")
    plt.xticks(rotation=45, ha='right', fontsize=5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "personality_comparison.png")
    plt.savefig(save_dir / "personality_comparison.pdf")
    plt.close()

    # radar plot
    from llm_gensim.radar_plot import plot_radar_chart

    def _plot_radar(personality, inferred_personality, colors, save_name):
        data = []
        data_labels = []
        for i, personality in enumerate([personality, inferred_personality]):
            factors = hexaco_facets_to_factors(personality)
            data.append(list(factors.values()).copy())
            data_labels.append(f"Original" if i == 0 else "Recovered")

        labels = factors.keys()
        plot_radar_chart(
            data,
            data_labels,
            labels,
            title="Personality Factors",
            figsize=(4, 4),
            colors=colors
        )
        ax = plt.gca()
        ax.set_rgrids([1, 2, 3, 4, 5])
        ax.set_ylim(0, 5)
        plt.savefig(save_dir / f"{save_name}.png")
        plt.savefig(save_dir / f"{save_name}.pdf")


    colors1 = ["black", "cornflowerblue"]
    colors2 = ["black", "coral"]
    _plot_radar(personality1, inferred_personality1, colors1, "radar_personality1")
    _plot_radar(personality2, inferred_personality2, colors2, "radar_personality2")
    plt.close('all')


def eval_pipeline(sim_pipeline: SimPipeline, factor_name: str, num_sim_steps, num_samples, num_runs, save_dir):
    data = yaml.safe_load(open(hexaco_data_path, "r"))["factors"]

    def _set_factor(personality, factor_name, value):
        personality = personality.copy()
        for item in data:
            if item["category"] == factor_name:
                personality[item["name"]] = value
        return personality

    def _run(personality, save_dir=None):
        res = simulate(sim_pipeline, personality, num_steps=num_sim_steps)

        if save_dir:
            analyze(sim_pipeline, res, save_dir=save_dir)

        scores = sim_pipeline.eval_score(res.actions)
        inferred_personality = interpret_hexaco_personality(scores)
        return res, inferred_personality
        # target = np.array(list(personality.values()))
        # pred = np.array([inferred_personality[k] for k in personality.keys()])

        # metrics = {
        #     "pearsonr": pearsonr(target, pred),
        #     "mse": mean_squared_error(target, pred),
        #     "mae": mean_absolute_error(target, pred)
        # }
        # return metrics, inferred_personality


    results = defaultdict(list)
    for i in range(num_samples):
        _save_dir = save_dir / f"sample{i}"
        _save_dir.mkdir(parents=True, exist_ok=True)

        rdm_personality = sample_hexaco_personality()
        # for k, v in rdm_personality.items():
        #     rdm_personality[k] = 3

        personality_low = _set_factor(rdm_personality, factor_name, 1)
        personality_high = _set_factor(rdm_personality, factor_name, 5)

        inferred_personalities_low = []
        inferred_personalities_high = []
        for k in range(num_runs):
            __save_dir = _save_dir / f"run{k}"

            res_low, inferred_personality_low = _run(personality_low, save_dir=__save_dir / "low")
            res_high, inferred_personality_high = _run(personality_high, save_dir=__save_dir / "high")
            inferred_personalities_low.append(inferred_personality_low)
            inferred_personalities_high.append(inferred_personality_high)

            plot_activity_comparison(
                sim_pipeline,
                res_high,
                res_low,
                label1=f"High {factor_name}",
                label2=f"Low {factor_name}",
                save_dir=__save_dir
            )

            plot_personality_comparison(
                personality_low,
                personality_high,
                inferred_personalities_low,
                inferred_personalities_high,
                save_dir=__save_dir
            )

        plot_personalities(
            personality_low,
            personality_high,
            inferred_personalities_low,
            inferred_personalities_high,
            save_dir / f"sample{i}"
        )




if __name__ == "__main__":
    model_id = "claude-3-5-sonnet-20240620"

    # pipeline_name = "hexaco_state-free_activities-no_effects-test"
    # pipeline_name = "hexaco_state-free_activities-separate_questions"
    # pipeline_name = "hexaco_state-personality_activities-separate_questions"
    # pipeline_name = "hexaco_state-personality_factor_activities-separate_questions"
    pipeline_name = "hexaco_state-question_activities-separate_questions"

    batch_name = "batch1"

    fix_state = True if "hexaco_state" in pipeline_name else False
    no_effects = True if "no_effects" in pipeline_name else False

    _save_dir = save_dir / "stoch_policy" / pipeline_name / model_id / batch_name

    pipe_save_dir = _save_dir / "gen"
    sim_save_dir = _save_dir / "sim"

    pipeline_path = Path(__file__).parent / "configs" / "separate_questions" / (pipeline_name + ".yaml")
    pipeline_config = yaml.safe_load(pipeline_path.read_text())

    sim_pipeline = make_sim_pipeline(model_id, pipeline_config, pipe_save_dir, fix_state, no_effects)


    num_steps = 1000
    num_samples = 1
    num_runs = 5
    for factor_name in ["Extraversion", "Conscientiousness", "Openness to Experience", "Agreeableness", "Honesty-Humility"]:
        _save_dir = sim_save_dir / factor_name / f"num_steps{num_steps}"
        eval_pipeline(sim_pipeline, factor_name=factor_name, num_sim_steps=num_steps, num_samples=num_samples, num_runs=num_runs, save_dir=_save_dir)
    
    # run simulation
    # personality = sample_hexaco_personality()
    # init_state = sim_pipeline.init_state_fn(personality)

    # result = simulate(sim_pipeline, init_state, num_steps=100)

    # scores, inferred_personality, inferred_factors = analyze(
    #     sim_pipeline,
    #     result,
    #     personality,
    #     sim_pipeline.eval_score,
    #     sim_save_dir,
    #     init_state=init_state
    # )

    # for factor_name in ["Extraversion", "Conscientiousness", "Openness to Experience", "Agreeableness", "Honesty-Humility"]:
    for factor_name in ["Extraversion"]:
        eval_single_factor(sim_pipeline, factor_name, 2, sim_save_dir / factor_name / "default_zero", default_value='zero')

