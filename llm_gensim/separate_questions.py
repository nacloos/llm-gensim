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


from llm_gensim.analysis import analyze, eval_single_factor, eval_random_personality, interpret_hexaco_personality
from llm_gensim.llm_utils import parse_output, llm_model
from llm_gensim.generate_sim import gen_pipeline_step, GenPipeline, GenPipelineStep, SimPipeline, Env, sample_hexaco_personality, simulate


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


def make_eval_code(gen_pipeline: GenPipeline, step: GenPipelineStep, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "user", "content": step.prompt.format(**gen_pipeline.context_vars)}
    ]

    question_blocks = []
    i = 1
    while len(question_blocks) < 100:
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
        blocks = re.findall(r'<q\d+>(.*?)</q\d+>', answer, re.DOTALL)
        question_blocks.extend(blocks)

        if len(question_blocks) < 100:
            messages += [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Continue implementing the code."}
            ]
            i += 1

    # concatenate question blocks and write in python file
    questions_code = "\n\n".join(question_blocks)
    # indent question code
    questions_code = "\n".join([f"    {line}" for line in questions_code.split("\n")])

    score_list = "[" + ", ".join([f"eval_q{i}(activities)" for i in range(1, 101)]) + "]"
    # place the question code inside a function eval_score
    code = f"""
def eval_score(activities: list[str]) -> list[int]:
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



def make_sim_pipeline(model_id, gen_pipeline_config, save_dir, fix_state=True, no_effects=True) -> SimPipeline:
    llm = llm_model(model_id)

    # load data
    with open(hexaco_questions_path, "r") as f:
        questions = f.read()

    with open(hexaco_data_path, "r") as f:
        hexaco_data = yaml.safe_load(f)

    factors = [d["name"] for d in hexaco_data["factors"]]

    context_vars = {
        "questions": questions,
        "personality_factors": factors
    }
    outputs = {}

    if fix_state: 
        # agent state = personality
        outputs['states'] = context_vars['states'] = factors

    if no_effects:
        outputs['effects'] = {}

    # make pipeline
    pipeline = GenPipeline(
        llm_name=model_id,
        llm=llm,
        steps=[
            GenPipelineStep(**step) for k, step in gen_pipeline_config["steps"].items() if k != "eval_score"
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

    step = GenPipelineStep(**gen_pipeline_config["steps"]["eval_score"])
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


def eval_pipeline(sim_pipeline: SimPipeline, factor_name: str, num_sim_steps, num_samples, num_runs, save_dir):
    data = yaml.safe_load(open(hexaco_data_path, "r"))["factors"]

    def _set_factor(personality, factor_name, value):
        personality = personality.copy()
        for item in data:
            if item["category"] == factor_name:
                personality[item["name"]] = value
        return personality

    def _run(personality):
        res = simulate(sim_pipeline, personality, num_steps=num_sim_steps)
    
        scores = sim_pipeline.eval_score(res.actions)
        inferred_personality = interpret_hexaco_personality(scores)
        return inferred_personality
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
            inferred_personality_low = _run(personality_low)
            inferred_personality_high = _run(personality_high)
            inferred_personalities_low.append(inferred_personality_low)
            inferred_personalities_high.append(inferred_personality_high)

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
    pipeline_name = "hexaco_state-free_activities-separate_questions"
    batch_name = "batch3"

    fix_state = True if "hexaco_state" in pipeline_name else False
    no_effects = True if "no_effects" in pipeline_name else False

    _save_dir = save_dir / pipeline_name / model_id / batch_name

    pipe_save_dir = _save_dir / "gen"
    sim_save_dir = _save_dir / "sim"

    pipeline_path = Path(__file__).parent / "configs" / "pipeline_free_activities" / (pipeline_name + ".yaml")
    pipeline_config = yaml.safe_load(pipeline_path.read_text())

    sim_pipeline = make_sim_pipeline(model_id, pipeline_config, pipe_save_dir, fix_state, no_effects)


    num_steps = 100
    for factor_name in ["Extraversion", "Conscientiousness", "Openness to Experience", "Agreeableness", "Honesty-Humility"]:
        _save_dir = sim_save_dir / factor_name / f"num_steps{num_steps}"
        eval_pipeline(sim_pipeline, factor_name=factor_name, num_sim_steps=num_steps, num_samples=10, num_runs=50, save_dir=_save_dir)
    
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
    # for factor_name in ["Extraversion"]:
    #     eval_single_factor(sim_pipeline, factor_name, 2, sim_save_dir / factor_name / "default_mean", default_value='mean')

