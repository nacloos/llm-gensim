from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Callable
import numpy as np
import yaml

from llm_gensim.llm_utils import parse_output, llm_model
from llm_gensim.constants import hexaco_questions_path, hexaco_data_path

save_dir = Path(__file__).parent / "results" / "generate_pipeline"


@dataclass
class GenPipelineStep:
    prompt: str
    output_name: str
    output_type: str


@dataclass
class GenPipeline:
    llm_name: str
    llm: Callable
    save_dir: Path
    context_vars: dict[str, str]  # strings, use in prompts
    outputs: dict[str, Any]  # parsed python objects
    steps: list[GenPipelineStep] = None
    system_prompt: str = None


def gen_pipeline_step(pipeline: GenPipeline, step: GenPipelineStep, load=True):
    """
    Args:
        pipeline: The pipeline object
        step: The step to execute
        load: if True, load the answer from the log_dir if it exists
    """
    _save_dir = pipeline.save_dir / step.output_name
    _save_dir.mkdir(parents=True, exist_ok=True)

    if load and (_save_dir / "answer.md").exists():
        print(f"Loading {step.output_name}")
        # Load outputs
        with open(_save_dir / "answer.md", "r") as f:
            answer = f.read()

        text_output, parsed_output = parse_output(answer, output_type=step.output_type, output_name=step.output_name)

        pipeline.context_vars[step.output_name] = text_output
        pipeline.outputs[step.output_name] = parsed_output
        return

    prompt = step.prompt.format(**pipeline.context_vars)

    # save prompt
    with open(_save_dir / "prompt.md", "w") as f:
        f.write(prompt)

    print(f"Generating {step.output_name}")
    answer = pipeline.llm(prompt)

    # save answer
    with open(_save_dir / "answer.md", "w") as f:
        f.write(answer)

    text_output, parsed_output = parse_output(answer, output_type=step.output_type, output_name=step.output_name)

    # save output
    # TODO: if output_type is python, save as .py file
    # with open(_save_dir / f"output.{step.output_type}", "w") as f:
    #     f.write(text_output)

    # Update the pipeline outputs
    pipeline.context_vars[step.output_name] = text_output
    pipeline.outputs[step.output_name] = parsed_output


@dataclass
class Env:
    states: list
    actions: list
    effects: dict
    # autonomous: dict

@dataclass
class SimPipeline:
    # env: Env
    actions: list
    effects: dict
    states: list
    init_state_fn: Callable
    policy: Callable
    eval_score: Callable


def make_sim_pipeline(model_id, gen_pipeline_config, save_dir, fix_state=False, no_effects=False) -> SimPipeline:
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
            GenPipelineStep(**step) for step in gen_pipeline_config["steps"]
        ],
        save_dir=save_dir,
        context_vars=context_vars,
        outputs=outputs
    )

    print(f"Generating pipeline with {model_id}")
    for step in pipeline.steps:
        gen_pipeline_step(pipeline, step)

    if fix_state:
        # identity function since agent state = personality (normalized)
        init_state_fn = lambda personality: {k: (v-1)/4*100 for k, v in personality.items()}
    else:
        init_state_fn = pipeline.outputs["init_state"]

    # env = Env(
    #     states=pipeline.outputs["states"],
    #     actions=pipeline.outputs["actions"],
    #     effects=pipeline.outputs["effects"],
    # )
    policy = pipeline.outputs["policy"]
    eval_score = pipeline.outputs["eval"]

    return SimPipeline(
        actions=pipeline.outputs["actions"],
        effects=pipeline.outputs["effects"],
        states=pipeline.outputs["states"],
        init_state_fn=init_state_fn,
        policy=policy,
        eval_score=eval_score
    )


def agent_step(state, policy, env: Env):
    # TODO: temp fix
    if isinstance(policy, Callable):
        return policy(state)
    else:
        assert isinstance(policy, list), "Policy must be a list of conditions and activities"

    possible_activities = []
    for condition, activity in policy:
        try:
            condition(state)
        except Exception as e:
            print(f"Condition {condition} failed with error: {e}")
            continue

        if condition(state):
            possible_activities.append(activity)

    if len(possible_activities) > 0:
        action = random.choices(possible_activities, k=1)[0]
    else:
        action = random.choices(env.actions, k=1)[0]
    return action


def env_step(state, action, env: Env):
    if action not in env.effects:
        # print(f"Action {action} not found in effects")
        return state

    # autonomous dynamics
    # for var in state.keys():
    #     if var in env.autonomous:
    #         state[var] += env.autonomous[var]

    # actions
    for key, delta in env.effects[action].items():
        if key not in state:
            continue
        state[key] = state[key] + delta

    # bounds
    for var in state.keys():
        state[var] = max(min(state[var], 100), 0)

    return state


@dataclass
class SimResult:
    personality: dict
    init_state: dict
    states: list
    actions: list
    average_states: dict
    average_actions: dict


def simulate(sim_pipeline: SimPipeline, personality, num_steps=100) -> SimResult:
    env = Env(
        actions=sim_pipeline.actions,
        effects=sim_pipeline.effects,
        states=sim_pipeline.states
    )

    policy = sim_pipeline.policy

    init_state = sim_pipeline.init_state_fn(personality)

    state = init_state.copy()
    states = [state]
    actions = []
    for _ in range(num_steps):
        action = agent_step(state, policy, env)
        state = env_step(state, action, env)
        states.append(state.copy())
        actions.append(action)

    average_states = {var: np.mean([state[var] for state in states]) for var in env.states}
    average_actions = {activity: np.mean([action == activity for action in actions]) for activity in env.actions}

    return SimResult(
        personality=personality,
        init_state=init_state,
        states=states,
        actions=actions,
        average_states=average_states,
        average_actions=average_actions
    )


def sample_hexaco_personality(factors=None, facets=None):
    with open(hexaco_data_path, "r") as f:
        hexaco_data = yaml.safe_load(f)

    data = hexaco_data["factors"]

    if factors is not None:
        for name, score in factors.items():
            for item in data:
                if item["category"] == name:
                    item["mean"] = score
                    item["sd"] = 0

    assert facets is None, "Facets are not supported yet"

    means = np.array([factor["mean"] for factor in data])
    stds = np.array([factor["sd"] for factor in data])
    scores = np.random.normal(means, stds, size=(len(means)))
    # truncate between 1 and 5
    scores = np.clip(scores, 1, 5)
    factor_scores = {factor["name"]: scores[i] for i, factor in enumerate(data)}

    return factor_scores


if __name__ == "__main__":
    # import analysis here to avoid circular import
    from llm_gensim.analysis import eval_random_personality, eval_single_factor

    # model_id = "gemini-1.5-flash-latest"
    # model_id = "gemini-1.5-flash-exp-0827"
    # model_id = "gemini-1.5-pro-latest"
    model_id = "claude-3-5-sonnet-20240620"

    # pipeline_name = "pipeline"
    # pipeline_name = "fixed_hexaco_state"
    # pipeline_name = "fixed_hexaco_state-no_effects"
    pipeline_name = "fixed_hexaco_state-no_effects-action_comments"
    # pipeline_name = "hexaco_state-free_activities-no_effects"
    # pipeline_name = "hexaco_state-free_activities_200-no_effects"
    # pipeline_name = "hexaco_state-free_activities-no_effects-test²"
    batch_name = "batch1"
    # batch_name = "batch2"

    fix_state = True if "hexaco_state" in pipeline_name else False
    no_effects = True if "no_effects" in pipeline_name else False

    _save_dir = save_dir / pipeline_name / model_id / batch_name
    # _save_dir = save_dir / "test_free_activities" / pipeline_name / model_id / batch_name

    pipe_save_dir = _save_dir / "gen"
    sim_save_dir = _save_dir / "sim"

    pipeline_path = Path(__file__).parent / "configs" / "pipeline" / (pipeline_name + ".yaml")
    # pipeline_path = Path(__file__).parent / "configs" / "pipeline_free_activities" / (pipeline_name + ".yaml")
    pipeline_config = yaml.safe_load(pipeline_path.read_text())

    sim_pipeline = make_sim_pipeline(model_id, pipeline_config, pipe_save_dir, fix_state=fix_state, no_effects=no_effects)

    for factor_name in ["Extraversion", "Conscientiousness", "Openness to Experience", "Agreeableness", "Honesty-Humility"]:
        eval_single_factor(sim_pipeline, factor_name, 2, sim_save_dir / factor_name, default_value='zero')

    # eval_single_factor(sim_pipeline, "Extraversion", 2, sim_save_dir / "Extraversion" / "default_zero", default_value='zero')
    # eval_single_factor(sim_pipeline, "Extraversion", 2, sim_save_dir / "Extraversion")
    eval_random_personality(sim_pipeline, sim_save_dir / "random_personality", num_runs=10)